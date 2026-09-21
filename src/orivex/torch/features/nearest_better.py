"""Tensor-native nearest-better clustering: the exact graph and its differentiable surrogate.

Two independent groups live here.

``nbc.*`` reproduces :mod:`orivex.features.nearest_better` exactly. The graph is selected under
``no_grad`` and the selected edge lengths are then recomputed from ``x``, so the distance features
carry a gradient with respect to the design while the selection stays a discrete decision. With a
fixed design they are locally constant in ``y``: the ordering, not the spacing, picks the edges.
``nbc.nb_fitness.cor`` is the exception, differentiating through ``y`` at a locally fixed graph.

``nbc_soft.*`` implements the first-success relaxation documented in
:mod:`orivex.features.nearest_better_soft`, whose weights depend smoothly on ``y``. Use it to
obtain training directions and the exact group to report results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from orivex.capabilities import FeatureCapability
from orivex.engine import FeatureUnavailable
from orivex.features.nearest_better import FEATURES as NUMPY_FEATURES
from orivex.features.nearest_better import INTERMEDIATES as NUMPY_INTERMEDIATES
from orivex.features.nearest_better_soft import DEFINITION as SOFT_DEFINITION
from orivex.features.nearest_better_soft import FEATURES as NUMPY_SOFT_FEATURES
from orivex.features.nearest_better_soft import GRAPH as SOFT_GRAPH
from orivex.features.nearest_better_soft import INTERMEDIATES as NUMPY_SOFT_INTERMEDIATES
from orivex.torch.engine import (
    TensorComputationContext,
    TensorFeatureDefinition,
    TensorIntermediateDefinition,
)

GRAPH = NUMPY_INTERMEDIATES[0].spec.name


@dataclass(frozen=True, slots=True)
class TensorNearestBetterGraph:
    nearest_distances: torch.Tensor
    nearest_better_distances: torch.Tensor
    indegrees: torch.Tensor
    fitness: torch.Tensor


@dataclass(frozen=True, slots=True)
class TensorSoftNearestBetterGraph:
    nearest_distances: torch.Tensor
    soft_nearest_better_distances: torch.Tensor
    soft_indegrees: torch.Tensor
    no_better_mass: torch.Tensor
    gates: torch.Tensor
    fitness: torch.Tensor


def _require_pairs(context: TensorComputationContext, description: str) -> int:
    observations = context.sample.x.shape[0]
    if observations < 2:
        raise FeatureUnavailable(f"{description} require at least 2 observations")
    return observations


def _masked_distances(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise distances plus a detached copy whose diagonal is infinite.

    The live matrix keeps the design gradient; the masked copy only drives discrete decisions.
    """
    distances = torch.cdist(x, x)
    masked = distances.detach().clone()
    masked.fill_diagonal_(float("inf"))
    return distances, masked


def nearest_better_graph(context: TensorComputationContext) -> TensorNearestBetterGraph:
    observations = _require_pairs(context, "nearest-better features")
    x = context.sample.x
    y = context.y
    distances, masked = _masked_distances(x)
    rows = torch.arange(observations, device=x.device)

    with torch.no_grad():
        detached_y = y.detach()
        infinite = torch.full_like(masked, float("inf"))
        strictly_better = detached_y[None, :] < detached_y[:, None]
        candidates = torch.where(strictly_better, masked, infinite)
        indices = torch.argmin(candidates, dim=1)
        found = torch.isfinite(candidates[rows, indices])

        # Equal-fitness candidates are the documented fallback of the exact implementation; the
        # diagonal must be excluded explicitly because y_i == y_i always holds.
        equal = detached_y[None, :] == detached_y[:, None]
        equal.fill_diagonal_(False)
        equal_candidates = torch.where(equal, masked, infinite)
        equal_indices = torch.argmin(equal_candidates, dim=1)
        equal_found = torch.isfinite(equal_candidates[rows, equal_indices])

        use_equal = ~found & equal_found
        indices = torch.where(use_equal, equal_indices, indices)
        found = found | use_equal
        nearest_indices = torch.argmin(masked, dim=1)

    nearest = distances[rows, nearest_indices]
    selected = distances[rows, indices]
    # A unique best observation has no outgoing edge and reuses its nearest-neighbour distance.
    nearest_better = torch.where(found, selected, nearest)
    indegrees = torch.zeros_like(nearest).scatter_add(
        0,
        indices[found],
        torch.ones_like(nearest[found]),
    )
    return TensorNearestBetterGraph(nearest, nearest_better, indegrees, y)


def _gate_values(context: TensorComputationContext, y: torch.Tensor) -> torch.Tensor:
    options = context.options["nbc_soft"]
    gaps = torch.diff(torch.sort(y).values)
    positive = gaps[gaps.detach() > 0]
    if positive.numel() == 0:
        raise FeatureUnavailable("soft nearest-better gates are undefined for a constant objective")
    # The median consecutive gap is the resolution the ranks have to separate. It stays attached
    # on purpose: it makes the gates invariant to a positive rescaling of y, and an invariant
    # function has no gradient along that direction. Detaching it would manufacture one and let
    # training reduce the loss by rescaling the objective alone.
    scale = positive.median()
    differences = (y[:, None] - y[None, :]) / (float(options["rank_temperature"]) * scale)
    ranks = torch.sigmoid(differences).sum(dim=1) - 0.5
    return ranks / (y.shape[0] - 1)


def soft_nearest_better_graph(
    context: TensorComputationContext,
) -> TensorSoftNearestBetterGraph:
    observations = _require_pairs(context, "soft nearest-better features")
    x = context.sample.x
    y = context.y
    temperature = float(context.options["nbc_soft"]["temperature"])
    gates = _gate_values(context, y)
    distances, masked = _masked_distances(x)

    with torch.no_grad():
        order = masked.argsort(stable=True, dim=1)[:, : observations - 1]
    ordered = distances.gather(1, order)
    nearest = ordered[:, 0]

    z = (gates[:, None] - gates[order]) / temperature
    # Log-space survival products avoid subtracting a nearly-one sigmoid from one.
    survival = F.logsigmoid(-z).cumsum(dim=1)
    prefix = torch.cat([torch.zeros_like(survival[:, :1]), survival[:, :-1]], dim=1)
    weights = (F.logsigmoid(z) + prefix).exp()
    no_better = survival[:, -1].exp()

    soft_better = (weights * ordered).sum(dim=1) + no_better * nearest
    indegrees = torch.zeros_like(nearest).scatter_add(
        0,
        order.reshape(-1),
        weights.reshape(-1),
    )
    return TensorSoftNearestBetterGraph(
        nearest,
        soft_better,
        indegrees,
        no_better,
        gates,
        y,
    )


def _sample_std(values: torch.Tensor, description: str) -> torch.Tensor:
    if values.numel() < 2:
        raise FeatureUnavailable(f"{description} requires at least 2 finite values")
    return torch.std(values, correction=1)


def _nonzero(value: torch.Tensor, description: str) -> torch.Tensor:
    if bool(value.detach().eq(0).item()):
        raise FeatureUnavailable(f"{description} has zero variation")
    return value


def _correlation(left: torch.Tensor, right: torch.Tensor, description: str) -> torch.Tensor:
    if left.numel() < 2:
        raise FeatureUnavailable(f"{description} requires at least 2 finite pairs")
    for values in (left, right):
        if bool((torch.amax(values) - torch.amin(values)).detach().eq(0).item()):
            raise FeatureUnavailable(f"{description} is undefined for a constant input")
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    return torch.dot(centered_left, centered_right) / (
        torch.linalg.vector_norm(centered_left) * torch.linalg.vector_norm(centered_right)
    )


def _sd_ratio(nearest: torch.Tensor, better: torch.Tensor, label: str) -> torch.Tensor:
    numerator = _sample_std(nearest, "nearest-neighbour standard deviation")
    denominator = _nonzero(
        _sample_std(better, f"{label} standard deviation"),
        f"{label} standard deviation",
    )
    return numerator / denominator


def _mean_ratio(nearest: torch.Tensor, better: torch.Tensor, label: str) -> torch.Tensor:
    denominator = better.mean()
    if bool(denominator.detach().eq(0).item()):
        raise FeatureUnavailable(f"{label} mean distance is zero")
    return nearest.mean() / denominator


def _ratio_coefficient_of_variation(
    nearest: torch.Tensor, better: torch.Tensor, label: str
) -> torch.Tensor:
    ratios = nearest / better
    ratios = ratios[torch.isfinite(ratios)]
    if ratios.numel() == 0 or bool(ratios.mean().detach().eq(0).item()):
        raise FeatureUnavailable(f"finite {label} distance ratios have zero mean")
    return _sample_std(ratios, f"{label} distance ratio") / ratios.mean()


def _exact(context: TensorComputationContext) -> TensorNearestBetterGraph:
    value = context.intermediate(GRAPH)
    if not isinstance(value, TensorNearestBetterGraph):
        raise TypeError(f"{GRAPH} has an invalid runtime type")
    return value


def _soft(context: TensorComputationContext) -> TensorSoftNearestBetterGraph:
    value = context.intermediate(SOFT_GRAPH)
    if not isinstance(value, TensorSoftNearestBetterGraph):
        raise TypeError(f"{SOFT_GRAPH} has an invalid runtime type")
    return value


EXACT_LABEL = "nearest-better"
SOFT_LABEL = "soft nearest-better"

_EXACT_CALCULATORS = {
    "nbc.nn_nb.sd_ratio": lambda c: _sd_ratio(
        _exact(c).nearest_distances, _exact(c).nearest_better_distances, EXACT_LABEL
    ),
    "nbc.nn_nb.mean_ratio": lambda c: _mean_ratio(
        _exact(c).nearest_distances, _exact(c).nearest_better_distances, EXACT_LABEL
    ),
    "nbc.nn_nb.cor": lambda c: _correlation(
        _exact(c).nearest_distances,
        _exact(c).nearest_better_distances,
        "nearest/nearest-better correlation",
    ),
    "nbc.dist_ratio.coeff_var": lambda c: _ratio_coefficient_of_variation(
        _exact(c).nearest_distances, _exact(c).nearest_better_distances, "nearest"
    ),
    "nbc.nb_fitness.cor": lambda c: _correlation(
        _exact(c).indegrees,
        _exact(c).fitness,
        "nearest-better indegree/fitness correlation",
    ),
}

_SOFT_CALCULATORS = {
    "nbc_soft.nn_nb.sd_ratio": lambda c: _sd_ratio(
        _soft(c).nearest_distances, _soft(c).soft_nearest_better_distances, SOFT_LABEL
    ),
    "nbc_soft.nn_nb.mean_ratio": lambda c: _mean_ratio(
        _soft(c).nearest_distances, _soft(c).soft_nearest_better_distances, SOFT_LABEL
    ),
    "nbc_soft.nn_nb.cor": lambda c: _correlation(
        _soft(c).nearest_distances,
        _soft(c).soft_nearest_better_distances,
        "nearest/soft nearest-better correlation",
    ),
    "nbc_soft.dist_ratio.coeff_var": lambda c: _ratio_coefficient_of_variation(
        _soft(c).nearest_distances, _soft(c).soft_nearest_better_distances, "soft nearest"
    ),
    "nbc_soft.nb_fitness.cor": lambda c: _correlation(
        _soft(c).soft_indegrees,
        _soft(c).fitness,
        "soft nearest-better indegree/fitness correlation",
    ),
}

INTERMEDIATES = (
    TensorIntermediateDefinition(NUMPY_INTERMEDIATES[0].spec, nearest_better_graph),
    TensorIntermediateDefinition(NUMPY_SOFT_INTERMEDIATES[0].spec, soft_nearest_better_graph),
)

FEATURES = tuple(
    TensorFeatureDefinition(definition.spec, _EXACT_CALCULATORS[definition.spec.name])
    for definition in NUMPY_FEATURES
) + tuple(
    TensorFeatureDefinition(definition.spec, _SOFT_CALCULATORS[definition.spec.name])
    for definition in NUMPY_SOFT_FEATURES
)

_DEVICES = ("cpu", "cuda", "mps")
_DTYPES = ("float32", "float64")
_DENSE_NOTE = "Dense O(n^2) pairwise work and storage; the NumPy backend blocks its memory."

_EXACT_NOTES = (
    "The graph is selected under no_grad; selected edge lengths keep the design gradient.",
    "Locally constant in y for a fixed design: only the fitness ordering selects edges.",
    _DENSE_NOTE,
)

CAPABILITIES = tuple(
    FeatureCapability(
        feature_name=definition.spec.name,
        backend="torch",
        autograd="piecewise",
        devices=_DEVICES,
        dtypes=_DTYPES,
        notes=(
            _EXACT_NOTES
            if definition.spec.definition != SOFT_DEFINITION
            else (
                "Differentiable in y through the gates at a fixed temperature.",
                "Surrogate values are temperature dependent; report the exact nbc group.",
                "Smooth in y away from the kinks of the median-gap rank scale.",
                _DENSE_NOTE,
            )
        )
        + (
            ("Differentiates through y at a locally fixed graph.",)
            if definition.spec.name == "nbc.nb_fitness.cor"
            else ()
        ),
    )
    for definition in FEATURES
)
