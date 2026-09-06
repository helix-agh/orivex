"""Tensor-native fitness-distance statistics with piecewise gradients through selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from orivex.capabilities import FeatureCapability
from orivex.engine import FeatureUnavailable
from orivex.features.fitness_distance import DISTANCES, SELECTION, selected_observations
from orivex.features.fitness_distance import FEATURES as NUMPY_FEATURES
from orivex.features.fitness_distance import INTERMEDIATES as NUMPY_INTERMEDIATES
from orivex.torch.engine import (
    TensorComputationContext,
    TensorFeatureDefinition,
    TensorIntermediateDefinition,
)


@dataclass(frozen=True, slots=True)
class Selection:
    indices: torch.Tensor
    y: torch.Tensor


def select_best(context: TensorComputationContext) -> Selection:
    raw = context.sample.minimization_y
    count = selected_observations(
        raw.numel(), context.options["fitness_distance"]["proportion_of_best"]
    )
    if count == raw.numel():
        indices = torch.arange(raw.numel(), device=raw.device)
    else:
        threshold = torch.amax(torch.topk(raw, count, largest=False, sorted=False).values)
        mask = raw < threshold
        tied = torch.nonzero(raw == threshold).flatten()[: count - int(mask.sum().item())]
        mask[tied] = True
        indices = torch.nonzero(mask).flatten()
    return Selection(indices, context.y[indices])


def distances(context: TensorComputationContext) -> torch.Tensor:
    selected = cast(Selection, context.intermediate(SELECTION))
    indices = selected.indices
    reference = indices[torch.argmin(context.sample.minimization_y[indices])]
    delta = context.sample.x[indices] - context.sample.x[reference]
    scale = torch.amax(torch.abs(delta), dim=1).detach()
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.linalg.vector_norm(delta / divisor[:, None], ord=2, dim=1) * divisor


def scaled_values(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Auxiliary transforms cancel analytically; detach them to avoid extrema derivatives.
    midpoint = (torch.amin(values) / 2 + torch.amax(values) / 2).detach()
    deviations = values - midpoint
    scale = torch.amax(torch.abs(deviations)).detach()
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    return deviations / divisor, divisor, midpoint


def stable_mean(values: torch.Tensor) -> torch.Tensor:
    scaled, scale, midpoint = scaled_values(values)
    return torch.mean(scaled) * scale + midpoint


def stable_std(values: torch.Tensor) -> torch.Tensor:
    scaled, scale, _ = scaled_values(values)
    return torch.std(scaled, correction=1) * scale


def fitness_std(context: TensorComputationContext) -> torch.Tensor:
    return stable_std(cast(Selection, context.intermediate(SELECTION)).y)


def fitness_mean(context: TensorComputationContext) -> torch.Tensor:
    return stable_mean(cast(Selection, context.intermediate(SELECTION)).y)


def distance_mean(context: TensorComputationContext) -> torch.Tensor:
    return stable_mean(cast(torch.Tensor, context.intermediate(DISTANCES)))


def distance_std(context: TensorComputationContext) -> torch.Tensor:
    return stable_std(cast(torch.Tensor, context.intermediate(DISTANCES)))


def fd_cov(context: TensorComputationContext) -> torch.Tensor:
    y, y_scale, _ = scaled_values(cast(Selection, context.intermediate(SELECTION)).y)
    d, d_scale, _ = scaled_values(cast(torch.Tensor, context.intermediate(DISTANCES)))
    return torch.mean((y - torch.mean(y)) * (d - torch.mean(d))) * y_scale * d_scale


def fd_correlation(context: TensorComputationContext) -> torch.Tensor:
    y, _, _ = scaled_values(cast(Selection, context.intermediate(SELECTION)).y)
    d, _, _ = scaled_values(cast(torch.Tensor, context.intermediate(DISTANCES)))
    y_std, d_std = torch.std(y, correction=1), torch.std(d, correction=1)
    if bool((y_std == 0) | (d_std == 0)):
        raise FeatureUnavailable(
            "fitness-distance correlation requires non-zero fitness and distance variance"
        )
    return torch.mean((y - torch.mean(y)) * (d - torch.mean(d))) / (y_std * d_std)


INTERMEDIATES = (
    TensorIntermediateDefinition(NUMPY_INTERMEDIATES[0].spec, select_best),
    TensorIntermediateDefinition(NUMPY_INTERMEDIATES[1].spec, distances),
)
_CALCULATORS = (fitness_std, fitness_mean, distance_mean, distance_std, fd_cov, fd_correlation)
FEATURES = tuple(
    TensorFeatureDefinition(definition.spec, calculate)
    for definition, calculate in zip(NUMPY_FEATURES, _CALCULATORS, strict=True)
)
CAPABILITIES = tuple(
    FeatureCapability(
        feature_name=definition.spec.name,
        backend="torch",
        autograd="piecewise",
        devices=("cpu", "cuda", "mps"),
        dtypes=("float32", "float64"),
        notes=(
            "Gradients flow through selected objectives and reference distances where applicable.",
            "Selection and reference ties choose the first original row; boundaries are nonsmooth.",
            "Zero deviations and coincident distances use a zero gradient convention.",
        ),
    )
    for definition in FEATURES
)
