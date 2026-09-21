"""Differentiable surrogates for the nearest-better clustering features.

These are **not** the published NBC features. They replace the discrete "first strictly better
neighbour" search with a smooth first-success distribution over distance-ordered candidates, so
that the resulting statistics have a useful derivative with respect to the objective values. The
exact features live in :mod:`orivex.features.nearest_better` and remain the only ones that should
be reported as NBC.

Construction, for each observation ``i`` and its candidates ordered by increasing distance:

* gate ``p_im = sigmoid((g_i - g_{j_im}) / tau)`` decides whether candidate ``m`` counts as better,
  where ``g`` is the scale-free soft rank of the objectives (see :func:`gate_values`);
* weight ``w_im = p_im * prod_{l<m} (1 - p_il)`` selects the *first* passing candidate;
* leftover mass ``q_i = prod_m (1 - p_im)`` represents "no better neighbour exists" and reuses the
  nearest-neighbour distance, matching the replacement convention of the exact implementation.

``sum_j w_ij + q_i = 1`` holds exactly, and every soft quantity converges to its exact counterpart
as ``tau -> 0`` for distinct objective values. Equal objective values are the documented gap: the
gate is ``1/2`` at every temperature, so plateaus do not reproduce the exact equal-fitness
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from orivex.engine import (
    ComputationContext,
    FeatureCalculator,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
)
from orivex.features.nearest_better import (
    BLOCK_MEMORY_BYTES,
    REFERENCE,
    _correlation,
    _nonzero_denominator,
    _sample_std,
)
from orivex.planner import IntermediateSpec
from orivex.specs import (
    CostModel,
    CostTier,
    FeatureSpec,
    InputRequirement,
    InvarianceBehavior,
    InvarianceClaim,
    MetricKind,
    Reference,
    Transformation,
)

GRAPH = "nbc_soft.graph"
DEFINITION = "soft-first-better-v1"

SURROGATE_REFERENCE = Reference(
    citation="orivex differentiable NBC surrogate (soft-first-better-v1)",
    url="https://github.com/WojtAcht/orivex",
)


@dataclass(frozen=True, slots=True)
class SoftNearestBetterGraph:
    """Distance-ordered first-success relaxation of the nearest-better graph."""

    nearest_distances: np.ndarray
    soft_nearest_better_distances: np.ndarray
    soft_indegrees: np.ndarray
    no_better_mass: np.ndarray
    gates: np.ndarray
    fitness: np.ndarray


def _log_sigmoid(values: np.ndarray) -> np.ndarray:
    """``log(sigmoid(v))`` without forming ``exp`` of a large positive argument."""
    return -np.logaddexp(0.0, -values)


def _robust_scale(y: np.ndarray) -> float:
    """Median positive gap between consecutive sorted objectives.

    This is the resolution the soft ranks actually have to separate, so it is a better scale than
    a spread statistic: an interquartile range is set by the bulk of the distribution and leaves
    the crowded end of a heavy-tailed objective unresolved. It is affine-equivariant, which keeps
    the gates invariant to a positive rescaling of y, and positive whenever y is not constant.
    """
    gaps = np.diff(np.sort(y))
    positive = gaps[gaps > 0]
    if positive.size == 0:
        raise FeatureUnavailable("soft nearest-better gates are undefined for a constant objective")
    return float(np.median(positive))


def gate_values(y: np.ndarray, rank_temperature: float) -> np.ndarray:
    """Map objectives to ``[0, 1]`` gate coordinates through a pairwise soft rank.

    ``g_i = mean_j sigmoid((y_i - y_j) / (t * s))``, with ``s`` the median gap between consecutive
    sorted objectives. Spacing then tends towards a uniform ``1 / (n - 1)`` whatever the shape of
    the objective distribution, which is what keeps a single gate temperature usable across
    BBOB-scale dynamic ranges. ``rank_temperature`` is measured in typical objective gaps, so
    pairs separated by much less than one gap stay unresolved.

    Plain min-max normalization was measured as the alternative and is not offered: it keeps the
    original spacing, so a heavy upper tail crowds most of the sample into a tiny interval. On
    BBOB d=2 with n=500 it diverged from the exact features 2.4 to 7.8 times further than the
    soft rank at every temperature tested.
    """
    scale = _robust_scale(y)
    observations = y.size
    ranks = np.empty(observations, dtype=np.float64)
    rows_per_block = max(1, min(observations, BLOCK_MEMORY_BYTES // (8 * observations)))
    for start in range(0, observations, rows_per_block):
        stop = min(start + rows_per_block, observations)
        differences = (y[start:stop, None] - y[None, :]) / (rank_temperature * scale)
        # The self term contributes sigmoid(0) = 1/2 and is removed instead of masked.
        ranks[start:stop] = np.sum(expit(differences), axis=1) - 0.5
    return ranks / (observations - 1)


def soft_nearest_better_graph(context: ComputationContext) -> SoftNearestBetterGraph:
    x = context.sample.x
    y = context.y
    observations = x.shape[0]
    if observations < 2:
        raise FeatureUnavailable("soft nearest-better features require at least 2 observations")

    options = context.options["nbc_soft"]
    temperature = float(options["temperature"])
    gates = gate_values(y, float(options["rank_temperature"]))

    nearest = np.empty(observations, dtype=np.float64)
    soft_better = np.empty(observations, dtype=np.float64)
    no_better = np.empty(observations, dtype=np.float64)
    indegrees = np.zeros(observations, dtype=np.float64)
    rows_per_block = max(1, min(observations, BLOCK_MEMORY_BYTES // (8 * observations)))

    for start in range(0, observations, rows_per_block):
        stop = min(start + rows_per_block, observations)
        distances = cdist(x[start:stop], x, metric="euclidean")
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        # Stable ordering reproduces the exact implementation's "lowest observation index" tie
        # break; the infinite self-distance always sorts last, so dropping it removes self-pairs.
        order = np.argsort(distances, axis=1, kind="stable")[:, :-1]
        ordered = np.take_along_axis(distances, order, axis=1)
        nearest[start:stop] = ordered[:, 0]

        z = (gates[start:stop, None] - gates[order]) / temperature
        survival = np.cumsum(_log_sigmoid(-z), axis=1)
        prefix = np.concatenate([np.zeros((z.shape[0], 1)), survival[:, :-1]], axis=1)
        weights = np.exp(_log_sigmoid(z) + prefix)
        block_no_better = np.exp(survival[:, -1])

        no_better[start:stop] = block_no_better
        soft_better[start:stop] = (
            np.sum(weights * ordered, axis=1) + block_no_better * nearest[start:stop]
        )
        indegrees += np.bincount(
            order.reshape(-1),
            weights=weights.reshape(-1),
            minlength=observations,
        )

    for array in (nearest, soft_better, indegrees, no_better, gates):
        array.flags.writeable = False
    return SoftNearestBetterGraph(nearest, soft_better, indegrees, no_better, gates, y)


def _graph(context: ComputationContext) -> SoftNearestBetterGraph:
    value = context.intermediate(GRAPH)
    if not isinstance(value, SoftNearestBetterGraph):
        raise TypeError(f"{GRAPH} has an invalid runtime type")
    return value


def sd_ratio(context: ComputationContext) -> float:
    graph = _graph(context)
    numerator = _sample_std(graph.nearest_distances, "nearest-neighbour standard deviation")
    denominator = _nonzero_denominator(
        _sample_std(graph.soft_nearest_better_distances, "soft nearest-better deviation"),
        "soft nearest-better standard deviation",
    )
    return numerator / denominator


def mean_ratio(context: ComputationContext) -> float:
    graph = _graph(context)
    denominator = float(np.mean(graph.soft_nearest_better_distances))
    if denominator == 0.0:
        raise FeatureUnavailable("soft nearest-better mean distance is zero")
    return float(np.mean(graph.nearest_distances)) / denominator


def distance_correlation(context: ComputationContext) -> float:
    graph = _graph(context)
    return _correlation(
        graph.nearest_distances,
        graph.soft_nearest_better_distances,
        "nearest/soft nearest-better correlation",
    )


def distance_ratio_coefficient_of_variation(context: ComputationContext) -> float:
    graph = _graph(context)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = graph.nearest_distances / graph.soft_nearest_better_distances
    ratios = ratios[np.isfinite(ratios)]
    mean = float(np.mean(ratios)) if ratios.size else 0.0
    if mean == 0.0:
        raise FeatureUnavailable("finite soft nearest-distance ratios have zero mean")
    return _sample_std(ratios, "soft nearest-distance ratio") / mean


def fitness_indegree_correlation(context: ComputationContext) -> float:
    graph = _graph(context)
    # Correlated against the canonical objectives, not the gate coordinates: Pearson correlation
    # is invariant to a positive affine change of y but not to the soft-rank transform, so using
    # the gates here would silently turn this into a rank correlation.
    return _correlation(
        graph.soft_indegrees,
        graph.fitness,
        "soft nearest-better indegree/fitness correlation",
    )


INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(GRAPH, (), frozenset({InputRequirement.X, InputRequirement.Y})),
        soft_nearest_better_graph,
    ),
)

COMMON_INVARIANCES = (
    InvarianceClaim(
        Transformation.ROW_PERMUTATION,
        InvarianceBehavior.INVARIANT,
        conditions="no exact equal-distance candidate ties",
    ),
    InvarianceClaim(Transformation.VARIABLE_PERMUTATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(Transformation.X_TRANSLATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(Transformation.X_ORTHOGONAL_ROTATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(Transformation.Y_TRANSLATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(
        Transformation.Y_POSITIVE_SCALING,
        InvarianceBehavior.INVARIANT,
        conditions="finite positive scale; the soft rank gates are affine-invariant",
    ),
    InvarianceClaim(
        Transformation.OBJECTIVE_SENSE_REVERSAL,
        InvarianceBehavior.INVARIANT,
        conditions="negate y and reverse the declared objective sense together",
    ),
)

NOTES = (
    "Surrogate feature: a temperature-dependent approximation, not the published NBC statistic.",
    "Converges to the exact nbc counterpart as the temperature approaches zero for distinct "
    "objective values.",
    "Equal objective values gate at 1/2 at every temperature, so plateaus do not reproduce the "
    "exact equal-fitness fallback.",
    "Finite temperature biases the value; compare against a target computed with the same "
    "temperature, sample size, and gate normalization.",
    "Controlled by the 'nbc_soft' options group: temperature and rank_temperature.",
    "Rank gate coordinates are spaced by 1/(n-1), so a fixed temperature blurs more "
    "candidates as n grows; calibrate it against the target sample size.",
)


def _feature(name: str, summary: str, calculator: FeatureCalculator) -> FeatureDefinition:
    return FeatureDefinition(
        FeatureSpec(
            name=name,
            summary=summary,
            group="nbc_soft",
            kind=MetricKind.LANDSCAPE,
            definition=DEFINITION,
            requirements=frozenset({InputRequirement.X, InputRequirement.Y}),
            intermediates=(GRAPH,),
            cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n^2 log n)", memory="O(n b)"),
            deterministic=True,
            invariances=COMMON_INVARIANCES,
            references=(REFERENCE, SURROGATE_REFERENCE),
            minimum_observations=2,
            notes=NOTES,
        ),
        calculator,
    )


FEATURES = (
    _feature(
        "nbc_soft.nn_nb.sd_ratio",
        "Differentiable surrogate of nbc.nn_nb.sd_ratio.",
        sd_ratio,
    ),
    _feature(
        "nbc_soft.nn_nb.mean_ratio",
        "Differentiable surrogate of nbc.nn_nb.mean_ratio.",
        mean_ratio,
    ),
    _feature(
        "nbc_soft.nn_nb.cor",
        "Differentiable surrogate of nbc.nn_nb.cor.",
        distance_correlation,
    ),
    _feature(
        "nbc_soft.dist_ratio.coeff_var",
        "Differentiable surrogate of nbc.dist_ratio.coeff_var.",
        distance_ratio_coefficient_of_variation,
    ),
    _feature(
        "nbc_soft.nb_fitness.cor",
        "Differentiable surrogate of nbc.nb_fitness.cor.",
        fitness_indegree_correlation,
    ),
)
