"""Exact blockwise nearest-better clustering features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from bflacco.engine import (
    ComputationContext,
    FeatureCalculator,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
)
from bflacco.planner import IntermediateSpec
from bflacco.specs import (
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

BLOCK_MEMORY_BYTES = 16 * 1024 * 1024

REFERENCE = Reference(
    citation="Kerschke et al. (2015), Detecting funnel structures by means of ELA",
    doi="10.1145/2739480.2754642",
)
R_REFERENCE = Reference(
    citation="flacco 1.8 nearest-better implementation",
    url="https://github.com/kerschke/flacco/blob/master/R/feature_nearest_better_clustering.R",
)


@dataclass(frozen=True, slots=True)
class NearestBetterGraph:
    nearest_distances: np.ndarray
    nearest_better_distances: np.ndarray
    nearest_better_indices: np.ndarray
    indegrees: np.ndarray
    fitness: np.ndarray
    message: str | None = None


def nearest_better_graph(context: ComputationContext) -> NearestBetterGraph:
    x = context.sample.x
    y = context.sample.minimization_y
    observations = x.shape[0]
    if observations < 2:
        empty = np.empty(0, dtype=np.float64)
        return NearestBetterGraph(
            empty,
            empty,
            np.empty(0, dtype=np.int64),
            empty,
            empty,
            "nearest-better features require at least 2 observations",
        )

    nearest = np.empty(observations, dtype=np.float64)
    nearest_better = np.empty(observations, dtype=np.float64)
    nearest_better_indices = np.full(observations, -1, dtype=np.int64)
    rows_per_block = max(1, min(observations, BLOCK_MEMORY_BYTES // (8 * observations)))

    for start in range(0, observations, rows_per_block):
        stop = min(start + rows_per_block, observations)
        distances = cdist(x[start:stop], x, metric="euclidean")
        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        distances[local_rows, global_rows] = np.inf
        nearest[start:stop] = np.min(distances, axis=1)

        strictly_better = y[None, :] < y[start:stop, None]
        candidate_distances = np.where(strictly_better, distances, np.inf)
        indices = np.argmin(candidate_distances, axis=1)
        values = candidate_distances[local_rows, indices]

        missing = ~np.isfinite(values)
        if np.any(missing):
            equal = y[None, :] == y[start:stop, None]
            equal[local_rows, global_rows] = False
            equal_distances = np.where(equal[missing], distances[missing], np.inf)
            equal_indices = np.argmin(equal_distances, axis=1)
            equal_values = equal_distances[np.arange(equal_indices.size), equal_indices]
            missing_rows = np.flatnonzero(missing)
            found_equal = np.isfinite(equal_values)
            indices[missing_rows[found_equal]] = equal_indices[found_equal]
            values[missing_rows[found_equal]] = equal_values[found_equal]

        found = np.isfinite(values)
        nearest_better_indices[global_rows[found]] = indices[found]
        nearest_better[global_rows] = values

    adjusted_better = nearest_better.copy()
    missing = nearest_better_indices < 0
    adjusted_better[missing] = nearest[missing]
    indegrees = np.bincount(
        nearest_better_indices[~missing],
        minlength=observations,
    ).astype(np.float64)
    for array in (nearest, adjusted_better, nearest_better_indices, indegrees):
        array.flags.writeable = False
    return NearestBetterGraph(
        nearest,
        adjusted_better,
        nearest_better_indices,
        indegrees,
        y,
    )


def _graph(context: ComputationContext) -> NearestBetterGraph:
    value = context.intermediate("nbc.graph")
    if not isinstance(value, NearestBetterGraph):
        raise TypeError("nbc.graph has an invalid runtime type")
    if value.message is not None:
        raise FeatureUnavailable(value.message)
    return value


def _sample_std(values: np.ndarray, description: str) -> float:
    if values.size < 2:
        raise FeatureUnavailable(f"{description} requires at least 2 finite values")
    result = float(np.std(values, ddof=1))
    if result == 0.0:
        raise FeatureUnavailable(f"{description} has zero variation")
    return result


def _correlation(left: np.ndarray, right: np.ndarray, description: str) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        raise FeatureUnavailable(f"{description} requires at least 2 finite pairs")
    left_values = left[finite]
    right_values = right[finite]
    if np.ptp(left_values) == 0.0 or np.ptp(right_values) == 0.0:
        raise FeatureUnavailable(f"{description} is undefined for a constant input")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def sd_ratio(context: ComputationContext) -> float:
    graph = _graph(context)
    numerator = _sample_std(graph.nearest_distances, "nearest-neighbour standard deviation")
    denominator = _sample_std(
        graph.nearest_better_distances,
        "nearest-better standard deviation",
    )
    return numerator / denominator


def mean_ratio(context: ComputationContext) -> float:
    graph = _graph(context)
    denominator = float(np.mean(graph.nearest_better_distances))
    if denominator == 0.0:
        raise FeatureUnavailable("nearest-better mean distance is zero")
    return float(np.mean(graph.nearest_distances)) / denominator


def distance_correlation(context: ComputationContext) -> float:
    graph = _graph(context)
    return _correlation(
        graph.nearest_distances,
        graph.nearest_better_distances,
        "nearest/nearest-better correlation",
    )


def distance_ratio_coefficient_of_variation(context: ComputationContext) -> float:
    graph = _graph(context)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = graph.nearest_distances / graph.nearest_better_distances
    ratios = ratios[np.isfinite(ratios)]
    mean = float(np.mean(ratios)) if ratios.size else 0.0
    if mean == 0.0:
        raise FeatureUnavailable("finite nearest-distance ratios have zero mean")
    return _sample_std(ratios, "nearest-distance ratio") / mean


def fitness_indegree_correlation(context: ComputationContext) -> float:
    graph = _graph(context)
    return _correlation(
        graph.indegrees,
        graph.fitness,
        "nearest-better indegree/fitness correlation",
    )


INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(
            "nbc.graph",
            (),
            frozenset({InputRequirement.X, InputRequirement.Y}),
        ),
        nearest_better_graph,
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
        conditions="finite positive scale",
    ),
    InvarianceClaim(
        Transformation.OBJECTIVE_SENSE_REVERSAL,
        InvarianceBehavior.INVARIANT,
        conditions="negate y and reverse the declared objective sense together",
    ),
)


def _feature(name: str, summary: str, calculator: FeatureCalculator) -> FeatureDefinition:
    return FeatureDefinition(
        FeatureSpec(
            name=name,
            summary=summary,
            group="nbc",
            kind=MetricKind.LANDSCAPE,
            definition="exact-euclidean-blockwise-v1",
            requirements=frozenset({InputRequirement.X, InputRequirement.Y}),
            intermediates=("nbc.graph",),
            cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n^2 d)", memory="O(n b)"),
            deterministic=True,
            invariances=COMMON_INVARIANCES,
            references=(REFERENCE, R_REFERENCE),
            legacy_names=(name,),
            minimum_observations=2,
            notes=(
                "Exact Euclidean search with bounded distance-block memory.",
                "Strictly better candidates are preferred; equal-fitness candidates are "
                "the fallback.",
                "Exact distance ties choose the lowest current observation index.",
            ),
        ),
        calculator,
    )


FEATURES = (
    _feature(
        "nbc.nn_nb.sd_ratio",
        "Ratio of nearest and nearest-better sample deviations.",
        sd_ratio,
    ),
    _feature(
        "nbc.nn_nb.mean_ratio",
        "Ratio of nearest and nearest-better mean distances.",
        mean_ratio,
    ),
    _feature(
        "nbc.nn_nb.cor",
        "Correlation of nearest and nearest-better distances.",
        distance_correlation,
    ),
    _feature(
        "nbc.dist_ratio.coeff_var",
        "Coefficient of variation of nearest/nearest-better distance ratios.",
        distance_ratio_coefficient_of_variation,
    ),
    _feature(
        "nbc.nb_fitness.cor",
        "Correlation of nearest-better indegree and canonical fitness.",
        fitness_indegree_correlation,
    ),
)
