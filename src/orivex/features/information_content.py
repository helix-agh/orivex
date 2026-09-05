"""Information-content features over a deterministic nearest-neighbour tour."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from orivex.engine import (
    ComputationContext,
    FeatureCalculator,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
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

EPSILON = np.insert(10.0 ** np.linspace(-5.0, 15.0, num=1_000), 0, 0.0)
EPSILON.flags.writeable = False
NEIGHBOURHOOD = 20
SETTLING_SENSITIVITY = 0.05
INFORMATION_SENSITIVITY = 0.5

REFERENCE = Reference(
    citation="Muñoz, Kirley, and Halgamuge (2015), ELA using information content",
    doi="10.1109/TEVC.2014.2302006",
)
R_REFERENCE = Reference(
    citation="flacco 1.8 information-content implementation",
    url="https://github.com/kerschke/flacco/blob/master/R/feature_information_content.R",
)


@dataclass(frozen=True, slots=True)
class SlopeSequence:
    values: np.ndarray
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolSchedule:
    """Compact description of how tour symbols change along the epsilon grid.

    Each slope switches to the zero symbol exactly once as epsilon grows, so the dense
    grid-by-slope matrix it used to be is fully described by the epsilon-zero sign of every
    slope and the first grid index at which that slope reads as zero.
    """

    signs: np.ndarray
    thresholds: np.ndarray
    count: int
    message: str | None = None


@dataclass(frozen=True, slots=True)
class Curve:
    values: np.ndarray
    message: str | None = None


def _aggregate_duplicate_points(context: ComputationContext) -> tuple[np.ndarray, np.ndarray]:
    x, inverse, counts = np.unique(
        context.sample.x,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    y_sums = np.zeros(x.shape[0], dtype=np.float64)
    np.add.at(y_sums, inverse, context.y)
    return x, y_sums / counts


def _closest(x: np.ndarray, candidates: np.ndarray, current: int) -> tuple[int, float]:
    """Lowest-index closest candidate under exactly recomputed Euclidean distances."""

    deltas = x[candidates] - x[current]
    exact = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
    minimum = np.min(exact)
    return int(np.min(candidates[exact == minimum])), float(minimum)


def _nearest_unvisited(
    x: np.ndarray,
    tree: KDTree,
    neighbour_distances: np.ndarray,
    neighbour_indices: np.ndarray,
    covered_radius: np.ndarray,
    current: int,
    visited: np.ndarray,
) -> tuple[int, float]:
    candidates = neighbour_indices[current]
    unvisited = ~visited[candidates]
    if unvisited.any():
        candidate_distances = neighbour_distances[current][unvisited]
        available = candidates[unvisited]
        distance = np.min(candidate_distances)
        radius = np.nextafter(float(distance), np.inf)
        if radius < covered_radius[current]:
            # Every point inside this radius is closer than the furthest stored neighbour and
            # therefore already listed, so the stored candidates are the complete tie set and
            # the range query below cannot add to it.
            return _closest(x, available[candidate_distances <= radius], current)
        tied = np.asarray(tree.query_ball_point(x[current], radius), dtype=np.int64)
        tied = tied[~visited[tied]]
        if tied.size:
            return _closest(x, tied, current)

    return _closest(x, np.flatnonzero(~visited), current)


def slope_sequence(context: ComputationContext) -> SlopeSequence:
    x, y = _aggregate_duplicate_points(context)
    if x.shape[0] < 3:
        return SlopeSequence(
            np.empty(0, dtype=np.float64),
            "information content requires at least 3 distinct decision points",
        )

    tree = KDTree(x)
    k = min(NEIGHBOURHOOD, x.shape[0])
    neighbour_distances, neighbour_indices = tree.query(x, k=k, workers=context.workers)
    if k == 1:
        neighbour_distances = neighbour_distances[:, None]
        neighbour_indices = neighbour_indices[:, None]
    covered_radius = neighbour_distances[:, -1]

    permutation = np.empty(x.shape[0], dtype=np.int64)
    distances = np.empty(x.shape[0] - 1, dtype=np.float64)
    visited = np.zeros(x.shape[0], dtype=bool)
    current = 0  # np.unique sorts rows lexicographically, making the start row-order independent.
    permutation[0] = current
    visited[current] = True
    for step in range(1, x.shape[0]):
        current, distance = _nearest_unvisited(
            x,
            tree,
            neighbour_distances,
            neighbour_indices,
            covered_radius,
            current,
            visited,
        )
        permutation[step] = current
        distances[step - 1] = distance
        visited[current] = True

    slopes = np.diff(y[permutation]) / distances
    slopes.flags.writeable = False
    return SlopeSequence(slopes)


def _slopes(context: ComputationContext) -> SlopeSequence:
    value = context.intermediate("ic.slopes")
    if not isinstance(value, SlopeSequence):
        raise TypeError("ic.slopes has an invalid runtime type")
    return value


def symbol_schedule(context: ComputationContext) -> SymbolSchedule:
    sequence = _slopes(context)
    if sequence.message is not None:
        return SymbolSchedule(
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.int64),
            0,
            sequence.message,
        )
    signs = np.sign(sequence.values).astype(np.int8)
    # A slope reads as zero from the first grid epsilon that strictly exceeds its magnitude.
    thresholds = np.searchsorted(EPSILON, np.abs(sequence.values), side="right")
    thresholds = thresholds.astype(np.int64)
    signs.flags.writeable = False
    thresholds.flags.writeable = False
    return SymbolSchedule(signs, thresholds, sequence.values.size)


def _schedule(context: ComputationContext) -> SymbolSchedule:
    value = context.intermediate("ic.symbols")
    if not isinstance(value, SymbolSchedule):
        raise TypeError("ic.symbols has an invalid runtime type")
    return value


def entropy_curve(context: ComputationContext) -> Curve:
    schedule = _schedule(context)
    if schedule.message is not None:
        return Curve(np.empty(0, dtype=np.float64), schedule.message)

    grid = EPSILON.size
    pairs = schedule.count - 1
    left = schedule.signs[:-1].astype(np.int64)
    right = schedule.signs[1:].astype(np.int64)
    left_threshold = schedule.thresholds[:-1]
    right_threshold = schedule.thresholds[1:]

    # A consecutive pair passes through at most three codes: both symbols intact, then the
    # smaller-magnitude slope zeroed, then both zeroed. Recording those two transitions is
    # enough to reconstruct every code count on the grid by a cumulative sum.
    initial = (left + 1) * 3 + (right + 1)
    middle = np.where(left_threshold <= right_threshold, 3 + (right + 1), (left + 1) * 3 + 1)
    first = np.minimum(np.minimum(left_threshold, right_threshold), grid)
    second = np.minimum(np.maximum(left_threshold, right_threshold), grid)

    # Transitions landing on the sink row `grid` never take effect inside the grid.
    positions = np.concatenate(
        (
            initial,
            first * 9 + initial,
            first * 9 + middle,
            second * 9 + middle,
            second * 9 + 4,
        )
    )
    weights = np.concatenate(
        (
            np.ones(pairs, dtype=np.int64),
            np.full(pairs, -1, dtype=np.int64),
            np.ones(pairs, dtype=np.int64),
            np.full(pairs, -1, dtype=np.int64),
            np.ones(pairs, dtype=np.int64),
        )
    )
    deltas = np.bincount(positions, weights=weights, minlength=(grid + 1) * 9)
    counts = np.cumsum(deltas[: grid * 9].reshape(grid, 9), axis=0)

    probabilities = counts[:, [1, 2, 3, 5, 6, 7]] / pairs
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive]) / np.log(6.0)
    values = -np.sum(terms, axis=1)
    values.flags.writeable = False
    return Curve(values)


def partial_information_curve(context: ComputationContext) -> Curve:
    schedule = _schedule(context)
    if schedule.message is not None:
        return Curve(np.empty(0, dtype=np.float64), schedule.message)

    grid = EPSILON.size
    denominator = schedule.count - 1
    alive = np.flatnonzero(schedule.signs != 0)

    # Symbol changes are counted over the non-zero subsequence, so growing epsilon deletes
    # entries from a linked list and each deletion adjusts the count by a local amount.
    signs = schedule.signs[alive].tolist()
    deaths = schedule.thresholds[alive].tolist()
    size = len(alive)
    changes = int(np.count_nonzero(np.diff(schedule.signs[alive]) != 0)) if size > 1 else 0
    previous = list(range(-1, size - 1))
    following = list(range(1, size + 1))
    if size:
        following[-1] = -1

    deltas = np.zeros(grid, dtype=np.int64)
    for node in sorted(range(size), key=deaths.__getitem__):
        death = deaths[node]
        if death >= grid:
            break
        before, after = previous[node], following[node]
        symbol = signs[node]
        if before >= 0 and after >= 0:
            deltas[death] += (
                (signs[before] != signs[after])
                - (signs[before] != symbol)
                - (symbol != signs[after])
            )
        elif before >= 0:
            deltas[death] -= signs[before] != symbol
        elif after >= 0:
            deltas[death] -= symbol != signs[after]
        if before >= 0:
            following[before] = after
        if after >= 0:
            previous[after] = before

    values = (changes + np.cumsum(deltas)) / denominator
    values.flags.writeable = False
    return Curve(values)


def _curve(context: ComputationContext, name: str) -> Curve:
    value = context.intermediate(name)
    if not isinstance(value, Curve):
        raise TypeError(f"{name} has an invalid runtime type")
    if value.message is not None:
        raise FeatureUnavailable(value.message)
    return value


def h_max(context: ComputationContext) -> float:
    return float(np.max(_curve(context, "ic.entropy").values))


def eps_s(context: ComputationContext) -> float:
    entropy = _curve(context, "ic.entropy").values
    candidates = EPSILON[entropy < SETTLING_SENSITIVITY]
    if candidates.size == 0 or candidates[0] <= 0.0:
        raise FeatureUnavailable("settling sensitivity was not reached at a positive epsilon")
    return float(np.log10(candidates[0]))


def eps_max(context: ComputationContext) -> float:
    entropy = _curve(context, "ic.entropy").values
    return float(np.median(EPSILON[entropy == np.max(entropy)]))


def eps_ratio(context: ComputationContext) -> float:
    partial = _curve(context, "ic.partial").values
    candidates = EPSILON[partial > INFORMATION_SENSITIVITY * partial[0]]
    if candidates.size == 0 or candidates[-1] <= 0.0:
        raise FeatureUnavailable("partial information sensitivity was not reached")
    return float(np.log10(candidates[-1]))


def m0(context: ComputationContext) -> float:
    sequence = _slopes(context)
    if sequence.message is not None:
        raise FeatureUnavailable(sequence.message)
    symbols = np.sign(sequence.values).astype(np.int8)
    nonzero = symbols[symbols != 0]
    changes = np.count_nonzero(np.diff(nonzero) != 0) if nonzero.size > 1 else 0
    return float(changes / (symbols.size - 1))


INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(
            "ic.slopes",
            (),
            frozenset({InputRequirement.X, InputRequirement.Y}),
        ),
        slope_sequence,
    ),
    IntermediateDefinition(
        IntermediateSpec("ic.symbols", ("ic.slopes",), frozenset()),
        symbol_schedule,
    ),
    IntermediateDefinition(
        IntermediateSpec("ic.entropy", ("ic.symbols",), frozenset()),
        entropy_curve,
    ),
    IntermediateDefinition(
        IntermediateSpec("ic.partial", ("ic.symbols",), frozenset()),
        partial_information_curve,
    ),
)

COMMON_INVARIANCES = (
    InvarianceClaim(
        Transformation.ROW_PERMUTATION,
        InvarianceBehavior.INVARIANT,
        conditions="unique lexicographic start and deterministic distance ties",
    ),
    InvarianceClaim(
        Transformation.X_TRANSLATION,
        InvarianceBehavior.INVARIANT,
        conditions="finite translation preserving pairwise distances",
    ),
    InvarianceClaim(
        Transformation.OBJECTIVE_SENSE_REVERSAL,
        InvarianceBehavior.INVARIANT,
        conditions="negate y and reverse the declared objective sense together",
    ),
)


def _feature(
    name: str,
    summary: str,
    intermediate: str,
    calculator: FeatureCalculator,
) -> FeatureDefinition:
    dotted_legacy_name = name.replace("_", ".")
    legacy_names = (name,) if dotted_legacy_name == name else (name, dotted_legacy_name)
    return FeatureDefinition(
        FeatureSpec(
            name=name,
            definition="deterministic-nn-flacco-grid-v1",
            summary=summary,
            group="ic",
            kind=MetricKind.LANDSCAPE,
            requirements=frozenset({InputRequirement.X, InputRequirement.Y}),
            intermediates=(intermediate,),
            cost=CostModel(
                CostTier.SAMPLE_ONLY,
                cpu="expected O(n log n + e); worst O(n^2 d)",
                memory="O(nd + kn + e), with k=20",
            ),
            deterministic=True,
            invariances=COMMON_INVARIANCES,
            references=(REFERENCE, R_REFERENCE),
            legacy_names=legacy_names,
            minimum_observations=3,
            notes=(
                "Uses a lexicographically anchored nearest-neighbour tour over "
                "duplicate-aggregated X.",
                "Uses the fixed flacco epsilon grid with e=1001.",
                "Epsilon-indexed curves are accumulated from per-slope threshold events, so "
                "the grid-by-slope symbol matrix is never materialised.",
                "Nearest-neighbour construction respects the explicit compute workers setting; "
                "the default is one worker.",
            ),
        ),
        calculator,
    )


FEATURES = (
    _feature(
        "ic.h_max",
        "Maximum information entropy of the fitness sequence.",
        "ic.entropy",
        h_max,
    ),
    _feature(
        "ic.eps_s",
        "Log10 settling sensitivity of the fitness sequence.",
        "ic.entropy",
        eps_s,
    ),
    _feature("ic.eps_max", "Epsilon at maximum information entropy.", "ic.entropy", eps_max),
    _feature(
        "ic.eps_ratio",
        "Log10 half-partial-information sensitivity.",
        "ic.partial",
        eps_ratio,
    ),
    _feature("ic.m0", "Initial partial information content at epsilon zero.", "ic.slopes", m0),
)
