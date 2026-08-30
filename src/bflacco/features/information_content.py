"""Information-content features over a deterministic nearest-neighbour tour."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from ..engine import (
    ComputationContext,
    FeatureCalculator,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
)
from ..planner import IntermediateSpec
from ..specs import (
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
class SymbolMatrix:
    values: np.ndarray
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
    np.add.at(y_sums, inverse, context.sample.minimization_y)
    return x, y_sums / counts


def _nearest_unvisited(
    x: np.ndarray,
    tree: KDTree,
    neighbour_distances: np.ndarray,
    neighbour_indices: np.ndarray,
    current: int,
    visited: np.ndarray,
) -> tuple[int, float]:
    candidates = neighbour_indices[current]
    available = candidates[~visited[candidates]]
    if available.size:
        candidate_distances = neighbour_distances[current][~visited[candidates]]
        distance = float(np.min(candidate_distances))
        tied = tree.query_ball_point(x[current], np.nextafter(distance, np.inf))
        tied = np.asarray(tied, dtype=np.int64)
        tied = tied[~visited[tied]]
        if tied.size:
            deltas = x[tied] - x[current]
            exact = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
            minimum = np.min(exact)
            return int(np.min(tied[exact == minimum])), float(minimum)

    available = np.flatnonzero(~visited)
    deltas = x[available] - x[current]
    exact = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
    minimum = np.min(exact)
    return int(np.min(available[exact == minimum])), float(minimum)


def slope_sequence(context: ComputationContext) -> SlopeSequence:
    x, y = _aggregate_duplicate_points(context)
    if x.shape[0] < 3:
        return SlopeSequence(
            np.empty(0, dtype=np.float64),
            "information content requires at least 3 distinct decision points",
        )

    tree = KDTree(x)
    k = min(NEIGHBOURHOOD, x.shape[0])
    neighbour_distances, neighbour_indices = tree.query(x, k=k)
    if k == 1:
        neighbour_distances = neighbour_distances[:, None]
        neighbour_indices = neighbour_indices[:, None]

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


def symbol_matrix(context: ComputationContext) -> SymbolMatrix:
    sequence = _slopes(context)
    if sequence.message is not None:
        return SymbolMatrix(np.empty((0, 0), dtype=np.int8), sequence.message)
    signs = np.sign(sequence.values).astype(np.int8)
    symbols = np.where(
        np.abs(sequence.values)[None, :] < EPSILON[:, None],
        np.int8(0),
        signs[None, :],
    )
    symbols.flags.writeable = False
    return SymbolMatrix(symbols)


def _symbols(context: ComputationContext) -> SymbolMatrix:
    value = context.intermediate("ic.symbols")
    if not isinstance(value, SymbolMatrix):
        raise TypeError("ic.symbols has an invalid runtime type")
    return value


def entropy_curve(context: ComputationContext) -> Curve:
    symbols = _symbols(context)
    if symbols.message is not None:
        return Curve(np.empty(0, dtype=np.float64), symbols.message)
    left = symbols.values[:, :-1]
    right = symbols.values[:, 1:]
    codes = (left + 1) * 3 + (right + 1)
    probabilities = np.stack(
        [np.mean(codes == code, axis=1) for code in (1, 2, 3, 5, 6, 7)],
        axis=1,
    )
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive]) / np.log(6.0)
    values = -np.sum(terms, axis=1)
    values.flags.writeable = False
    return Curve(values)


def partial_information_curve(context: ComputationContext) -> Curve:
    symbols = _symbols(context)
    if symbols.message is not None:
        return Curve(np.empty(0, dtype=np.float64), symbols.message)
    denominator = symbols.values.shape[1] - 1
    values = np.empty(EPSILON.size, dtype=np.float64)
    for index, row in enumerate(symbols.values):
        nonzero = row[row != 0]
        changes = np.count_nonzero(np.diff(nonzero) != 0) if nonzero.size > 1 else 0
        values[index] = changes / denominator
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
        symbol_matrix,
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
            cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n^2 d + en)", memory="O(en)"),
            deterministic=True,
            invariances=COMMON_INVARIANCES,
            references=(REFERENCE, R_REFERENCE),
            legacy_names=legacy_names,
            minimum_observations=3,
            notes=(
                "Uses a lexicographically anchored nearest-neighbour tour over "
                "duplicate-aggregated X.",
                "Uses the fixed flacco epsilon grid with e=1001.",
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
