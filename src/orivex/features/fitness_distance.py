"""Fitness-distance statistics with shared selection and reference-only distances."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np

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


def selected_observations(observations: int, proportion: float) -> int:
    """Match pflacco's Python rounding, including ties to even."""
    count = round(observations * proportion)
    if count < 2:
        raise FeatureUnavailable(
            "fitness-distance features require at least 2 selected observations "
            f"(proportion_of_best={proportion} selects {count} of {observations})"
        )
    return count


SELECTION = "fitness_distance.selection"
DISTANCES = "fitness_distance.distances"


@dataclass(frozen=True, slots=True)
class Selection:
    indices: np.ndarray
    y: np.ndarray


def select_best(context: ComputationContext) -> Selection:
    # Rank raw canonical objectives so normalization cannot create selection ties.
    raw = context.sample.minimization_y
    count = selected_observations(
        raw.size, context.options["fitness_distance"]["proportion_of_best"]
    )
    if count == raw.size:
        indices = np.arange(raw.size)
    else:
        threshold = np.partition(raw, count - 1)[count - 1]
        mask = raw < threshold
        tied = np.flatnonzero(raw == threshold)[: count - np.count_nonzero(mask)]
        mask[tied] = True
        indices = np.flatnonzero(mask)
    return Selection(indices, context.y[indices])


def distances(context: ComputationContext) -> np.ndarray:
    selected = cast(Selection, context.intermediate(SELECTION))
    indices = selected.indices
    reference = indices[np.argmin(context.sample.minimization_y[indices])]
    delta = np.abs(context.sample.x[indices] - context.sample.x[reference])
    scale = np.max(delta, axis=1)
    divisor = np.where(scale > 0, scale, 1.0)
    return np.sqrt(np.sum((delta / divisor[:, None]) ** 2, axis=1)) * scale


def scaled_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    midpoint = float(np.min(values) / 2 + np.max(values) / 2)
    deviations = values - midpoint
    scale = float(np.max(np.abs(deviations)))
    return deviations / (scale or 1.0), scale, midpoint


def stable_mean(values: np.ndarray) -> float:
    scaled, scale, midpoint = scaled_values(values)
    return float(np.mean(scaled) * scale + midpoint)


def stable_std(values: np.ndarray) -> float:
    scaled, scale, _ = scaled_values(values)
    return float(np.std(scaled, ddof=1) * scale)


def fitness_std(context: ComputationContext) -> float:
    return stable_std(cast(Selection, context.intermediate(SELECTION)).y)


def fitness_mean(context: ComputationContext) -> float:
    return stable_mean(cast(Selection, context.intermediate(SELECTION)).y)


def distance_mean(context: ComputationContext) -> float:
    return stable_mean(cast(np.ndarray, context.intermediate(DISTANCES)))


def distance_std(context: ComputationContext) -> float:
    return stable_std(cast(np.ndarray, context.intermediate(DISTANCES)))


def fd_cov(context: ComputationContext) -> float:
    y, y_scale, _ = scaled_values(cast(Selection, context.intermediate(SELECTION)).y)
    d, d_scale, _ = scaled_values(cast(np.ndarray, context.intermediate(DISTANCES)))
    covariance = float(np.mean((y - np.mean(y)) * (d - np.mean(d))))
    return covariance * y_scale * d_scale


def fd_correlation(context: ComputationContext) -> float:
    y, y_scale, _ = scaled_values(cast(Selection, context.intermediate(SELECTION)).y)
    d, d_scale, _ = scaled_values(cast(np.ndarray, context.intermediate(DISTANCES)))
    if y_scale == 0 or d_scale == 0:
        raise FeatureUnavailable(
            "fitness-distance correlation requires non-zero fitness and distance variance"
        )
    # pflacco uses population covariance but sample standard deviations.
    return float(
        np.mean((y - np.mean(y)) * (d - np.mean(d))) / (np.std(y, ddof=1) * np.std(d, ddof=1))
    )


INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(SELECTION, (), frozenset({InputRequirement.Y})), select_best
    ),
    IntermediateDefinition(
        IntermediateSpec(
            DISTANCES, (SELECTION,), frozenset({InputRequirement.X, InputRequirement.Y})
        ),
        distances,
    ),
)


FITNESS_STD = FeatureDefinition(
    spec=FeatureSpec(
        name="fitness_distance.fitness_std",
        group="fitness_distance",
        kind=MetricKind.LANDSCAPE,
        definition="best-fraction-sample-std-v1",
        summary="Sample standard deviation of the best objective observations (ddof=1).",
        requirements=frozenset({InputRequirement.Y}),
        intermediates=(SELECTION,),
        cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n)", memory="O(n)"),
        deterministic=True,
        invariances=(
            InvarianceClaim(Transformation.ROW_PERMUTATION, InvarianceBehavior.INVARIANT),
            InvarianceClaim(Transformation.Y_TRANSLATION, InvarianceBehavior.INVARIANT),
            InvarianceClaim(
                Transformation.Y_POSITIVE_SCALING,
                InvarianceBehavior.EQUIVARIANT,
                notes="Multiplying input objectives by a positive factor scales the output by it.",
            ),
            InvarianceClaim(
                Transformation.OBJECTIVE_SENSE_REVERSAL,
                InvarianceBehavior.INVARIANT,
                conditions="negate y and reverse the declared objective sense together",
            ),
        ),
        references=(
            Reference(
                citation="pflacco, calculate_fitness_distance_correlation",
                url="https://github.com/Reiyan/pflacco/blob/main/pflacco/misc_features.py",
            ),
        ),
        legacy_names=("fitness_distance.fitness_std",),
        minimum_observations=2,
        notes=(
            "Keep round(n * proportion_of_best) smallest canonical objective values; default 0.1.",
            "Objective normalization uses the full sample before selecting the best fraction.",
            "Use proportion_of_best=1.0 for the full sample; no optimum or distances are needed.",
            "Fewer than two selected observations are invalid; a constant selection returns zero.",
        ),
    ),
    calculate=fitness_std,
)


def _feature(name: str, summary: str, calculate: FeatureCalculator) -> FeatureDefinition:
    uses_distances = name not in ("fitness_mean", "fitness_std")
    return FeatureDefinition(
        replace(
            FITNESS_STD.spec,
            name=f"fitness_distance.{name}",
            definition=f"best-fraction-{name.replace('_', '-')}-v1",
            summary=summary,
            requirements=frozenset({InputRequirement.X, InputRequirement.Y})
            if uses_distances
            else frozenset({InputRequirement.Y}),
            intermediates=(DISTANCES,) if uses_distances else (SELECTION,),
            cost=CostModel(
                CostTier.SAMPLE_ONLY,
                cpu="O(n + k*d)" if uses_distances else "O(n)",
                memory="O(n + k*d)" if uses_distances else "O(n)",
            ),
            invariances=(
                InvarianceClaim(
                    Transformation.OBJECTIVE_SENSE_REVERSAL,
                    InvarianceBehavior.INVARIANT,
                    conditions="negate y and reverse objective sense together",
                ),
            ),
            legacy_names=(f"fitness_distance.{name}",),
            notes=(
                "Select round(n * proportion_of_best) best canonical objectives; default 0.1.",
                "Normalize objectives over the full sample before computing selected statistics.",
                "Selection and reference ties use the first original row; "
                "tied samples may be row-order dependent.",
                "Euclidean distances use raw X and the best selected observation as reference.",
                "Covariance divides by k; standard deviations use ddof=1.",
                "fd_correlation is (k-1)/k times Pearson correlation; zero variance is invalid.",
                "At least two selected observations are required. k is the selected count.",
            ),
        ),
        calculate,
    )


FEATURES = (
    FITNESS_STD,
    _feature("fitness_mean", "Mean of the best objective observations.", fitness_mean),
    _feature(
        "distance_mean", "Mean distance to the selected reference observation.", distance_mean
    ),
    _feature(
        "distance_std", "Sample standard deviation of reference distances (ddof=1).", distance_std
    ),
    _feature("fd_cov", "Population covariance of selected fitness and reference distance.", fd_cov),
    _feature(
        "fd_correlation",
        "Population fitness-distance covariance divided by sample deviations.",
        fd_correlation,
    ),
)
