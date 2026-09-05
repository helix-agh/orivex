"""Objective-value distribution features compatible with flacco type-3 estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bflacco.engine import (
    ComputationContext,
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

REFERENCE = Reference(
    citation="Mersmann et al. (2011), Exploratory Landscape Analysis",
    doi="10.1145/2001576.2001690",
)


@dataclass(frozen=True, slots=True)
class CenteredObjectives:
    observations: int
    values: np.ndarray


def centered_objectives(context: ComputationContext) -> CenteredObjectives:
    y = context.y
    centered = y - np.mean(y)
    centered.flags.writeable = False
    return CenteredObjectives(observations=y.size, values=centered)


CENTERED = IntermediateDefinition(
    spec=IntermediateSpec(
        name="y.centered",
        dependencies=(),
        requirements=frozenset({InputRequirement.Y}),
    ),
    calculate=centered_objectives,
)


def _centered(context: ComputationContext) -> CenteredObjectives:
    value = context.intermediate("y.centered")
    if not isinstance(value, CenteredObjectives):
        raise TypeError("y.centered has an invalid runtime type")
    return value


def sum2(context: ComputationContext) -> float:
    centered = _centered(context).values
    return float(centered @ centered)


def sum3(context: ComputationContext) -> float:
    return float(np.sum(_centered(context).values ** 3))


def sum4(context: ComputationContext) -> float:
    return float(np.sum(_centered(context).values ** 4))


SUM2 = IntermediateDefinition(
    IntermediateSpec("y.sum2", ("y.centered",), frozenset()),
    sum2,
)
SUM3 = IntermediateDefinition(
    IntermediateSpec("y.sum3", ("y.centered",), frozenset()),
    sum3,
)
SUM4 = IntermediateDefinition(
    IntermediateSpec("y.sum4", ("y.centered",), frozenset()),
    sum4,
)


def _sum(context: ComputationContext, name: str) -> float:
    value = context.intermediate(name)
    if not isinstance(value, float):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def skewness_type3(context: ComputationContext) -> float:
    centered = _centered(context)
    second = _sum(context, "y.sum2")
    third = _sum(context, "y.sum3")
    if centered.observations < 3:
        raise FeatureUnavailable("type-3 skewness requires at least 3 observations")
    if second == 0.0:
        raise FeatureUnavailable("skewness is undefined for constant objective values")
    n = centered.observations
    type1 = np.sqrt(n) * third / second**1.5
    return float(type1 * ((n - 1) / n) ** 1.5)


def kurtosis_type3(context: ComputationContext) -> float:
    centered = _centered(context)
    second = _sum(context, "y.sum2")
    fourth = _sum(context, "y.sum4")
    if centered.observations < 4:
        raise FeatureUnavailable("type-3 kurtosis requires at least 4 observations")
    if second == 0.0:
        raise FeatureUnavailable("kurtosis is undefined for constant objective values")
    n = centered.observations
    ratio = n * fourth / second**2
    return float(ratio * (1.0 - 1.0 / n) ** 2 - 3.0)


COMMON_INVARIANCES = (
    InvarianceClaim(
        Transformation.ROW_PERMUTATION,
        InvarianceBehavior.INVARIANT,
        conditions="paired finite observations",
    ),
    InvarianceClaim(
        Transformation.Y_TRANSLATION,
        InvarianceBehavior.INVARIANT,
        conditions="finite y with non-zero variance",
    ),
    InvarianceClaim(
        Transformation.Y_POSITIVE_SCALING,
        InvarianceBehavior.INVARIANT,
        conditions="finite positive scale and non-zero y variance",
    ),
)


SKEWNESS = FeatureDefinition(
    spec=FeatureSpec(
        name="ela_distr.skewness",
        group="ela_distr",
        kind=MetricKind.LANDSCAPE,
        definition="flacco-type3-v1",
        summary="Type-3 sample skewness of objective observations under minimization convention.",
        requirements=frozenset({InputRequirement.Y}),
        intermediates=("y.sum2", "y.sum3"),
        cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n)", memory="O(n) shared"),
        deterministic=True,
        invariances=(
            *COMMON_INVARIANCES,
            InvarianceClaim(
                Transformation.OBJECTIVE_SENSE_REVERSAL,
                InvarianceBehavior.INVARIANT,
                conditions="negate y and reverse the declared objective sense together",
            ),
        ),
        references=(REFERENCE,),
        legacy_names=("ela_distr.skewness",),
        minimum_observations=3,
    ),
    calculate=skewness_type3,
)


KURTOSIS = FeatureDefinition(
    spec=FeatureSpec(
        name="ela_distr.kurtosis",
        group="ela_distr",
        kind=MetricKind.LANDSCAPE,
        definition="flacco-type3-v1",
        summary="Type-3 excess sample kurtosis of objective observations.",
        requirements=frozenset({InputRequirement.Y}),
        intermediates=("y.sum2", "y.sum4"),
        cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n)", memory="O(n) shared"),
        deterministic=True,
        invariances=(
            *COMMON_INVARIANCES,
            InvarianceClaim(
                Transformation.OBJECTIVE_SENSE_REVERSAL,
                InvarianceBehavior.INVARIANT,
                conditions="negate y and reverse the declared objective sense together",
            ),
        ),
        references=(REFERENCE,),
        legacy_names=("ela_distr.kurtosis",),
        minimum_observations=4,
    ),
    calculate=kurtosis_type3,
)

FEATURES = (SKEWNESS, KURTOSIS)
INTERMEDIATES = (CENTERED, SUM2, SUM3, SUM4)
