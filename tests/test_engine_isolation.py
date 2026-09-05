"""Expected numerical failures are isolated per feature; programming errors still propagate."""

import numpy as np
import pytest

from orivex.engine import (
    ComputationContext,
    Engine,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
)
from orivex.planner import IntermediateSpec
from orivex.result import FeatureStatus
from orivex.sample import LandscapeSample
from orivex.specs import (
    CostModel,
    CostTier,
    FeatureSpec,
    InputRequirement,
    MetricKind,
)


def _spec(name: str, *intermediates: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        group=name.split(".", 1)[0],
        kind=MetricKind.LANDSCAPE,
        definition="test-v1",
        summary=f"Test feature {name}.",
        requirements=frozenset({InputRequirement.Y}),
        intermediates=intermediates,
        cost=CostModel(CostTier.SAMPLE_ONLY, "O(n)", "O(1)"),
        deterministic=True,
    )


def _feature(name: str, calculate, *intermediates: str) -> FeatureDefinition:
    return FeatureDefinition(_spec(name, *intermediates), calculate)


def _intermediate(name: str, calculate, *dependencies: str) -> IntermediateDefinition:
    return IntermediateDefinition(
        IntermediateSpec(name, dependencies, frozenset({InputRequirement.Y})),
        calculate,
    )


def _sample() -> LandscapeSample:
    return LandscapeSample(
        np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 2.0, 3.0]), [-1.0], [4.0]
    )


def _overflow(_: ComputationContext) -> float:
    raise OverflowError("(34, 'Result too large')")


def _type_bug(_: ComputationContext) -> float:
    raise TypeError("programming error: wrong type")


def _singular(_: ComputationContext) -> float:
    raise np.linalg.LinAlgError("singular design matrix")


def test_numerical_failure_is_isolated_and_unrelated_features_survive() -> None:
    engine = Engine(
        (
            _feature("grp.ok", lambda _: 1.5),
            _feature("grp.overflow", _overflow),
            _feature("grp.linalg", _singular),
        ),
        (),
    )

    result = engine.compute(_sample(), ("grp.ok", "grp.overflow", "grp.linalg"))

    assert result.values["grp.ok"].status is FeatureStatus.OK
    assert result.values["grp.ok"].value == 1.5
    assert result.values["grp.overflow"].status is FeatureStatus.INVALID
    assert "too large" in result.values["grp.overflow"].message
    assert result.values["grp.linalg"].status is FeatureStatus.INVALID


def test_programming_error_still_propagates() -> None:
    engine = Engine((_feature("grp.ok", lambda _: 1.0), _feature("grp.bug", _type_bug)), ())

    with pytest.raises(TypeError, match="programming error"):
        engine.compute(_sample(), ("grp.ok", "grp.bug"))


def test_failed_intermediate_propagates_to_dependents_only() -> None:
    engine = Engine(
        (
            _feature("grp.needs_bad", lambda ctx: float(ctx.intermediate("bad")), "bad"),
            _feature("grp.needs_good", lambda ctx: float(ctx.intermediate("good")), "good"),
        ),
        (
            _intermediate("bad", _overflow),
            _intermediate("good", lambda _: 42.0),
        ),
    )

    result = engine.compute(_sample(), ("grp.needs_bad", "grp.needs_good"))

    assert result.values["grp.needs_bad"].status is FeatureStatus.INVALID
    assert "too large" in result.values["grp.needs_bad"].message
    assert result.values["grp.needs_good"].status is FeatureStatus.OK
    assert result.values["grp.needs_good"].value == 42.0


def test_failure_propagates_transitively_through_intermediate_chain() -> None:
    engine = Engine(
        (_feature("grp.leaf", lambda ctx: float(ctx.intermediate("second")), "second"),),
        (
            _intermediate("first", _overflow),
            _intermediate("second", lambda ctx: float(ctx.intermediate("first")) + 1.0, "first"),
        ),
    )

    result = engine.compute(_sample(), "grp.leaf")

    assert result.values["grp.leaf"].status is FeatureStatus.INVALID
    assert "too large" in result.values["grp.leaf"].message


def test_feature_unavailable_from_an_intermediate_is_isolated() -> None:
    def unavailable(_: ComputationContext) -> object:
        raise FeatureUnavailable("intermediate is undefined here")

    engine = Engine(
        (_feature("grp.dependent", lambda ctx: float(ctx.intermediate("missing")), "missing"),),
        (_intermediate("missing", unavailable),),
    )

    output = engine.compute(_sample(), "grp.dependent").values["grp.dependent"]

    assert output.status is FeatureStatus.INVALID
    assert "undefined here" in output.message
