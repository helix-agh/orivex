import pytest

from bflacco.planner import IntermediateSpec, Planner, PlanningError
from bflacco.registry import DuplicateFeatureError, FeatureRegistry, UnknownFeatureSelection
from bflacco.specs import (
    CostModel,
    CostTier,
    FeatureSpec,
    InputRequirement,
    MetricKind,
)


def feature(name: str, *intermediates: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        group=name.split(".", 1)[0],
        kind=MetricKind.LANDSCAPE,
        definition="corrected-v1",
        summary=f"Test definition for {name}.",
        requirements=frozenset({InputRequirement.Y}),
        intermediates=intermediates,
        cost=CostModel(CostTier.SAMPLE_ONLY, "O(n)", "O(1)"),
        deterministic=True,
    )


def test_registry_supports_exact_and_glob_selection_without_duplicates() -> None:
    registry = FeatureRegistry(
        (
            feature("ic.h_max", "ic.curve"),
            feature("ic.eps_ratio", "ic.curve"),
            feature("nbc.nn_nb_cor", "neighbors.nearest_better"),
        )
    )

    selected = registry.select(["ic.*", "ic.h_max"])

    assert tuple(spec.name for spec in selected) == ("ic.eps_ratio", "ic.h_max")


def test_registry_rejects_duplicate_and_unknown_features() -> None:
    registry = FeatureRegistry((feature("ic.h_max"),))
    with pytest.raises(DuplicateFeatureError):
        registry.register(feature("ic.h_max"))
    with pytest.raises(UnknownFeatureSelection, match="matched no features"):
        registry.select("nbc.*")


def test_planner_topologically_orders_and_shares_intermediates() -> None:
    registry = FeatureRegistry(
        (
            feature("ic.h_max", "ic.curve"),
            feature("ic.eps_ratio", "ic.curve"),
        )
    )
    planner = Planner(
        registry,
        (
            IntermediateSpec(
                "neighbors.tour", (), frozenset({InputRequirement.X, InputRequirement.RNG})
            ),
            IntermediateSpec("ic.slopes", ("neighbors.tour",), frozenset()),
            IntermediateSpec("ic.curve", ("ic.slopes",), frozenset()),
        ),
    )

    plan = planner.plan("ic.*")

    assert plan.intermediate_names == ("neighbors.tour", "ic.slopes", "ic.curve")
    assert plan.requirements == frozenset(
        {InputRequirement.X, InputRequirement.Y, InputRequirement.RNG}
    )


def test_planner_rejects_unknown_and_cyclic_intermediates() -> None:
    missing = Planner(FeatureRegistry((feature("ic.h_max", "missing"),)))
    with pytest.raises(PlanningError, match="unknown intermediate"):
        missing.plan("ic.h_max")

    cyclic = Planner(
        FeatureRegistry((feature("ic.h_max", "a"),)),
        (
            IntermediateSpec("a", ("b",), frozenset()),
            IntermediateSpec("b", ("a",), frozenset()),
        ),
    )
    with pytest.raises(PlanningError, match="cycle"):
        cyclic.plan("ic.h_max")
