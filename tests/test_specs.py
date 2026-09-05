import pytest

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


def valid_spec(**overrides: object) -> FeatureSpec:
    values = {
        "name": "ela_distr.skewness",
        "group": "ela_distr",
        "kind": MetricKind.LANDSCAPE,
        "definition": "corrected-v1",
        "summary": "Third standardized central moment of sampled objective values.",
        "requirements": frozenset({InputRequirement.Y}),
        "intermediates": ("y.central_moments",),
        "cost": CostModel(CostTier.SAMPLE_ONLY, cpu="O(n)", memory="O(1)"),
        "deterministic": True,
        "invariances": (
            InvarianceClaim(
                Transformation.Y_TRANSLATION,
                InvarianceBehavior.INVARIANT,
                conditions="finite y with non-zero variance",
            ),
        ),
        "references": (
            Reference(
                citation="Mersmann et al. (2011), Exploratory Landscape Analysis",
                doi="10.1145/2001576.2001690",
            ),
        ),
        "legacy_names": ("ela_distr.skewness",),
        "minimum_observations": 3,
    }
    values.update(overrides)
    return FeatureSpec(**values)  # type: ignore[arg-type]


def test_feature_spec_accepts_a_complete_contract() -> None:
    spec = valid_spec()

    assert spec.name == "ela_distr.skewness"
    assert spec.cost.additional_objective_evaluations == "0"
    assert spec.invariances[0].behavior is InvarianceBehavior.INVARIANT


@pytest.mark.parametrize("name", ["skewness", "ELA.skewness", "ela-distr.skewness", "ela."])
def test_feature_name_is_stable_and_machine_readable(name: str) -> None:
    with pytest.raises(ValueError, match="invalid feature name"):
        valid_spec(name=name)


def test_group_must_match_feature_prefix() -> None:
    with pytest.raises(ValueError, match="feature group"):
        valid_spec(group="other")


def test_objective_observations_are_an_explicit_requirement() -> None:
    with pytest.raises(ValueError, match="require objective observations"):
        valid_spec(requirements=frozenset({InputRequirement.X}))


def test_design_descriptor_requires_x_but_not_y() -> None:
    spec = valid_spec(
        name="pca.expl_var_cov_x",
        group="pca",
        kind=MetricKind.DESIGN,
        requirements=frozenset({InputRequirement.X}),
    )

    assert spec.kind is MetricKind.DESIGN


def test_invariance_claims_cannot_conflict() -> None:
    duplicate_claims = (
        InvarianceClaim(Transformation.Y_TRANSLATION, InvarianceBehavior.INVARIANT),
        InvarianceClaim(Transformation.Y_TRANSLATION, InvarianceBehavior.NON_INVARIANT),
    )

    with pytest.raises(ValueError, match="at most one claim"):
        valid_spec(invariances=duplicate_claims)


def test_reference_requires_a_resolvable_identifier() -> None:
    with pytest.raises(ValueError, match="DOI or URL"):
        Reference(citation="An otherwise valid citation")
