import numpy as np
import pytest
from fitness_distance_reference import REFERENCE

from orivex import LandscapeSample, compute, list_features
from orivex.result import FeatureStatus

FEATURE = "fitness_distance.fitness_std"


def sample_for(y, *, sense="minimize"):
    y = np.asarray(y, dtype=np.float64)
    return LandscapeSample(np.linspace(0, 1, len(y))[:, None], y, [0], [1], sense=sense)


@pytest.mark.parametrize("proportion", [0.1, 0.25, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_matches_sorted_sample_estimator(proportion, sense):
    y = np.random.default_rng(42).normal(size=25)
    sample = sample_for(y, sense=sense)
    expected = np.std(np.sort(sample.minimization_y)[: round(25 * proportion)], ddof=1)
    result = compute(
        sample,
        FEATURE,
        y_normalization=None,
        options={"fitness_distance": {"proportion_of_best": proportion}},
    )
    assert result.values[FEATURE].value == pytest.approx(expected, rel=1e-13)
    assert result.metadata.options["fitness_distance"]["proportion_of_best"] == proportion
    assert result.metadata.computed_intermediates == ("fitness_distance.selection",)
    assert result.metadata.additional_objective_evaluations == 0
    np.testing.assert_array_equal(sample.y, y)


def test_default_selection_uses_python_rounding_and_sample_deviation():
    # round(25 * 0.1) == 2: keep [0, 1], not three observations or the full sample.
    result = compute(sample_for(np.arange(25.0) ** 2), FEATURE, y_normalization=None)
    assert result.values[FEATURE].value == pytest.approx(1 / np.sqrt(2))
    assert result.metadata.options["fitness_distance"]["proportion_of_best"] == 0.1


@pytest.mark.parametrize("mode", [None, "minmax"])
def test_normalizes_full_sample_before_selection_and_does_not_filter_other_features(mode):
    sample = sample_for([0, 2, 4, 10])
    y = sample.y
    if mode == "minmax":
        y = y / 10
    result = compute(
        sample,
        [FEATURE, "ela_distr.skewness"],
        y_normalization=mode,
        options={"fitness_distance": {"proportion_of_best": 0.5}},
    )
    assert result.values[FEATURE].value == pytest.approx(np.std(y[:2], ddof=1))
    reference = compute(sample, "ela_distr.skewness", y_normalization=mode)
    assert result.values["ela_distr.skewness"] == reference.values["ela_distr.skewness"]


@pytest.mark.parametrize("proportion", [0, -0.1, 1.01, np.nan, np.inf, True, "0.1", None])
def test_invalid_proportion_is_rejected(proportion):
    with pytest.raises(ValueError, match="proportion_of_best"):
        compute(
            sample_for(np.arange(20.0)),
            FEATURE,
            options={"fitness_distance": {"proportion_of_best": proportion}},
        )


@pytest.mark.parametrize(("n", "proportion"), [(1, 1.0), (14, 0.1), (20, 0.01)])
def test_too_few_selected_observations_are_invalid(n, proportion):
    result = compute(
        sample_for(np.arange(float(n))),
        FEATURE,
        options={"fitness_distance": {"proportion_of_best": proportion}},
    )
    item = result.values[FEATURE]
    assert item.status is FeatureStatus.INVALID
    assert item.value is None
    assert "at least 2 selected observations" in item.message


@pytest.mark.parametrize("mode", [None, "minmax"])
def test_constant_selection_is_valid_zero(mode):
    result = compute(sample_for([3.0] * 19 + [8.0]), FEATURE, y_normalization=mode)
    assert result.values[FEATURE].status is FeatureStatus.OK
    assert result.values[FEATURE].value == 0.0


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200, 1e307])
def test_raw_deviation_is_scale_equivariant_without_variance_overflow(scale):
    y = np.arange(20.0) / 10
    result = compute(sample_for(y * scale), FEATURE, y_normalization=None)
    assert result.values[FEATURE].value / scale == pytest.approx(np.std(y[:2], ddof=1))


def test_raw_overflowing_range_is_supported():
    result = compute(
        sample_for([-1.7e308, 0, 1.7e308]),
        FEATURE,
        y_normalization=None,
        options={"fitness_distance": {"proportion_of_best": 1.0}},
    )
    assert result.values[FEATURE].value / 1.7e308 == pytest.approx(1.0)


def test_raw_affine_equivariance_permutation_and_sense_reversal():
    y = np.random.default_rng(8).normal(size=100)
    reference = compute(sample_for(y), FEATURE, y_normalization=None).values[FEATURE].value
    for sample, factor in (
        (sample_for(3 * y[::-1] + 7), 3),
        (sample_for(-y, sense="maximize"), 1),
    ):
        actual = compute(sample, FEATURE, y_normalization=None).values[FEATURE].value
        assert actual == pytest.approx(reference * factor)


def test_feature_is_discoverable():
    assert {spec.name for spec in list_features() if spec.group == "fitness_distance"} == {
        f"fitness_distance.{name}"
        for name in (
            "fitness_std",
            "fitness_mean",
            "distance_std",
            "distance_mean",
            "fd_cov",
            "fd_correlation",
        )
    }


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_family_matches_pflacco_reference(case):
    sample = LandscapeSample(REFERENCE["x"], REFERENCE["y"], [-4] * 3, [4] * 3, sense=case["sense"])
    result = compute(
        sample,
        "fitness_distance.*",
        y_normalization=None,
        options={"fitness_distance": case["options"]},
    )
    for name, expected in case["expected"].items():
        assert result.values[name].status is FeatureStatus.OK
        assert result.values[name].value == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert result.metadata.options == {"fitness_distance": case["options"]}
    assert result.metadata.computed_intermediates == (
        "fitness_distance.selection",
        "fitness_distance.distances",
    )


def test_full_sample_and_correlation_convention():
    sample = LandscapeSample([[0], [1], [2]], [0, 1, 2], [0], [2])
    result = compute(
        sample,
        "fitness_distance.*",
        y_normalization=None,
        options={"fitness_distance": {"proportion_of_best": 1}},
    )
    expected = {
        "fitness_mean": 1,
        "fitness_std": 1,
        "distance_mean": 1,
        "distance_std": 1,
        "fd_cov": 2 / 3,
        "fd_correlation": 2 / 3,
    }
    for name, value in expected.items():
        assert result.values[f"fitness_distance.{name}"].value == pytest.approx(value)


@pytest.mark.parametrize("constant_x, constant_y", [(True, False), (False, True), (True, True)])
def test_degenerate_family_isolated_to_correlation(constant_x, constant_y):
    sample = LandscapeSample(
        np.zeros((20, 1)) if constant_x else np.linspace(0, 1, 20)[:, None],
        np.ones(20) if constant_y else np.arange(20.0),
        [0],
        [1],
    )
    result = compute(sample, "fitness_distance.*")
    for name, item in result.values.items():
        assert item.status is (
            FeatureStatus.INVALID if name.endswith("fd_correlation") else FeatureStatus.OK
        )
    assert result.values["fitness_distance.fd_cov"].value == 0


def test_ties_choose_first_rows_and_first_best_reference():
    sample = LandscapeSample([[0], [2], [5], [9]], [1, 0, 0, 1], [0], [10])
    result = compute(
        sample,
        "fitness_distance.*",
        y_normalization=None,
        options={"fitness_distance": {"proportion_of_best": 0.75}},
    )
    assert result.values["fitness_distance.distance_mean"].value == pytest.approx(5 / 3)


def test_fitness_only_plan_does_not_compute_distances():
    result = compute(sample_for(np.arange(20.0)), [FEATURE, "fitness_distance.fitness_mean"])
    assert result.metadata.computed_intermediates == ("fitness_distance.selection",)


def test_too_small_selection_invalidates_entire_family():
    result = compute(sample_for([0, 1, 2]), "fitness_distance.*")
    assert all(item.status is FeatureStatus.INVALID for item in result.values.values())


def test_distances_are_euclidean_to_best_sampled_observation():
    sample = LandscapeSample([[1, 1], [4, 5], [7, 9]], [10, 11, 12], [0, 0], [10, 10])
    result = compute(
        sample, "fitness_distance.*", options={"fitness_distance": {"proportion_of_best": 1}}
    )
    # Best sampled y is 10: distances are 0, 5, 10, regardless of the unknown true optimum.
    assert result.values["fitness_distance.distance_mean"].value == pytest.approx(5)
    assert result.values["fitness_distance.distance_std"].value == pytest.approx(5)
