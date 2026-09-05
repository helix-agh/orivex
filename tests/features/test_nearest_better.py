import csv
import json
from pathlib import Path

import numpy as np
import pytest

from orivex import LandscapeSample, compute, list_features
from orivex.result import FeatureStatus

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "r_flacco"
FEATURE_NAMES = (
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "nbc.nn_nb.cor",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.sd_ratio",
)


def sample_for(x: np.ndarray, y: np.ndarray) -> LandscapeSample:
    lower = np.min(x, axis=0) - 100.0
    upper = np.max(x, axis=0) + 100.0
    return LandscapeSample(x, y, lower, upper)


def numeric_values(result) -> dict[str, float]:
    return {name: float(item.value) for name, item in result.values.items()}


def test_nearest_better_features_match_hand_calculation() -> None:
    x = np.array([[0.0], [1.0], [3.0]])
    y = np.array([0.0, 2.0, 1.0])
    nearest = np.array([1.0, 1.0, 2.0])
    nearest_better = np.array([1.0, 1.0, 3.0])
    ratios = nearest / nearest_better
    indegrees = np.array([2.0, 0.0, 0.0])

    result = compute(sample_for(x, y), "nbc.*")

    assert result.values["nbc.nn_nb.sd_ratio"].value == pytest.approx(
        np.std(nearest, ddof=1) / np.std(nearest_better, ddof=1)
    )
    assert result.values["nbc.nn_nb.mean_ratio"].value == pytest.approx(
        np.mean(nearest) / np.mean(nearest_better)
    )
    assert result.values["nbc.nn_nb.cor"].value == pytest.approx(
        np.corrcoef(nearest, nearest_better)[0, 1]
    )
    assert result.values["nbc.dist_ratio.coeff_var"].value == pytest.approx(
        np.std(ratios, ddof=1) / np.mean(ratios)
    )
    assert result.values["nbc.nb_fitness.cor"].value == pytest.approx(
        np.corrcoef(indegrees, y)[0, 1]
    )


def test_individual_nbc_selection_computes_graph_once() -> None:
    x = np.array([[0.0], [1.0], [3.0]])
    y = np.array([0.0, 2.0, 1.0])

    result = compute(sample_for(x, y), "nbc.nn_nb.mean_ratio")

    assert result.metadata.computed_intermediates == ("nbc.graph",)


def test_nbc_is_invariant_under_supported_transformations() -> None:
    rng = np.random.Generator(np.random.PCG64(2020))
    x = rng.uniform(-4.0, 4.0, size=(80, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 - x[:, 2]
    permutation = rng.permutation(x.shape[0])
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    original = numeric_values(compute(sample_for(x, y), "nbc.*"))
    transformed = numeric_values(
        compute(
            sample_for(x[permutation] @ orthogonal + 17.0, 5.0 * y[permutation] - 9.0),
            "nbc.*",
        )
    )

    assert transformed == pytest.approx(original, rel=2e-12, abs=2e-12)


def test_nbc_uses_canonical_objective_sense() -> None:
    x = np.array([[0.0], [1.0], [3.0]])
    y = np.array([0.0, 2.0, 1.0])
    minimizing = sample_for(x, y)
    maximizing = LandscapeSample(x, -y, [-100.0], [103.0], sense="maximize")

    assert numeric_values(compute(maximizing, "nbc.*")) == pytest.approx(
        numeric_values(compute(minimizing, "nbc.*"))
    )


def test_nbc_reports_structured_status_for_one_observation() -> None:
    result = compute(sample_for(np.array([[0.0]]), np.array([1.0])), "nbc.*")

    assert all(item.status is FeatureStatus.INVALID for item in result.values.values())
    assert all("at least 2" in item.message for item in result.values.values())


def test_zero_nearest_sd_over_positive_denominator_is_valid_zero() -> None:
    # Nearest distances are constant ([1,1,1,1]) so their SD is exactly zero, while the
    # nearest-better distances ([1,1,1,3]) vary. The ratio is a defined 0.0, not INVALID.
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 3.0, 2.0, 1.0])

    item = compute(sample_for(x, y), "nbc.nn_nb.sd_ratio", y_normalization="none").values[
        "nbc.nn_nb.sd_ratio"
    ]

    assert item.status is FeatureStatus.OK
    assert item.value == 0.0


def test_constant_distance_ratios_have_zero_coefficient_of_variation() -> None:
    # Every nearest/nearest-better distance ratio equals one, so the coefficient of variation
    # is a defined 0.0 rather than a zero-variation failure.
    x = np.array([[0.0], [1.0], [3.0], [6.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    item = compute(sample_for(x, y), "nbc.dist_ratio.coeff_var", y_normalization="none").values[
        "nbc.dist_ratio.coeff_var"
    ]

    assert item.status is FeatureStatus.OK
    assert item.value == 0.0


def test_zero_nearest_better_denominator_remains_invalid() -> None:
    # Constant nearest-better distances make the sd_ratio denominator zero: still undefined.
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([3.0, 2.0, 1.0, 0.0])
    result = compute(sample_for(x, y), "nbc.nn_nb.sd_ratio", y_normalization="none")
    item = result.values["nbc.nn_nb.sd_ratio"]

    assert item.status is FeatureStatus.INVALID
    assert item.message is not None and "zero variation" in item.message


@pytest.mark.parametrize("case", ["linear_d2", "sphere_d2"])
def test_nbc_matches_r_flacco_1_8(case: str) -> None:
    with (FIXTURE_ROOT / "inputs" / f"{case}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row["x0"]), float(row["x1"])] for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    expected = json.loads((FIXTURE_ROOT / "expected" / f"{case}.json").read_text(encoding="utf-8"))[
        "values"
    ]

    result = compute(
        LandscapeSample(x, y, [-5.0, -5.0], [5.0, 5.0]), "nbc.*", y_normalization="none"
    )

    for name in FEATURE_NAMES:
        assert result.values[name].value == pytest.approx(expected[name], rel=2e-13, abs=2e-13)


def test_feature_discovery_returns_nbc_specs() -> None:
    assert tuple(spec.name for spec in list_features() if spec.group == "nbc") == FEATURE_NAMES
