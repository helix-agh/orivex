import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bflacco import LandscapeSample, compute, list_features
from bflacco.result import FeatureStatus

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "r_flacco"


def sample_for(y) -> LandscapeSample:
    y_array = np.asarray(y, dtype=np.float64)
    x = np.linspace(0.0, 1.0, y_array.size).reshape(-1, 1)
    return LandscapeSample(x, y_array, lower=[0.0], upper=[1.0])


def manual_type3(y: np.ndarray) -> tuple[float, float]:
    n = y.size
    centered = y - y.mean()
    sum2 = np.sum(centered**2)
    skewness = np.sqrt(n) * np.sum(centered**3) / sum2**1.5
    skewness *= ((n - 1) / n) ** 1.5
    ratio = n * np.sum(centered**4) / sum2**2
    kurtosis = ratio * (1 - 1 / n) ** 2 - 3
    return float(skewness), float(kurtosis)


def values(result) -> dict[str, float]:
    return {name: float(item.value) for name, item in result.values.items()}


def test_distribution_features_match_closed_form_type3_estimators() -> None:
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])
    expected_skewness, expected_kurtosis = manual_type3(y)

    result = compute(sample_for(y), "ela_distr.*")

    assert result.values["ela_distr.skewness"].value == pytest.approx(expected_skewness)
    assert result.values["ela_distr.kurtosis"].value == pytest.approx(expected_kurtosis)
    assert result.metadata.computed_intermediates == (
        "y.centered",
        "y.sum2",
        "y.sum4",
        "y.sum3",
    )


def test_individual_selection_avoids_unrelated_moments() -> None:
    sample = sample_for([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])

    skewness = compute(sample, "ela_distr.skewness")
    kurtosis = compute(sample, "ela_distr.kurtosis")

    assert skewness.metadata.computed_intermediates == ("y.centered", "y.sum2", "y.sum3")
    assert kurtosis.metadata.computed_intermediates == ("y.centered", "y.sum2", "y.sum4")


@pytest.mark.parametrize("offset", [-100.0, 0.5, 1e6])
@pytest.mark.parametrize("scale", [0.01, 2.0, 1e5])
def test_distribution_features_are_translation_and_positive_scale_invariant(
    offset: float, scale: float
) -> None:
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])

    original = values(compute(sample_for(y), "ela_distr.*"))
    transformed = values(compute(sample_for(scale * y + offset), "ela_distr.*"))

    # The largest offset intentionally stresses float64 cancellation after the input values
    # have already lost low-order bits; algebraic invariance does not imply bit identity.
    assert transformed == pytest.approx(original, rel=1e-8, abs=1e-12)


def test_distribution_features_are_row_permutation_invariant() -> None:
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])
    permutation = np.array([5, 2, 0, 4, 1, 3])

    original = values(compute(sample_for(y), "ela_distr.*"))
    permuted = values(compute(sample_for(y[permutation]), "ela_distr.*"))

    assert permuted == pytest.approx(original, rel=1e-14, abs=1e-14)


def test_objective_sense_is_canonicalized() -> None:
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])
    minimizing = sample_for(y)
    maximizing = LandscapeSample(
        minimizing.x,
        -y,
        minimizing.lower,
        minimizing.upper,
        sense="maximize",
    )

    assert values(compute(maximizing, "ela_distr.*")) == pytest.approx(
        values(compute(minimizing, "ela_distr.*"))
    )


@pytest.mark.parametrize(
    ("feature", "y", "message"),
    [
        ("ela_distr.skewness", [1.0, 2.0], "at least 3"),
        ("ela_distr.kurtosis", [1.0, 2.0, 3.0], "at least 4"),
        ("ela_distr.skewness", [1.0, 1.0, 1.0, 1.0], "constant"),
        ("ela_distr.kurtosis", [1.0, 1.0, 1.0, 1.0], "constant"),
    ],
)
def test_undefined_distribution_features_have_structured_status(feature, y, message) -> None:
    result = compute(sample_for(y), feature)
    output = result.values[feature]

    assert output.status is FeatureStatus.INVALID
    assert output.value is None
    assert message in output.message


@pytest.mark.parametrize("case", ["linear_d2", "sphere_d2"])
def test_distribution_features_match_r_flacco_1_8(case: str) -> None:
    with (FIXTURE_ROOT / "inputs" / f"{case}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row["x0"]), float(row["x1"])] for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    expected = json.loads((FIXTURE_ROOT / "expected" / f"{case}.json").read_text(encoding="utf-8"))[
        "values"
    ]

    result = compute(
        LandscapeSample(x, y, [-5.0, -5.0], [5.0, 5.0]), "ela_distr.*", y_normalization="none"
    )

    assert result.values["ela_distr.skewness"].value == pytest.approx(
        expected["ela_distr.skewness"], rel=1e-13, abs=1e-13
    )
    assert result.values["ela_distr.kurtosis"].value == pytest.approx(
        expected["ela_distr.kurtosis"], rel=1e-13, abs=1e-13
    )


def test_feature_discovery_returns_individual_specs() -> None:
    assert tuple(spec.name for spec in list_features() if spec.group == "ela_distr") == (
        "ela_distr.kurtosis",
        "ela_distr.skewness",
    )
