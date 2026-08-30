import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bflacco import LandscapeSample, compute, list_features
from bflacco.features.information_content import EPSILON
from bflacco.result import FeatureStatus

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "r_flacco"
FEATURE_NAMES = (
    "ic.eps_max",
    "ic.eps_ratio",
    "ic.eps_s",
    "ic.h_max",
    "ic.m0",
)
R_NAMES = {
    "ic.eps_max": "ic.eps.max",
    "ic.eps_ratio": "ic.eps.ratio",
    "ic.eps_s": "ic.eps.s",
    "ic.h_max": "ic.h.max",
    "ic.m0": "ic.m0",
}


def sample_for(x: np.ndarray, y: np.ndarray) -> LandscapeSample:
    lower = np.min(x, axis=0) - 1.0
    upper = np.max(x, axis=0) + 1.0
    return LandscapeSample(x, y, lower, upper)


def numeric_values(result) -> dict[str, float]:
    return {name: float(item.value) for name, item in result.values.items()}


def test_information_content_matches_hand_built_alternating_sequence() -> None:
    x = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    result = compute(sample_for(x, y), "ic.*")

    probabilities = np.array([1.0 / 3.0, 2.0 / 3.0])
    expected_entropy = -np.sum(probabilities * np.log(probabilities) / np.log(6.0))
    active_epsilon = EPSILON[EPSILON <= 1.0]

    assert result.values["ic.h_max"].value == pytest.approx(expected_entropy)
    assert result.values["ic.m0"].value == pytest.approx(1.0)
    assert result.values["ic.eps_s"].value == pytest.approx(np.log10(EPSILON[EPSILON > 1.0][0]))
    assert result.values["ic.eps_max"].value == pytest.approx(np.median(active_epsilon))
    assert result.values["ic.eps_ratio"].value == pytest.approx(np.log10(active_epsilon[-1]))


def test_individual_information_selection_avoids_unneeded_curve() -> None:
    x = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    sample = sample_for(x, y)

    m0_result = compute(sample, "ic.m0")
    entropy_result = compute(sample, "ic.h_max")

    assert m0_result.metadata.computed_intermediates == ("ic.slopes",)
    assert entropy_result.metadata.computed_intermediates == (
        "ic.slopes",
        "ic.symbols",
        "ic.entropy",
    )


def test_duplicate_points_are_aggregated_by_mean_objective() -> None:
    duplicated_x = np.array([[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]])
    duplicated_y = np.array([0.0, 1.0, 3.0, 0.0, 4.0, 2.0])
    unique_x = np.arange(4.0).reshape(-1, 1)
    aggregated_y = np.array([0.0, 2.0, 0.0, 3.0])

    duplicated = numeric_values(compute(sample_for(duplicated_x, duplicated_y), "ic.*"))
    aggregated = numeric_values(compute(sample_for(unique_x, aggregated_y), "ic.*"))

    assert duplicated == pytest.approx(aggregated)


def test_information_content_is_row_permutation_and_x_translation_invariant() -> None:
    rng = np.random.Generator(np.random.PCG64(1010))
    x = rng.uniform(-4.0, 4.0, size=(60, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 - x[:, 2]
    permutation = rng.permutation(x.shape[0])

    original = numeric_values(compute(sample_for(x, y), "ic.*"))
    transformed = numeric_values(compute(sample_for(x[permutation] + 12.0, y[permutation]), "ic.*"))

    assert transformed == pytest.approx(original, rel=1e-12, abs=1e-12)


def test_information_content_uses_canonical_objective_sense() -> None:
    x = np.arange(5.0).reshape(-1, 1)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    minimizing = sample_for(x, y)
    maximizing = LandscapeSample(x, -y, [-1.0], [5.0], sense="maximize")

    assert numeric_values(compute(maximizing, "ic.*")) == pytest.approx(
        numeric_values(compute(minimizing, "ic.*"))
    )


def test_information_content_reports_too_few_unique_points() -> None:
    x = np.array([[0.0], [0.0], [1.0]])
    y = np.array([0.0, 1.0, 2.0])

    result = compute(sample_for(x, y), "ic.*")

    assert all(item.status is FeatureStatus.INVALID for item in result.values.values())
    assert all("distinct decision points" in item.message for item in result.values.values())


@pytest.mark.parametrize("case", ["linear_d2", "sphere_d2"])
def test_information_content_matches_controlled_r_flacco_1_8(case: str) -> None:
    with (FIXTURE_ROOT / "inputs" / f"{case}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row["x0"]), float(row["x1"])] for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    expected = json.loads((FIXTURE_ROOT / "expected" / f"{case}.json").read_text(encoding="utf-8"))[
        "values"
    ]

    result = compute(LandscapeSample(x, y, [-5.0, -5.0], [5.0, 5.0]), "ic.*")

    for name, r_name in R_NAMES.items():
        assert result.values[name].value == pytest.approx(expected[r_name], rel=2e-13, abs=2e-13)


def test_feature_discovery_returns_information_content_specs() -> None:
    assert tuple(spec.name for spec in list_features() if spec.group == "ic") == FEATURE_NAMES
