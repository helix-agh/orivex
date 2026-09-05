import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bflacco import LandscapeSample, compute, list_features
from bflacco.result import FeatureStatus

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "r_flacco"
FEATURE_NAMES = (
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.intercept",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
)


def sample_for(x: np.ndarray, y: np.ndarray) -> LandscapeSample:
    dimension = x.shape[1]
    return LandscapeSample(x, y, np.full(dimension, -5.0), np.full(dimension, 5.0))


def numeric_values(result) -> dict[str, float]:
    return {name: float(item.value) for name, item in result.values.items()}


def test_linear_landscape_has_exact_linear_fit() -> None:
    rng = np.random.Generator(np.random.PCG64(100))
    x = rng.uniform(-4.0, 4.0, size=(80, 3))
    y = 4.0 + x @ np.array([2.0, -3.0, 0.5])

    result = compute(sample_for(x, y), "ela_meta.*", y_normalization="none")

    assert result.values["ela_meta.lin_simple.adj_r2"].value == pytest.approx(1.0)
    assert result.values["ela_meta.lin_simple.intercept"].value == pytest.approx(4.0)
    assert result.values["ela_meta.lin_w_interact.adj_r2"].value == pytest.approx(1.0)
    assert result.values["ela_meta.quad_simple.adj_r2"].value == pytest.approx(1.0)
    assert result.values["ela_meta.quad_w_interact.adj_r2"].value == pytest.approx(1.0)


def test_quadratic_landscape_has_exact_quadratic_fit() -> None:
    rng = np.random.Generator(np.random.PCG64(200))
    x = rng.uniform(-4.0, 4.0, size=(100, 3))
    y = np.sum(x * x, axis=1)

    result = compute(sample_for(x, y), "ela_meta.*", y_normalization="none")

    assert result.values["ela_meta.quad_simple.adj_r2"].value == pytest.approx(1.0)
    assert result.values["ela_meta.quad_w_interact.adj_r2"].value == pytest.approx(1.0)
    assert result.values["ela_meta.lin_simple.adj_r2"].value < 0.2


def test_linear_interaction_model_represents_pairwise_product() -> None:
    rng = np.random.Generator(np.random.PCG64(300))
    x = rng.uniform(-4.0, 4.0, size=(80, 2))
    y = 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.7 * x[:, 0] * x[:, 1]

    result = compute(sample_for(x, y), "ela_meta.lin_w_interact.adj_r2")

    assert result.values["ela_meta.lin_w_interact.adj_r2"].value == pytest.approx(1.0)


def test_quadratic_interaction_model_is_a_complete_degree_two_polynomial() -> None:
    rng = np.random.Generator(np.random.PCG64(350))
    x = rng.uniform(-4.0, 4.0, size=(80, 2))
    y = 1.0 + 2.0 * x[:, 0] ** 2 - x[:, 1] ** 2 + 0.7 * x[:, 0] * x[:, 1]

    result = compute(sample_for(x, y), "ela_meta.quad_w_interact.adj_r2")

    assert result.values["ela_meta.quad_w_interact.adj_r2"].value == pytest.approx(1.0)


def test_individual_selection_only_fits_the_requested_model() -> None:
    rng = np.random.Generator(np.random.PCG64(400))
    x = rng.uniform(-4.0, 4.0, size=(80, 2))
    sample = sample_for(x, np.sum(x * x, axis=1))

    result = compute(sample, "ela_meta.quad_simple.adj_r2")

    assert result.metadata.computed_intermediates == (
        "ela_meta.predictors.quad_simple",
        "ela_meta.fit.quad_simple",
    )


def test_shared_linear_fit_is_computed_once() -> None:
    rng = np.random.Generator(np.random.PCG64(500))
    x = rng.uniform(-4.0, 4.0, size=(80, 2))
    sample = sample_for(x, 2.0 + x[:, 0] - x[:, 1])

    result = compute(
        sample,
        ("ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept"),
    )

    assert result.metadata.computed_intermediates == (
        "ela_meta.predictors.lin_simple",
        "ela_meta.fit.lin_simple",
    )


def load_fixture(case: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with (FIXTURE_ROOT / "inputs" / f"{case}.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(row["x0"]), float(row["x1"])] for row in rows])
    y = np.array([float(row["y"]) for row in rows])
    expected = json.loads((FIXTURE_ROOT / "expected" / f"{case}.json").read_text(encoding="utf-8"))[
        "values"
    ]

    return x, y, expected


@pytest.mark.parametrize("case", ["linear_d2", "sphere_d2"])
def test_selected_meta_features_match_r_flacco_where_definitions_agree(case: str) -> None:
    x, y, expected = load_fixture(case)

    result = compute(sample_for(x, y), "ela_meta.*", y_normalization="none")

    names = (
        FEATURE_NAMES
        if case == "linear_d2"
        else (
            "ela_meta.lin_simple.intercept",
            "ela_meta.quad_simple.adj_r2",
            "ela_meta.quad_w_interact.adj_r2",
        )
    )
    for name in names:
        assert result.values[name].value == pytest.approx(expected[name], rel=2e-12, abs=2e-12)


def test_corrected_adjusted_r2_does_not_copy_r_flacco_degrees_of_freedom_bug() -> None:
    x, y, r_expected = load_fixture("sphere_d2")
    design = np.column_stack((np.ones(y.size), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coefficients
    total = y - np.mean(y)
    r_squared = 1.0 - (residuals @ residuals) / (total @ total)
    expected = 1.0 - (1.0 - r_squared) * (y.size - 1) / (y.size - x.shape[1] - 1)

    result = compute(sample_for(x, y), "ela_meta.lin_simple.adj_r2")
    actual = result.values["ela_meta.lin_simple.adj_r2"].value

    assert actual == pytest.approx(expected, rel=2e-13, abs=2e-13)
    assert actual != pytest.approx(r_expected["ela_meta.lin_simple.adj_r2"], abs=1e-6)


def test_adjusted_r2_is_invariant_to_affine_objective_transform() -> None:
    rng = np.random.Generator(np.random.PCG64(600))
    x = rng.uniform(-4.0, 4.0, size=(100, 2))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2
    sample = sample_for(x, y)
    transformed = sample_for(x, 7.5 * y - 123.0)
    selector = (
        "ela_meta.lin_simple.adj_r2",
        "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2",
        "ela_meta.quad_w_interact.adj_r2",
    )

    assert numeric_values(compute(transformed, selector)) == pytest.approx(
        numeric_values(compute(sample, selector)), rel=1e-12, abs=1e-12
    )


def test_linear_intercept_is_equivariant_to_objective_transform() -> None:
    rng = np.random.Generator(np.random.PCG64(700))
    x = rng.uniform(-4.0, 4.0, size=(80, 2))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2

    original = compute(sample_for(x, y), "ela_meta.lin_simple.intercept", y_normalization="none")
    transformed = compute(
        sample_for(x, 3.0 * y + 11.0), "ela_meta.lin_simple.intercept", y_normalization="none"
    )

    original_value = original.values["ela_meta.lin_simple.intercept"].value
    transformed_value = transformed.values["ela_meta.lin_simple.intercept"].value
    assert transformed_value == pytest.approx(3.0 * original_value + 11.0)


def test_row_and_variable_permutations_preserve_selected_features() -> None:
    rng = np.random.Generator(np.random.PCG64(800))
    x = rng.uniform(-4.0, 4.0, size=(100, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 - x[:, 2]
    row_order = rng.permutation(x.shape[0])
    variable_order = np.array([2, 0, 1])

    original = numeric_values(compute(sample_for(x, y), "ela_meta.*", y_normalization="none"))
    permuted = numeric_values(
        compute(
            sample_for(x[row_order][:, variable_order], y[row_order]),
            "ela_meta.*",
            y_normalization="none",
        )
    )

    assert permuted == pytest.approx(original, rel=1e-11, abs=1e-11)


def test_undefined_adjusted_r2_has_structured_status() -> None:
    x = np.array([[-1.0, -1.0], [0.0, 0.5], [1.0, 1.0]])
    y = np.array([1.0, 0.0, 2.0])

    result = compute(sample_for(x, y), "ela_meta.lin_simple.adj_r2")
    output = result.values["ela_meta.lin_simple.adj_r2"]

    assert output.status is FeatureStatus.INVALID
    assert output.value is None
    assert "more observations" in output.message


def test_constant_objective_has_undefined_adjusted_r2() -> None:
    rng = np.random.Generator(np.random.PCG64(900))
    x = rng.uniform(-4.0, 4.0, size=(20, 2))

    result = compute(sample_for(x, np.ones(20)), "ela_meta.lin_simple.adj_r2")

    assert result.values["ela_meta.lin_simple.adj_r2"].status is FeatureStatus.INVALID


def test_feature_discovery_returns_selected_meta_specs() -> None:
    assert tuple(spec.name for spec in list_features() if spec.group == "ela_meta") == FEATURE_NAMES
