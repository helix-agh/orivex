import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import KDTree

from orivex import LandscapeSample, compute, list_features
from orivex.engine import ComputationContext
from orivex.features.information_content import (
    EPSILON,
    NEIGHBOURHOOD,
    SlopeSequence,
    entropy_curve,
    partial_information_curve,
    slope_sequence,
    symbol_schedule,
)
from orivex.result import FeatureStatus

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


def test_information_content_worker_policy_is_explicit_and_output_invariant() -> None:
    rng = np.random.Generator(np.random.PCG64(72))
    x = rng.uniform(-4.0, 4.0, size=(100, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 - x[:, 2]
    sample = sample_for(x, y)

    single = compute(sample, "ic.*")
    parallel = compute(sample, "ic.*", workers=-1)

    assert single.metadata.workers == 1
    assert parallel.metadata.workers == -1
    assert numeric_values(parallel) == numeric_values(single)


@pytest.mark.parametrize("workers", [0, -2])
def test_invalid_worker_policy_is_rejected(workers: int) -> None:
    x = np.arange(5.0).reshape(-1, 1)
    sample = sample_for(x, x[:, 0] ** 2)

    with pytest.raises(ValueError, match="workers"):
        compute(sample, "ic.*", workers=workers)


def test_duplicate_points_are_aggregated_by_mean_objective() -> None:
    duplicated_x = np.array([[0.0], [1.0], [1.0], [2.0], [3.0], [3.0]])
    duplicated_y = np.array([0.0, 1.0, 3.0, 0.0, 4.0, 2.0])
    unique_x = np.arange(4.0).reshape(-1, 1)
    aggregated_y = np.array([0.0, 2.0, 0.0, 3.0])

    duplicated = numeric_values(
        compute(sample_for(duplicated_x, duplicated_y), "ic.*", y_normalization=None)
    )
    aggregated = numeric_values(
        compute(sample_for(unique_x, aggregated_y), "ic.*", y_normalization=None)
    )

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

    result = compute(LandscapeSample(x, y, [-5.0, -5.0], [5.0, 5.0]), "ic.*", y_normalization=None)

    for name, r_name in R_NAMES.items():
        assert result.values[name].value == pytest.approx(expected[r_name], rel=2e-13, abs=2e-13)


def test_feature_discovery_returns_information_content_specs() -> None:
    assert tuple(spec.name for spec in list_features() if spec.group == "ic") == FEATURE_NAMES


def dense_reference_curves(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct grid-by-slope transcription of the information-content curve definitions.

    The engine accumulates the same curves from per-slope threshold events, so this
    materialised form is the independent check that the accumulation is equivalent.
    """
    signs = np.sign(slopes).astype(np.int8)
    symbols = np.where(np.abs(slopes)[None, :] < EPSILON[:, None], np.int8(0), signs[None, :])
    codes = (symbols[:, :-1] + 1) * 3 + (symbols[:, 1:] + 1)
    probabilities = np.stack(
        [np.mean(codes == code, axis=1) for code in (1, 2, 3, 5, 6, 7)],
        axis=1,
    )
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive]) / np.log(6.0)
    entropy = -np.sum(terms, axis=1)

    partial = np.empty(EPSILON.size, dtype=np.float64)
    for index, row in enumerate(symbols):
        nonzero = row[row != 0]
        changes = np.count_nonzero(np.diff(nonzero) != 0) if nonzero.size > 1 else 0
        partial[index] = changes / (symbols.shape[1] - 1)
    return entropy, partial


def computed_curves(slopes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = LandscapeSample(np.arange(4.0).reshape(-1, 1), np.zeros(4), [-1.0], [4.0])
    cache: dict[str, object] = {"ic.slopes": SlopeSequence(slopes)}
    cache["ic.symbols"] = symbol_schedule(ComputationContext(sample, cache, None))
    context = ComputationContext(sample, cache, None)
    return entropy_curve(context).values, partial_information_curve(context).values


@pytest.mark.parametrize(
    ("name", "slopes"),
    [
        ("alternating", np.array([1.0, -1.0, 1.0, -1.0])),
        ("all_zero", np.zeros(6)),
        ("interleaved_zeros", np.array([0.0, 1.0, 0.0, -1.0, 0.0])),
        ("equal_magnitudes", np.array([1e-6, -1e-6, 1e-6])),
        ("spanning_the_grid", np.array([1e-20, 1e20, 1.0, -1.0])),
        ("shortest", np.array([-3.0, 3.0])),
        ("on_grid_points", EPSILON[[1, 5, 500, 999]].copy()),
        ("negated_grid_points", -EPSILON[[1, 5, 500, 999]]),
        ("runs", np.array([1.0, 1.0, -1.0, -1.0, 1.0])),
    ],
)
def test_curves_match_the_dense_grid_definition(name: str, slopes: np.ndarray) -> None:
    expected_entropy, expected_partial = dense_reference_curves(slopes)
    entropy, partial = computed_curves(slopes)

    assert np.array_equal(entropy, expected_entropy)
    assert np.array_equal(partial, expected_partial)


@pytest.mark.parametrize("seed", range(8))
def test_curves_match_the_dense_grid_definition_on_random_slopes(seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    size = int(rng.integers(2, 200))
    scale = 10.0 ** rng.integers(-8, 8, size=size)
    # Rounding produces exactly tied magnitudes and exact zeros, the cases where an
    # event-driven curve is most likely to disagree with the dense definition.
    slopes = np.round(rng.normal(size=size) * scale, 3)
    expected_entropy, expected_partial = dense_reference_curves(slopes)
    entropy, partial = computed_curves(slopes)

    assert np.array_equal(entropy, expected_entropy)
    assert np.array_equal(partial, expected_partial)


def test_symbol_schedule_stays_proportional_to_the_sample() -> None:
    """The schedule must not grow with the epsilon grid, as the dense matrix used to."""
    x = np.arange(50.0).reshape(-1, 1)
    sample = sample_for(x, np.sin(x[:, 0]))
    cache: dict[str, object] = {}
    cache["ic.slopes"] = slope_sequence(ComputationContext(sample, cache, None))
    schedule = symbol_schedule(ComputationContext(sample, cache, None))

    assert schedule.signs.shape == (49,)
    assert schedule.thresholds.shape == (49,)
    assert schedule.signs.size + schedule.thresholds.size < EPSILON.size


@pytest.mark.parametrize(("side", "dimension"), [(6, 2), (4, 3), (10, 2)])
def test_tour_resolves_exact_distance_ties_on_integer_lattices(side: int, dimension: int) -> None:
    """Every lattice neighbour is exactly equidistant, so tie handling decides the tour."""
    axes = [np.arange(float(side))] * dimension
    x = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, dimension)
    rng = np.random.Generator(np.random.PCG64(4242))
    y = rng.normal(size=x.shape[0])
    sample = sample_for(x, y)

    slopes = slope_sequence(ComputationContext(sample, {}, None)).values
    permuted = rng.permutation(x.shape[0])
    shuffled = slope_sequence(
        ComputationContext(sample_for(x[permuted], y[permuted]), {}, None)
    ).values

    assert slopes.size == x.shape[0] - 1
    assert np.array_equal(slopes, shuffled)


def test_stored_neighbours_cover_the_tie_radius_only_strictly_inside_it() -> None:
    """Justifies the strict comparison guarding the tour's fast tie path.

    The tour skips its range query when the winning radius is strictly inside the stored
    neighbour radius, because the stored list then provably holds every tied point. At the
    stored radius itself that containment breaks, so the comparison must not be relaxed.
    """
    axes = [np.arange(6.0)] * 2
    x = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, 2)
    tree = KDTree(x)
    distances, indices = tree.query(x, k=NEIGHBOURHOOD)

    escapes_at_the_radius = 0
    for row in range(x.shape[0]):
        radius = float(distances[row, -1])
        stored = set(indices[row].tolist())
        inside = tree.query_ball_point(x[row], np.nextafter(radius, -np.inf))
        assert set(inside) <= stored
        if not set(tree.query_ball_point(x[row], radius)) <= stored:
            escapes_at_the_radius += 1

    assert escapes_at_the_radius > 0
