import numpy as np
import pytest

from orivex import LandscapeSample, compute
from orivex.engine import ComputationContext
from orivex.features.nearest_better_soft import gate_values, soft_nearest_better_graph
from orivex.options import resolve_options
from orivex.result import FeatureStatus

SOFT_NAMES = (
    "nbc_soft.nn_nb.sd_ratio",
    "nbc_soft.nn_nb.mean_ratio",
    "nbc_soft.nn_nb.cor",
    "nbc_soft.dist_ratio.coeff_var",
    "nbc_soft.nb_fitness.cor",
)


def rastrigin_sample(observations=150, dimensions=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((observations, dimensions)) * 10 - 5
    y = 10 * dimensions + ((x**2) - 10 * np.cos(2 * np.pi * x)).sum(axis=1)
    return LandscapeSample(x, y, [-5] * dimensions, [5] * dimensions)


def soft(sample, **options):
    return compute(sample, "nbc_soft.*", options={"nbc_soft": options} if options else None)


def exact_counterpart(name):
    return "nbc" + name[len("nbc_soft") :]


def test_soft_features_converge_to_the_exact_features():
    sample = rastrigin_sample()
    exact = compute(sample, "nbc.*")

    errors = [
        max(
            abs(
                soft(sample, temperature=tau).values[name].value
                - exact.values[exact_counterpart(name)].value
            )
            for name in SOFT_NAMES
        )
        for tau in (0.03, 0.003, 1e-5)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_a_heavy_tailed_objective_stays_resolved_at_the_default_temperature():
    rng = np.random.default_rng(1)
    x = rng.random((150, 2)) * 10 - 5
    y = 100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (x[:, 0] - 1) ** 2
    sample = LandscapeSample(x, y, [-5, -5], [5, 5])
    exact = compute(sample, "nbc.*")

    # Half the sample sits in the bottom 7% of the objective range: the crowding that plain
    # min-max scaling cannot resolve at any single temperature.
    assert np.median((y - y.min()) / np.ptp(y)) < 0.07

    result = soft(sample)
    error = max(
        abs(result.values[name].value - exact.values[exact_counterpart(name)].value)
        for name in SOFT_NAMES
    )

    # This is the reason the gates rank the objectives instead of min-max scaling them: plain
    # normalization crowds a four-order-of-magnitude range into an interval no single temperature
    # resolves, and diverged several times further at every temperature measured on BBOB.
    assert error < 0.1


def test_selection_weights_and_the_no_better_mass_form_a_partition():
    sample = rastrigin_sample(observations=80)
    options = resolve_options({"nbc_soft": {"temperature": 0.05}})
    context = ComputationContext(
        sample,
        {},
        rng=None,
        objective_y=sample.minimization_y,
        options=options,
    )

    graph = soft_nearest_better_graph(context)

    # Every row spends exactly one unit of mass: on its outgoing edges, or on having no better
    # neighbour at all. The temperature must be loose enough for that to be a real mixture.
    total = graph.soft_indegrees.sum() + graph.no_better_mass.sum()
    assert total == pytest.approx(sample.x.shape[0], abs=1e-9)
    assert graph.no_better_mass.max() < 1.0
    assert np.any(graph.soft_nearest_better_distances > graph.nearest_distances + 1e-6)
    assert np.all(graph.no_better_mass >= 0)
    assert np.all(graph.soft_nearest_better_distances >= graph.nearest_distances - 1e-12)


def test_gates_are_invariant_to_a_positive_affine_change_of_objectives():
    sample = rastrigin_sample(observations=60)
    scaled = LandscapeSample(
        sample.x,
        sample.minimization_y * 7.5 + 3.0,
        sample.lower,
        sample.upper,
    )

    base = soft(sample, temperature=0.05)
    moved = soft(scaled, temperature=0.05)

    for name in SOFT_NAMES:
        assert base.values[name].value == pytest.approx(moved.values[name].value, abs=1e-12)


def test_rank_gates_are_uniformly_spaced_whatever_the_objective_shape():
    # Five orders of magnitude of objective range, with most observations crowded at the bottom.
    y = np.exp(np.linspace(0.0, 12.0, 200))
    ideal = 1.0 / (y.size - 1)

    gates = np.sort(gate_values(y, 0.5))
    spacing = np.diff(gates)

    assert np.mean(spacing > 0.5 * ideal) > 0.8
    assert np.median(gates) == pytest.approx(0.5, abs=0.1)
    # Including the crowded lower half, which plain min-max scaling leaves entirely unresolved.
    assert np.mean(spacing[:100] > 0.5 * ideal) > 0.6

    crowded = np.sort((y - y.min()) / np.ptp(y))
    assert np.mean(np.diff(crowded)[:100] > 0.5 * ideal) == 0.0


def test_plateaus_gate_at_one_half_and_are_reported_as_a_known_gap():
    x = np.linspace(0.0, 1.0, 12)[:, None]
    y = np.concatenate([np.zeros(6), np.ones(6)])
    sample = LandscapeSample(x, y, [0], [1])

    result = soft(sample, temperature=1e-6)

    # The construction stays finite and usable on a plateau; it simply does not reproduce the
    # exact equal-fitness fallback, which is documented in the feature notes.
    assert all(result.values[name].status is FeatureStatus.OK for name in SOFT_NAMES)
    assert gate_values(y, 1e-6)[0] == pytest.approx(2.5 / 11.0, abs=1e-9)


def test_constant_objectives_and_single_observations_are_reported_unavailable():
    constant = LandscapeSample(np.linspace(0, 1, 8)[:, None], np.ones(8), [0], [1])
    for name, item in soft(constant).values.items():
        assert item.status is FeatureStatus.INVALID
        assert name in SOFT_NAMES

    single = LandscapeSample([[0.0]], [1.0], [0], [1])
    for item in soft(single).values.values():
        assert item.status is FeatureStatus.INVALID
