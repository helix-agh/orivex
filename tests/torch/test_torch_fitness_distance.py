import numpy as np
import pytest
from fitness_distance_reference import REFERENCE

torch = pytest.importorskip("torch")

from orivex import LandscapeSample
from orivex import compute as compute_numpy
from orivex.result import FeatureStatus
from orivex.torch import TensorLandscapeSample, compute

FEATURE = "fitness_distance.fitness_std"


def sample_for(y, *, sense="minimize"):
    x = torch.linspace(0, 1, y.numel(), dtype=y.dtype, device=y.device)[:, None]
    return TensorLandscapeSample(x, y, [0], [1], sense=sense)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", [None, "minmax"])
@pytest.mark.parametrize("proportion", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_matches_numpy_and_preserves_dtype(dtype, mode, proportion, sense):
    y = torch.tensor(np.random.default_rng(31).normal(size=25), dtype=dtype)
    sample = sample_for(y, sense=sense)
    numpy_sample = LandscapeSample(sample.x.numpy(), y.numpy(), [0], [1], sense=sense)
    expected = (
        compute_numpy(
            numpy_sample,
            FEATURE,
            y_normalization=mode,
            options={"fitness_distance": {"proportion_of_best": proportion}},
        )
        .values[FEATURE]
        .value
    )
    result = compute(
        sample,
        FEATURE,
        y_normalization=mode,
        options={"fitness_distance": {"proportion_of_best": proportion}},
    )
    value = result.values[FEATURE].value
    assert value.dtype is dtype
    assert value.device == y.device
    assert value.item() == pytest.approx(expected, rel=2e-5 if dtype is torch.float32 else 1e-13)
    assert result.metadata.options["fitness_distance"]["proportion_of_best"] == proportion
    assert result.metadata.computed_intermediates == ("fitness_distance.selection",)


@pytest.mark.parametrize("mode", [None, "minmax"])
@pytest.mark.parametrize("proportion", [0.1, 0.5, 1.0])
def test_gradcheck_through_selection_and_normalization(mode, proportion):
    y = torch.tensor(
        np.random.default_rng(31).normal(size=25), dtype=torch.float64, requires_grad=True
    )

    def calculate(y):
        return (
            compute(
                sample_for(y),
                FEATURE,
                y_normalization=mode,
                options={"fitness_distance": {"proportion_of_best": proportion}},
            )
            .values[FEATURE]
            .value
        )

    assert torch.autograd.gradcheck(calculate, (y,))


def test_raw_gradients_flow_only_to_selected_objectives():
    y = torch.arange(25.0, dtype=torch.float64, requires_grad=True)
    value = compute(sample_for(y), FEATURE, y_normalization=None).values[FEATURE].value
    value.backward()
    expected = torch.zeros_like(y)
    expected[0], expected[1] = -1 / np.sqrt(2), 1 / np.sqrt(2)
    torch.testing.assert_close(y.grad, expected)


@pytest.mark.parametrize("mode", [None, "minmax"])
@pytest.mark.parametrize("constant_sample", [False, True])
def test_constant_selection_returns_connected_zero_with_finite_gradients(mode, constant_sample):
    values = [3.0] * 20 if constant_sample else [3.0] * 19 + [8.0]
    y = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    item = compute(sample_for(y), FEATURE, y_normalization=mode).values[FEATURE]
    assert item.status is FeatureStatus.OK
    assert item.value.item() == 0.0
    item.value.backward()
    torch.testing.assert_close(y.grad, torch.zeros_like(y))


@pytest.mark.parametrize("scale", [1e-200, 1e200, 1e307])
def test_raw_extreme_scales_preserve_values_and_gradients(scale):
    y = torch.arange(20.0, dtype=torch.float64).mul(scale / 10).requires_grad_()
    item = compute(sample_for(y), FEATURE, y_normalization=None).values[FEATURE]
    assert item.value.item() / scale == pytest.approx(0.1 / np.sqrt(2))
    item.value.backward()
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("proportion", [0, -0.1, 1.01, float("nan"), float("inf"), True, None])
def test_invalid_proportion_is_rejected(proportion):
    with pytest.raises(ValueError, match="proportion_of_best"):
        compute(
            sample_for(torch.arange(20.0)),
            FEATURE,
            options={"fitness_distance": {"proportion_of_best": proportion}},
        )


@pytest.mark.parametrize(("n", "proportion"), [(1, 1.0), (14, 0.1), (20, 0.01)])
def test_too_few_selected_observations_are_invalid(n, proportion):
    item = compute(
        sample_for(torch.arange(float(n))),
        FEATURE,
        options={"fitness_distance": {"proportion_of_best": proportion}},
    ).values[FEATURE]
    assert item.status is FeatureStatus.INVALID
    assert "at least 2 selected observations" in item.message


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_available_accelerator_preserves_device_and_gradients(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is not available")
    y = torch.arange(25.0, device=device, requires_grad=True)
    result = compute(sample_for(y), "fitness_distance.*")
    for item in result.values.values():
        assert item.status is FeatureStatus.OK
        assert item.value.device == y.device
    sum(item.value for item in result.values.values()).backward()
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_family_matches_pflacco_reference(case):
    sample = TensorLandscapeSample(
        torch.tensor(REFERENCE["x"], dtype=torch.float64),
        torch.tensor(REFERENCE["y"], dtype=torch.float64),
        [-4] * 3,
        [4] * 3,
        sense=case["sense"],
    )
    result = compute(
        sample,
        "fitness_distance.*",
        y_normalization=None,
        options={"fitness_distance": case["options"]},
    )
    for name, expected in case["expected"].items():
        assert result.values[name].status is FeatureStatus.OK
        assert result.values[name].value.item() == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert result.metadata.options == {"fitness_distance": case["options"]}


@pytest.mark.parametrize("mode", [None, "minmax"])
@pytest.mark.parametrize("proportion", [0.5, 1.0])
def test_family_parity_and_gradients_in_x_and_y(mode, proportion):
    rng = np.random.default_rng(78)
    x = torch.tensor(rng.uniform(-2, 2, (8, 3)), dtype=torch.float64, requires_grad=True)
    y = torch.tensor(rng.normal(size=8), dtype=torch.float64, requires_grad=True)
    options = {
        "y_normalization": mode,
        "options": {"fitness_distance": {"proportion_of_best": proportion}},
    }

    def calculate(x, y):
        sample = TensorLandscapeSample(x, y, [-3] * 3, [3] * 3)
        return torch.stack(
            [
                item.value
                for item in compute(sample, "fitness_distance.*", **options).values.values()
            ]
        )

    expected = compute_numpy(
        LandscapeSample(x.detach().numpy(), y.detach().numpy(), [-3] * 3, [3] * 3),
        "fitness_distance.*",
        **options,
    )
    np.testing.assert_allclose(
        calculate(x, y).detach().numpy(),
        [item.value for item in expected.values.values()],
        rtol=1e-12,
        atol=1e-14,
    )
    assert torch.autograd.gradcheck(calculate, (x, y))


@pytest.mark.parametrize("constant_x, constant_y", [(True, False), (False, True), (True, True)])
def test_degenerate_family_values_and_finite_gradients(constant_x, constant_y):
    x = (
        torch.zeros((20, 1), dtype=torch.float64)
        if constant_x
        else torch.linspace(0, 1, 20, dtype=torch.float64)[:, None]
    ).requires_grad_()
    y = (
        torch.ones(20, dtype=torch.float64)
        if constant_y
        else torch.arange(20.0, dtype=torch.float64)
    ).requires_grad_()
    result = compute(TensorLandscapeSample(x, y, [0], [1]), "fitness_distance.*")
    for name, item in result.values.items():
        assert item.status is (
            FeatureStatus.INVALID if name.endswith("fd_correlation") else FeatureStatus.OK
        )
    sum(item.value for item in result.values.values() if item.status is FeatureStatus.OK).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("mode", [None, "minmax"])
@pytest.mark.parametrize("sense, expected", [("minimize", 7 / 3), ("maximize", 11 / 3)])
def test_best_reference_follows_objective_sense_after_normalization(mode, sense, expected):
    x = torch.tensor([[0.0], [2.0], [5.0], [9.0]], dtype=torch.float64)
    y = torch.tensor([100.0, 103.0, 108.0, 112.0], dtype=torch.float64)
    sample = TensorLandscapeSample(x, y, [0], [10], sense=sense)
    result = compute(
        sample,
        "fitness_distance.*",
        y_normalization=mode,
        options={"fitness_distance": {"proportion_of_best": 0.75}},
    )
    assert result.values["fitness_distance.distance_mean"].value.item() == pytest.approx(expected)


@pytest.mark.parametrize("mode", [None, "minmax"])
def test_family_float32_parity_and_tie_handling(mode):
    x = torch.tensor([[0.0, 1.0], [2.0, 0.0], [5.0, 2.0], [9.0, 8.0]], requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)
    options = {
        "options": {"fitness_distance": {"proportion_of_best": 0.75}},
        "y_normalization": mode,
    }
    actual = compute(TensorLandscapeSample(x, y, [0, 0], [10, 10]), "fitness_distance.*", **options)
    expected = compute_numpy(
        LandscapeSample(x.detach().numpy(), y.detach().numpy(), [0, 0], [10, 10]),
        "fitness_distance.*",
        **options,
    )
    for name, item in actual.values.items():
        assert item.value.dtype == torch.float32
        assert item.value.item() == pytest.approx(expected.values[name].value, rel=2e-5, abs=1e-7)
    sum(item.value for item in actual.values.values()).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


def test_dictionary_options_and_metadata_snapshot():
    options = {"fitness_distance": {"proportion_of_best": 1.0}}
    result = compute(sample_for(torch.arange(20.0)), "fitness_distance.*", options=options)
    assert result.metadata.options == options
    options["fitness_distance"]["proportion_of_best"] = 0.5
    assert result.metadata.options["fitness_distance"]["proportion_of_best"] == 1.0
    with pytest.raises(TypeError):
        result.metadata.options["fitness_distance"]["proportion_of_best"] = 0.5
    for group_options in [{"f_opt": 0}, {"minkowski_p": 2}, {"proportion_of_bset": 0.5}]:
        with pytest.raises(ValueError, match="unknown option"):
            compute(
                sample_for(torch.arange(20.0)), FEATURE, options={"fitness_distance": group_options}
            )
