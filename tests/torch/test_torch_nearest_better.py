import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orivex import LandscapeSample
from orivex import compute as compute_numpy
from orivex.result import DeviceType, FeatureStatus
from orivex.torch import TensorLandscapeSample, compute, list_capabilities

EXACT_NAMES = (
    "nbc.nn_nb.sd_ratio",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
)
SOFT_NAMES = tuple("nbc_soft" + name[len("nbc") :] for name in EXACT_NAMES)
ALL_NAMES = EXACT_NAMES + SOFT_NAMES


def rastrigin(observations=120, dimensions=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((observations, dimensions)) * 10 - 5
    y = 10 * dimensions + ((x**2) - 10 * np.cos(2 * np.pi * x)).sum(axis=1)
    return x, y


def samples(x, y, *, dtype=torch.float64, device="cpu", requires_grad=False):
    numpy_sample = LandscapeSample(x, y, [-5] * x.shape[1], [5] * x.shape[1])
    tensors = {"device": device, "dtype": dtype}
    tensor_sample = TensorLandscapeSample(
        torch.tensor(x, **tensors, requires_grad=requires_grad),
        torch.tensor(y, **tensors, requires_grad=requires_grad),
        torch.tensor([-5.0] * x.shape[1], **tensors),
        torch.tensor([5.0] * x.shape[1], **tensors),
    )
    return numpy_sample, tensor_sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_and_soft_features_match_the_numpy_backend(dtype) -> None:
    numpy_sample, tensor_sample = samples(*rastrigin(), dtype=dtype)
    options = {"nbc_soft": {"temperature": 0.01}}

    expected = compute_numpy(numpy_sample, ["nbc.*", "nbc_soft.*"], options=options)
    result = compute(tensor_sample, ["nbc.*", "nbc_soft.*"], options=options)

    tolerance = 2e-5 if dtype is torch.float32 else 1e-12
    assert set(result.values) == set(ALL_NAMES)
    for name, item in result.values.items():
        assert isinstance(item.value, torch.Tensor)
        assert item.value.dtype is dtype
        assert item.value.item() == pytest.approx(expected.values[name].value, rel=tolerance)
    assert result.metadata.backend == "torch"


def test_the_exact_group_reproduces_the_equal_fitness_fallback() -> None:
    # Two plateaus plus a unique best: exercises the equal-fitness edge and the replacement
    # distance of the root, which are the two branches the surrogate cannot reproduce.
    x = np.linspace(0.0, 1.0, 9).reshape(-1, 1)
    y = np.array([0.0, 2.0, 2.0, 2.0, 5.0, 5.0, 7.0, 7.0, 7.0])
    numpy_sample, tensor_sample = samples(x, y)

    expected = compute_numpy(numpy_sample, "nbc.*")
    result = compute(tensor_sample, "nbc.*")

    for name in EXACT_NAMES:
        assert result.values[name].status is expected.values[name].status
        if result.values[name].status is FeatureStatus.OK:
            assert result.values[name].value.item() == pytest.approx(
                expected.values[name].value, rel=1e-12
            )


def test_gradients_reach_the_inputs_each_group_actually_depends_on() -> None:
    x, y = rastrigin(observations=80)
    tensors = {"dtype": torch.float64}
    x_tensor = torch.tensor(x, **tensors, requires_grad=True)
    y_tensor = torch.tensor(y, **tensors, requires_grad=True)
    sample = TensorLandscapeSample(
        x_tensor,
        y_tensor,
        torch.tensor([-5.0, -5.0], **tensors),
        torch.tensor([5.0, 5.0], **tensors),
    )

    result = compute(sample, ["nbc.*", "nbc_soft.*"])

    def gradient_norm(value, target):
        if value.grad_fn is None:
            return None
        (found,) = torch.autograd.grad(value, target, retain_graph=True, allow_unused=True)
        return None if found is None else found.norm().item()

    for name in ALL_NAMES:
        value = result.values[name].value
        design = gradient_norm(value, x_tensor)
        objective = gradient_norm(value, y_tensor)
        if name.startswith("nbc_soft."):
            # The surrogate exists precisely to supply this gradient.
            assert objective is not None and objective > 0
        elif name == "nbc.nb_fitness.cor":
            # Differentiates through y at a locally fixed graph; indegrees are integer counts.
            assert objective is not None and objective > 0
            assert design is None
        else:
            # Ordering picks the edges, so the exact distance features are locally constant in y.
            assert objective is None
            assert design is not None and design > 0


def test_soft_features_pass_finite_difference_gradcheck() -> None:
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.random((30, 2)))
    # Irregular spacing on purpose: the gate scale is a median consecutive gap, which is only
    # piecewise linear and has a kink exactly where many gaps coincide.
    y = torch.tensor(np.sort(rng.random(30)), requires_grad=True)
    bounds = (torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))

    def features(objectives):
        result = compute(
            TensorLandscapeSample(x, objectives, *bounds),
            ["nbc_soft.nn_nb.sd_ratio", "nbc_soft.nb_fitness.cor"],
            y_normalization=None,
            options={"nbc_soft": {"temperature": 0.05}},
        )
        return torch.stack([result.values[name].value for name in sorted(result.values)])

    assert torch.autograd.gradcheck(features, (y,), eps=1e-6, atol=1e-6, rtol=1e-3)


def test_a_lower_temperature_moves_the_surrogate_towards_the_exact_features() -> None:
    _, tensor_sample = samples(*rastrigin())
    exact = compute(tensor_sample, "nbc.*")

    def error(temperature):
        soft = compute(
            tensor_sample, "nbc_soft.*", options={"nbc_soft": {"temperature": temperature}}
        )
        return max(
            abs(
                soft.values["nbc_soft" + name[len("nbc") :]].value.item()
                - exact.values[name].value.item()
            )
            for name in EXACT_NAMES
        )

    assert error(0.03) > error(0.003) > error(1e-5)
    assert error(1e-5) < 1e-3


def test_capabilities_state_where_gradients_come_from() -> None:
    capabilities = {item.feature_name: item for item in list_capabilities(y_normalization=None)}

    for name in ALL_NAMES:
        assert capabilities[name].backend == "torch"
        assert capabilities[name].autograd == "piecewise"
        assert "float64" in capabilities[name].dtypes
    assert any("Locally constant in y" in note for note in capabilities[EXACT_NAMES[0]].notes)
    assert any("locally fixed graph" in note for note in capabilities["nbc.nb_fitness.cor"].notes)
    assert any("temperature" in note for note in capabilities[SOFT_NAMES[0]].notes)


def test_short_samples_are_reported_unavailable() -> None:
    _, tensor_sample = samples(np.zeros((1, 2)), np.zeros(1))

    result = compute(tensor_sample, ["nbc.*", "nbc_soft.*"])

    assert all(item.status is FeatureStatus.INVALID for item in result.values.values())


def test_a_constant_objective_keeps_the_exact_group_and_disables_the_surrogate() -> None:
    numpy_sample, tensor_sample = samples(np.linspace(0.0, 1.0, 8).reshape(-1, 2), np.ones(4))

    expected = compute_numpy(numpy_sample, ["nbc.*", "nbc_soft.*"])
    result = compute(tensor_sample, ["nbc.*", "nbc_soft.*"])

    # Every point is its own nearest equal-fitness neighbour, so the exact distance features stay
    # defined; the gates have no gap to resolve, so the surrogate reports itself unavailable.
    for name in ALL_NAMES:
        assert result.values[name].status is expected.values[name].status
    assert result.values["nbc.nn_nb.sd_ratio"].status is FeatureStatus.OK
    assert all(result.values[name].status is FeatureStatus.INVALID for name in SOFT_NAMES)


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_available_accelerator_preserves_device(device_type: DeviceType) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    x, y = rastrigin(observations=60)
    reference = compute(samples(x, y)[1], ["nbc.*", "nbc_soft.*"])
    _, sample = samples(x, y, dtype=torch.float32, device=device_type)

    result = compute(sample, ["nbc.*", "nbc_soft.*"])

    for name in ALL_NAMES:
        value = result.values[name].value
        assert value is not None
        assert value.device.type == device_type
        assert value.item() == pytest.approx(reference.values[name].value.item(), abs=2e-5)
    assert result.metadata.device == device_type
