import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bflacco import LandscapeSample
from bflacco import compute as compute_numpy
from bflacco.result import FeatureStatus
from bflacco.torch import TensorLandscapeSample
from bflacco.torch import compute as compute_torch

FEATURE_NAMES = (
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.intercept",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
)


def samples_for(
    x: np.ndarray,
    y: np.ndarray,
    dtype,
) -> tuple[LandscapeSample, TensorLandscapeSample]:
    dimension = x.shape[1]
    lower = np.full(dimension, -5.0)
    upper = np.full(dimension, 5.0)
    numpy_sample = LandscapeSample(x, y, lower, upper)
    tensor_sample = TensorLandscapeSample(
        torch.as_tensor(x, dtype=dtype),
        torch.as_tensor(y, dtype=dtype),
        torch.as_tensor(lower, dtype=dtype),
        torch.as_tensor(upper, dtype=dtype),
    )
    return numpy_sample, tensor_sample


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_meta_model_features_match_numpy_and_preserve_tensor_properties(dtype) -> None:
    rng = np.random.Generator(np.random.PCG64(1200))
    x = rng.uniform(-4.0, 4.0, size=(80, 3))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 - 0.3 * x[:, 0] * x[:, 2]
    numpy_sample, tensor_sample = samples_for(x, y, dtype)

    expected = compute_numpy(numpy_sample, "ela_meta.*")
    result = compute_torch(tensor_sample, "ela_meta.*")

    tolerance = 2e-5 if dtype is torch.float32 else 2e-12
    assert tuple(result.values) == FEATURE_NAMES
    for name, item in result.values.items():
        assert isinstance(item.value, torch.Tensor)
        assert item.value.dtype is dtype
        assert item.value.device.type == "cpu"
        assert item.value.item() == pytest.approx(
            expected.values[name].value,
            rel=tolerance,
            abs=tolerance,
        )


def test_meta_model_selection_reuses_only_required_fit() -> None:
    rng = np.random.Generator(np.random.PCG64(1300))
    x = rng.uniform(-4.0, 4.0, size=(40, 2))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2
    _, sample = samples_for(x, y, torch.float64)

    result = compute_torch(
        sample,
        ("ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept"),
    )

    assert result.metadata.computed_intermediates == (
        "ela_meta.predictors.lin_simple",
        "ela_meta.fit.lin_simple",
    )


@pytest.mark.parametrize(
    "feature",
    [
        "ela_meta.lin_simple.adj_r2",
        "ela_meta.lin_simple.intercept",
        "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2",
        "ela_meta.quad_w_interact.adj_r2",
    ],
)
def test_meta_model_features_pass_gradcheck(feature: str) -> None:
    generator = torch.Generator().manual_seed(1400)
    x = -4.0 + 8.0 * torch.rand((30, 2), generator=generator, dtype=torch.float64)
    lower = torch.full((2,), -5.0, dtype=torch.float64)
    upper = torch.full((2,), 5.0, dtype=torch.float64)

    def calculate(y):
        sample = TensorLandscapeSample(x, y, lower, upper)
        value = compute_torch(sample, feature).values[feature].value
        assert value is not None
        return value

    y = (torch.sin(x[:, 0]) + x[:, 1].square() + 0.2 * x[:, 0] * x[:, 1]).requires_grad_()
    assert torch.autograd.gradcheck(calculate, (y,))


def test_meta_model_feature_backpropagates_through_objective_model() -> None:
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    x = torch.linspace(-2.0, 2.0, 30, dtype=torch.float64).reshape(-1, 1)
    y = torch.sin(x[:, 0]) + parameter * x[:, 0].square()
    sample = TensorLandscapeSample(x, y, [-3.0], [3.0])

    value = (
        compute_torch(sample, "ela_meta.quad_simple.adj_r2")
        .values["ela_meta.quad_simple.adj_r2"]
        .value
    )
    assert value is not None
    value.backward()

    assert value.grad_fn is not None
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert not torch.isclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_meta_model_reports_rank_sample_size_and_variance_failures() -> None:
    rank_deficient = TensorLandscapeSample(
        torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([1.0, 0.0, 2.0, 3.0]),
        [-3.0, -3.0],
        [3.0, 3.0],
    )
    too_few = TensorLandscapeSample(
        torch.tensor([[-1.0, -1.0], [0.0, 0.5], [1.0, 1.0]]),
        torch.tensor([1.0, 0.0, 2.0]),
        [-2.0, -2.0],
        [2.0, 2.0],
    )
    constant = TensorLandscapeSample(
        torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
        torch.ones(4),
        [-2.0],
        [3.0],
    )

    cases = (
        (rank_deficient, "ela_meta.lin_simple.intercept", "full-rank"),
        (too_few, "ela_meta.lin_simple.adj_r2", "more observations"),
        (constant, "ela_meta.lin_simple.adj_r2", "constant"),
    )
    for sample, feature, message in cases:
        output = compute_torch(sample, feature).values[feature]
        assert output.status is FeatureStatus.INVALID
        assert output.value is None
        assert output.message is not None
        assert message in output.message


@pytest.mark.parametrize("device_type", ["cuda"])
def test_available_accelerator_preserves_device(device_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device_type)
    generator = torch.Generator(device=device).manual_seed(1500)
    x = -4.0 + 8.0 * torch.rand((40, 2), generator=generator, device=device, dtype=torch.float32)
    y = torch.sin(x[:, 0]) + x[:, 1].square()
    sample = TensorLandscapeSample(
        x,
        y,
        torch.full((2,), -5.0, device=device),
        torch.full((2,), 5.0, device=device),
    )

    value = (
        compute_torch(sample, "ela_meta.lin_simple.adj_r2")
        .values["ela_meta.lin_simple.adj_r2"]
        .value
    )

    assert value is not None
    assert value.device.type == device_type
