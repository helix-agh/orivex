import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bflacco import LandscapeSample
from bflacco import compute as compute_numpy
from bflacco import list_features as list_numpy_features
from bflacco.result import DeviceType, FeatureStatus
from bflacco.torch import (
    TensorLandscapeSample,
    UnsupportedFeatureError,
    compute,
    list_capabilities,
    list_features,
)


def tensor_sample(y, *, dtype=None) -> TensorLandscapeSample:
    dtype = torch.float64 if dtype is None else dtype
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=dtype)
    x = torch.linspace(0.0, 1.0, y_tensor.numel(), dtype=dtype).reshape(-1, 1)
    return TensorLandscapeSample(
        x,
        y_tensor,
        torch.tensor([0.0], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_distribution_features_match_numpy_and_preserve_tensor_properties(dtype) -> None:
    y = [-4.0, -1.0, 0.0, 2.0, 8.0, 9.0]
    numpy_sample = LandscapeSample(
        np.linspace(0.0, 1.0, len(y)).reshape(-1, 1),
        y,
        [0.0],
        [1.0],
    )
    expected = compute_numpy(numpy_sample, "ela_distr.*")

    result = compute(tensor_sample(y, dtype=dtype), "ela_distr.*")

    tolerance = 2e-6 if dtype is torch.float32 else 1e-13
    for name, item in result.values.items():
        assert isinstance(item.value, torch.Tensor)
        assert item.value.dtype is dtype
        assert item.value.device.type == "cpu"
        assert item.value.item() == pytest.approx(expected.values[name].value, rel=tolerance)
    assert result.metadata.backend == "torch"
    assert result.metadata.device == "cpu"
    assert result.metadata.device_index is None
    assert result.metadata.dtype == str(dtype).removeprefix("torch.")


def test_feature_selection_reuses_only_required_intermediates() -> None:
    sample = tensor_sample([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])

    skewness = compute(sample, "ela_distr.skewness")
    kurtosis = compute(sample, "ela_distr.kurtosis")

    assert skewness.metadata.computed_intermediates == ("y.centered", "y.sum2", "y.sum3")
    assert kurtosis.metadata.computed_intermediates == ("y.centered", "y.sum2", "y.sum4")


@pytest.mark.parametrize(
    ("feature", "y", "message"),
    [
        ("ela_distr.skewness", [1.0, 2.0], "at least 3"),
        ("ela_distr.kurtosis", [1.0, 2.0, 3.0], "at least 4"),
        ("ela_distr.skewness", [1.0, 1.0, 1.0, 1.0], "constant"),
        ("ela_distr.kurtosis", [1.0, 1.0, 1.0, 1.0], "constant"),
    ],
)
def test_undefined_features_have_structured_status(feature, y, message) -> None:
    output = compute(tensor_sample(y), feature).values[feature]

    assert output.status is FeatureStatus.INVALID
    assert output.value is None
    assert output.message is not None
    assert message in output.message


@pytest.mark.parametrize("feature", ["ela_distr.skewness", "ela_distr.kurtosis"])
def test_distribution_features_pass_gradcheck(feature: str) -> None:
    x = torch.linspace(0.0, 1.0, 6, dtype=torch.float64).reshape(-1, 1)
    lower = torch.tensor([0.0], dtype=torch.float64)
    upper = torch.tensor([1.0], dtype=torch.float64)

    def calculate(y):
        value = compute(TensorLandscapeSample(x, y, lower, upper), feature).values[feature].value
        assert value is not None
        return value

    y = torch.tensor([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(calculate, (y,))


def test_feature_output_backpropagates_through_objective_model() -> None:
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    base = torch.tensor([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0], dtype=torch.float64)
    y = base + parameter * base.square()

    value = compute(tensor_sample(y), "ela_distr.skewness").values["ela_distr.skewness"].value
    assert value is not None
    value.backward()

    assert value.grad_fn is not None
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert not torch.isclose(parameter.grad, torch.zeros_like(parameter.grad))


def test_known_but_unsupported_feature_raises_backend_specific_error() -> None:
    with pytest.raises(UnsupportedFeatureError, match=r"ic\.h_max"):
        compute(tensor_sample([1.0, 2.0, 3.0]), "ic.h_max")


def test_discovery_exposes_shared_specs_and_literal_capabilities() -> None:
    specs = list_features()
    capabilities = list_capabilities()
    numpy_specs = {spec.name: spec for spec in list_numpy_features()}

    assert tuple(spec.name for spec in specs) == (
        "ela_distr.kurtosis",
        "ela_distr.skewness",
        "ela_meta.lin_simple.adj_r2",
        "ela_meta.lin_simple.intercept",
        "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2",
        "ela_meta.quad_w_interact.adj_r2",
    )
    assert tuple(capability.feature_name for capability in capabilities) == (
        "ela_distr.kurtosis",
        "ela_distr.skewness",
        "ela_meta.lin_simple.adj_r2",
        "ela_meta.lin_simple.intercept",
        "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2",
        "ela_meta.quad_w_interact.adj_r2",
    )
    assert all(capability.backend == "torch" for capability in capabilities)
    assert all(capability.autograd == "piecewise" for capability in capabilities)
    assert all(item.autograd == "smooth" for item in list_capabilities(y_normalization="none"))
    assert all(spec is numpy_specs[spec.name] for spec in specs)


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_available_accelerator_preserves_device(device_type: DeviceType) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    device = torch.device(device_type)
    dtype = torch.float32
    y = torch.tensor([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0], device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, y.numel(), device=device, dtype=dtype).reshape(-1, 1)
    sample = TensorLandscapeSample(
        x,
        y,
        torch.tensor([0.0], device=device, dtype=dtype),
        torch.tensor([1.0], device=device, dtype=dtype),
    )

    result = compute(sample, "ela_distr.skewness")
    value = result.values["ela_distr.skewness"].value

    assert value is not None
    assert value.device.type == device_type
    assert result.metadata.device == device_type
    assert result.metadata.device_index == value.device.index
