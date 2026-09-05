import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orivex import LandscapeSample
from orivex import compute as compute_numpy
from orivex.torch import TensorLandscapeSample, compute, list_capabilities
from orivex.torch.normalization import normalize_objectives


@pytest.mark.parametrize("mode", ["none", "minmax", "zscore"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numpy_parity_and_sense_canonicalization(mode, dtype):
    rng = np.random.default_rng(14)
    x = torch.tensor(rng.uniform(-4, 4, (40, 2)), dtype=dtype)
    y = 12 + torch.sin(x[:, 0]) + x[:, 1].square()
    expected = compute_numpy(
        LandscapeSample(x.numpy(), y.numpy(), [-5, -5], [5, 5]),
        ["ela_distr.*", "ela_meta.*"],
        y_normalization=mode,
    )
    s = TensorLandscapeSample(x, -y, [-5, -5], [5, 5], sense="maximize")
    result = compute(s, ["ela_distr.*", "ela_meta.*"], y_normalization=mode)
    for name, item in result.values.items():
        assert item.status == expected.values[name].status
        assert item.value.dtype == dtype
        assert item.value.item() == pytest.approx(expected.values[name].value, abs=2e-5, rel=2e-5)
    assert result.metadata.y_normalization == mode
    torch.testing.assert_close(s.y, -y)


@pytest.mark.parametrize("mode", ["none", "minmax", "zscore"])
def test_gradcheck_includes_normalization_and_regression(mode):
    x = torch.tensor(
        [[-2.0], [-1.0], [0.2], [0.8], [1.5], [2.0]], dtype=torch.float64, requires_grad=True
    )
    y = torch.tensor([7.0, 3.0, 8.0, 2.0, 6.0, 11.0], dtype=torch.float64, requires_grad=True)

    def calculate(x, y):
        result = compute(
            TensorLandscapeSample(x, y, [-3], [3]),
            ["ela_meta.lin_simple.intercept", "ela_meta.quad_simple.adj_r2"],
            y_normalization=mode,
        )
        return tuple(item.value for item in result.values.values())

    assert torch.autograd.gradcheck(calculate, (x, y))


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_constant_and_extreme_tensor_preprocessing(mode):
    y = torch.full((6,), 23.0, dtype=torch.float64, requires_grad=True)
    value, constant = normalize_objectives(y, mode)
    assert constant
    value.sum().backward()
    torch.testing.assert_close(y.grad, torch.zeros_like(y))
    for dtype, magnitude in [(torch.float32, 3e38), (torch.float64, 1.7e308)]:
        y = torch.tensor([-magnitude, 0, magnitude], dtype=dtype)
        value, constant = normalize_objectives(y, mode)
        assert not constant and torch.isfinite(value).all()
        expected = [0, 0.5, 1] if mode == "minmax" else [-np.sqrt(1.5), 0, np.sqrt(1.5)]
        torch.testing.assert_close(value, torch.tensor(expected, dtype=dtype))


def test_capabilities_describe_effective_preprocessing():
    assert all(c.autograd == "piecewise" for c in list_capabilities())
    assert all(c.autograd == "smooth" for c in list_capabilities(y_normalization="zscore"))
    with pytest.raises(ValueError, match="y_normalization"):
        list_capabilities(y_normalization="typo")
