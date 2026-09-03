import pytest

torch = pytest.importorskip("torch")

from bflacco.torch import TensorLandscapeSample


def valid_inputs(*, requires_grad: bool = False):
    x = torch.tensor(
        [[0.0, 0.25], [0.5, 0.75], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    y = torch.tensor(
        [1.0, 2.0, 4.0],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    bounds = torch.tensor([0.0, 0.0], dtype=torch.float64)
    return x, y, bounds, torch.ones(2, dtype=torch.float64)


def test_sample_preserves_dtype_device_and_autograd_history() -> None:
    x, y, lower, upper = valid_inputs(requires_grad=True)

    sample = TensorLandscapeSample(x, y, lower, upper)
    sample.minimization_y.sum().backward()

    assert sample.x.dtype is torch.float64
    assert sample.x.device == x.device
    assert y.grad is not None
    torch.testing.assert_close(y.grad, torch.ones_like(y))


def test_sample_clones_original_inputs() -> None:
    x, y, lower, upper = valid_inputs()
    sample = TensorLandscapeSample(x, y, lower, upper)

    x.add_(10.0)
    y.add_(10.0)

    torch.testing.assert_close(sample.x[0], torch.tensor([0.0, 0.25], dtype=torch.float64))
    torch.testing.assert_close(sample.y, torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))


def test_sample_detects_mutation_through_its_own_tensors() -> None:
    sample = TensorLandscapeSample(*valid_inputs())
    sample.y.add_(1.0)

    with pytest.raises(RuntimeError, match="must not be modified"):
        sample.validate_unchanged()


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("integer", "float32 or torch.float64"),
        ("mismatched_dtype", "same dtype"),
        ("mismatched_shape", "y must have shape"),
        ("outside_bounds", "inclusive box bounds"),
        ("nonfinite", "finite"),
    ],
)
def test_sample_rejects_invalid_inputs(replacement: str, message: str) -> None:
    x, y, lower, upper = valid_inputs()
    if replacement == "integer":
        x = x.to(torch.int64)
    elif replacement == "mismatched_dtype":
        y = y.to(torch.float32)
    elif replacement == "mismatched_shape":
        y = y[:-1]
    elif replacement == "outside_bounds":
        x[0, 0] = -1.0
    elif replacement == "nonfinite":
        y[0] = torch.inf

    with pytest.raises((TypeError, ValueError), match=message):
        TensorLandscapeSample(x, y, lower, upper)


def test_maximization_canonicalization_preserves_gradient_sign() -> None:
    x, y, lower, upper = valid_inputs(requires_grad=True)
    sample = TensorLandscapeSample(x, y, lower, upper, sense="maximize")

    sample.minimization_y.sum().backward()

    assert y.grad is not None
    torch.testing.assert_close(y.grad, -torch.ones_like(y))
