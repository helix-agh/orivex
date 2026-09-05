import numpy as np
import pytest

from orivex.sample import LandscapeSample, ObjectiveSense


def test_sample_copies_and_freezes_caller_owned_arrays() -> None:
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    y = np.array([1.0, 13.0])
    sample = LandscapeSample(x, y, lower=[0.0, 0.0], upper=[3.0, 3.0])
    x[0, 0] = 99.0
    y[0] = 99.0

    assert sample.x[0, 0] == 0.0
    assert sample.y[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        sample.x[0, 0] = 2.0


def test_maximization_values_have_a_minimization_view() -> None:
    sample = LandscapeSample([[0.0], [1.0]], [2.0, 5.0], lower=[0.0], upper=[1.0], sense="maximize")

    assert sample.sense is ObjectiveSense.MAXIMIZE
    np.testing.assert_array_equal(sample.minimization_y, [-2.0, -5.0])
    assert not sample.minimization_y.flags.writeable


def test_fingerprint_depends_on_values_and_sense() -> None:
    minimizing = LandscapeSample([[0.0], [1.0]], [2.0, 5.0], [0.0], [1.0])
    maximizing = LandscapeSample([[0.0], [1.0]], [2.0, 5.0], [0.0], [1.0], ObjectiveSense.MAXIMIZE)

    assert minimizing.fingerprint != maximizing.fingerprint
    assert (
        minimizing.fingerprint
        == LandscapeSample([[0.0], [1.0]], [2.0, 5.0], [0.0], [1.0]).fingerprint
    )


@pytest.mark.parametrize(
    ("x", "y", "lower", "upper", "message"),
    [
        ([[0.0]], [np.nan], [0.0], [1.0], "finite"),
        ([[2.0]], [1.0], [0.0], [1.0], "within"),
        ([[0.0], [1.0]], [1.0], [0.0], [1.0], "shape"),
        ([[0.0]], [1.0], [1.0], [1.0], "strictly smaller"),
    ],
)
def test_sample_rejects_invalid_data(x, y, lower, upper, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LandscapeSample(x, y, lower, upper)
