import numpy as np
import pytest

from verification.landscapes import linear_landscape, sphere_landscape


def test_linear_landscape_has_exact_values_and_derivatives() -> None:
    landscape = linear_landscape([2.0, -3.0], intercept=4.0)
    points = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 1.0]])

    np.testing.assert_array_equal(landscape.evaluate(points), [4.0, 0.0, -1.0])
    np.testing.assert_array_equal(
        landscape.gradient(points),
        np.array([[2.0, -3.0], [2.0, -3.0], [2.0, -3.0]]),
    )
    np.testing.assert_array_equal(landscape.hessian(points), np.zeros((3, 2, 2)))


def test_shifted_sphere_has_known_optimum_gradient_and_hessian() -> None:
    landscape = sphere_landscape(2, shift=[1.0, -2.0])
    points = np.array([[1.0, -2.0], [2.0, 0.0]])

    np.testing.assert_array_equal(landscape.evaluate(points), [0.0, 5.0])
    np.testing.assert_array_equal(landscape.gradient(points), [[0.0, 0.0], [2.0, 4.0]])
    np.testing.assert_array_equal(
        landscape.hessian(points),
        np.broadcast_to(2.0 * np.eye(2), (2, 2, 2)),
    )


def test_analytic_landscape_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="expected last axis"):
        sphere_landscape(3).evaluate([[1.0, 2.0]])
