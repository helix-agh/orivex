import numpy as np
import pytest

from verification.transforms import (
    orthogonally_rotate_x,
    paired_row_permutation,
    positively_scale_y,
    translate_y,
    variable_permutation,
)


def test_paired_row_permutation_preserves_pairs() -> None:
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    y = np.array([100.0, 200.0, 300.0])

    transformed_x, transformed_y = paired_row_permutation(x, y, [2, 0, 1])

    np.testing.assert_array_equal(transformed_x[:, 0], [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(transformed_y, [300.0, 100.0, 200.0])


def test_variable_permutation_reorders_columns_only() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(
        variable_permutation(x, [2, 0, 1]),
        [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]],
    )


def test_y_transformations_are_explicit() -> None:
    y = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(translate_y(y, 3.0), [2.0, 3.0, 5.0])
    np.testing.assert_array_equal(positively_scale_y(y, 2.5), [-2.5, 0.0, 5.0])


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_positive_scaling_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        positively_scale_y([1.0, 2.0], scale)


def test_orthogonal_rotation_preserves_pairwise_distances() -> None:
    x = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    rotated = orthogonally_rotate_x(x, rotation)

    original_distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    rotated_distances = np.linalg.norm(rotated[:, None, :] - rotated[None, :, :], axis=-1)
    np.testing.assert_allclose(rotated_distances, original_distances, rtol=1e-14, atol=1e-14)


def test_rotation_rejects_nonorthogonal_matrix() -> None:
    with pytest.raises(AssertionError, match="rotation must be orthogonal"):
        orthogonally_rotate_x([[1.0, 2.0]], [[1.0, 1.0], [0.0, 1.0]])
