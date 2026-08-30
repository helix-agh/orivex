from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def paired_row_permutation(
    x: npt.ArrayLike, y: npt.ArrayLike, permutation: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    permutation_array = np.asarray(permutation, dtype=np.int64)
    if x_array.ndim != 2 or y_array.shape != (x_array.shape[0],):
        raise ValueError("expected X with shape (n, d) and y with shape (n,)")
    if not np.array_equal(np.sort(permutation_array), np.arange(x_array.shape[0])):
        raise ValueError("permutation must contain every row index exactly once")
    return x_array[permutation_array], y_array[permutation_array]


def variable_permutation(x: npt.ArrayLike, permutation: npt.ArrayLike) -> FloatArray:
    x_array = np.asarray(x, dtype=np.float64)
    permutation_array = np.asarray(permutation, dtype=np.int64)
    if x_array.ndim != 2:
        raise ValueError("expected X with shape (n, d)")
    if not np.array_equal(np.sort(permutation_array), np.arange(x_array.shape[1])):
        raise ValueError("permutation must contain every variable index exactly once")
    return x_array[:, permutation_array]


def translate_y(y: npt.ArrayLike, offset: float) -> FloatArray:
    return np.asarray(y, dtype=np.float64) + offset


def positively_scale_y(y: npt.ArrayLike, scale: float) -> FloatArray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("objective scale must be finite and positive")
    return np.asarray(y, dtype=np.float64) * scale


def orthogonally_rotate_x(x: npt.ArrayLike, rotation: npt.ArrayLike) -> FloatArray:
    x_array = np.asarray(x, dtype=np.float64)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    if x_array.ndim != 2:
        raise ValueError("expected X with shape (n, d)")
    dimension = x_array.shape[1]
    if rotation_array.shape != (dimension, dimension):
        raise ValueError(f"rotation must have shape ({dimension}, {dimension})")
    np.testing.assert_allclose(
        rotation_array.T @ rotation_array,
        np.eye(dimension),
        rtol=1e-12,
        atol=1e-12,
        err_msg="rotation must be orthogonal",
    )
    return x_array @ rotation_array.T
