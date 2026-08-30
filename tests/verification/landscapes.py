from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ArrayFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class AnalyticLandscape:
    """Vectorized objective with optional exact derivatives."""

    name: str
    dimension: int
    lower: FloatArray
    upper: FloatArray
    objective: ArrayFunction
    gradient: ArrayFunction
    hessian: ArrayFunction

    def evaluate(self, points: npt.ArrayLike) -> FloatArray:
        x = np.asarray(points, dtype=np.float64)
        if x.shape[-1] != self.dimension:
            raise ValueError(f"expected last axis of length {self.dimension}, got {x.shape}")
        return np.asarray(self.objective(x), dtype=np.float64)


def linear_landscape(coefficients: npt.ArrayLike, intercept: float = 0.0) -> AnalyticLandscape:
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    if coefficients_array.ndim != 1 or coefficients_array.size == 0:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    dimension = coefficients_array.size

    def objective(x: FloatArray) -> FloatArray:
        return np.asarray(x @ coefficients_array + intercept, dtype=np.float64)

    def gradient(x: FloatArray) -> FloatArray:
        return np.broadcast_to(coefficients_array, x.shape).copy()

    def hessian(x: FloatArray) -> FloatArray:
        return np.broadcast_to(
            np.zeros((dimension, dimension), dtype=np.float64),
            (*x.shape[:-1], dimension, dimension),
        ).copy()

    return AnalyticLandscape(
        name="linear",
        dimension=dimension,
        lower=np.full(dimension, -5.0),
        upper=np.full(dimension, 5.0),
        objective=objective,
        gradient=gradient,
        hessian=hessian,
    )


def sphere_landscape(dimension: int, shift: npt.ArrayLike | None = None) -> AnalyticLandscape:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    shift_array = (
        np.zeros(dimension, dtype=np.float64)
        if shift is None
        else np.asarray(shift, dtype=np.float64)
    )
    if shift_array.shape != (dimension,):
        raise ValueError(f"shift must have shape ({dimension},)")

    def objective(x: FloatArray) -> FloatArray:
        return np.asarray(np.sum((x - shift_array) ** 2, axis=-1), dtype=np.float64)

    def gradient(x: FloatArray) -> FloatArray:
        return 2.0 * (x - shift_array)

    def hessian(x: FloatArray) -> FloatArray:
        return np.broadcast_to(
            2.0 * np.eye(dimension, dtype=np.float64),
            (*x.shape[:-1], dimension, dimension),
        ).copy()

    return AnalyticLandscape(
        name="shifted_sphere" if np.any(shift_array) else "sphere",
        dimension=dimension,
        lower=np.full(dimension, -5.0),
        upper=np.full(dimension, 5.0),
        objective=objective,
        gradient=gradient,
        hessian=hessian,
    )
