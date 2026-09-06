"""Validated, immutable landscape observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ObjectiveSenseName: TypeAlias = Literal["minimize", "maximize"]


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def _readonly_float_array(value: npt.ArrayLike, *, dimensions: int, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != dimensions:
        raise ValueError(f"{name} must be {dimensions}-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _content_digest(arrays: tuple[FloatArray, ...], sense: ObjectiveSense) -> str:
    """Hash the numerical contents of ``arrays`` together with the objective sense."""

    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(sense.value.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class LandscapeSample:
    """Paired decision and objective observations with explicit box bounds."""

    x: FloatArray
    y: FloatArray
    lower: FloatArray
    upper: FloatArray
    sense: ObjectiveSense
    _minimization_y: FloatArray
    _fingerprint: str
    _integrity: str

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        lower: npt.ArrayLike,
        upper: npt.ArrayLike,
        sense: ObjectiveSense | ObjectiveSenseName = ObjectiveSense.MINIMIZE,
    ) -> None:
        x_array = _readonly_float_array(x, dimensions=2, name="X")
        y_array = _readonly_float_array(y, dimensions=1, name="y")
        lower_array = _readonly_float_array(lower, dimensions=1, name="lower")
        upper_array = _readonly_float_array(upper, dimensions=1, name="upper")
        sense_value = ObjectiveSense(sense)

        observations, dimension = x_array.shape
        if observations == 0 or dimension == 0:
            raise ValueError("X must contain at least one observation and one variable")
        if y_array.shape != (observations,):
            raise ValueError(f"y must have shape ({observations},), got {y_array.shape}")
        if lower_array.shape != (dimension,) or upper_array.shape != (dimension,):
            raise ValueError(f"bounds must both have shape ({dimension},)")
        if not np.all(lower_array < upper_array):
            raise ValueError("every lower bound must be strictly smaller than its upper bound")
        if np.any(x_array < lower_array) or np.any(x_array > upper_array):
            raise ValueError("all observations must lie within the inclusive box bounds")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "lower", lower_array)
        object.__setattr__(self, "upper", upper_array)
        object.__setattr__(self, "sense", sense_value)

        minimization_y = y_array if sense_value is ObjectiveSense.MINIMIZE else -y_array
        minimization_y.flags.writeable = False
        object.__setattr__(self, "_minimization_y", minimization_y)

        inputs = (x_array, y_array, lower_array, upper_array)
        object.__setattr__(self, "_fingerprint", _content_digest(inputs, sense_value))
        object.__setattr__(
            self, "_integrity", _content_digest((*inputs, minimization_y), sense_value)
        )

    @property
    def n_observations(self) -> int:
        return self.x.shape[0]

    @property
    def dimension(self) -> int:
        return self.x.shape[1]

    @property
    def minimization_y(self) -> FloatArray:
        """Objective observations transformed to minimization convention."""

        return self._minimization_y

    @property
    def fingerprint(self) -> str:
        """Stable checksum of numerical inputs and objective sense."""

        return self._fingerprint

    def validate_unchanged(self) -> None:
        """Detect in-place mutation of the exposed arrays since construction.

        The arrays are handed out read-only, but callers can re-enable the
        ``writeable`` flag on the owning storage (or reach it through an alias)
        and mutate the values in place. That would leave :attr:`fingerprint`
        stale, breaking the provenance / cache-key contract, so mutation is
        detected here before the sample is consumed.
        """

        current = _content_digest(
            (self.x, self.y, self.lower, self.upper, self._minimization_y), self.sense
        )
        if current != self._integrity:
            raise RuntimeError("LandscapeSample arrays must not be modified in place")
