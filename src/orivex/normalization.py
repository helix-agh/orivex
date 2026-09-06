"""Versioned objective preprocessing, applied after objective-sense canonicalization."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

YNormalization: TypeAlias = Literal["minmax"] | None


def normalization_definition(mode: YNormalization) -> str:
    if mode is None:
        return "objective-none-v1"
    if mode != "minmax":
        raise ValueError(f"unsupported y_normalization: {mode!r}; expected 'minmax' or None")
    return "objective-minmax-v1"


def normalize_objectives(y: np.ndarray, mode: YNormalization) -> tuple[np.ndarray, bool]:
    """Return transformed values and an exact constant-objective diagnostic.

    Min-max uses the observed range, following Prager and Trautmann (2023), Section 5,
    https://doi.org/10.1007/978-3-031-30229-9_27. None preserves raw objectives.
    Constant samples map to zero in min-max mode. No epsilon is added. Halving
    opposite-sign extremes avoids overflow of the range; centering before division retains
    close representable values when the original observations have a large common offset.
    """
    normalization_definition(mode)
    low, high = np.min(y), np.max(y)
    constant = bool(low == high)
    if mode is None:
        return y, constant
    if constant:
        values = np.zeros_like(y)
    else:
        with np.errstate(over="ignore"):
            span = high - low
        values = (y - low) / span if np.isfinite(span) else (y / 2 - low / 2) / (high / 2 - low / 2)
    values.flags.writeable = False
    return values, constant
