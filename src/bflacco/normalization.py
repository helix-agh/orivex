"""Versioned objective preprocessing, applied after objective-sense canonicalization."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

YNormalization: TypeAlias = Literal["none", "minmax", "zscore"]


def normalization_definition(mode: YNormalization) -> str:
    if mode not in ("none", "minmax", "zscore"):
        raise ValueError(f"unsupported y_normalization: {mode!r}")
    return f"objective-{mode}-v1"


def normalize_objectives(y: np.ndarray, mode: YNormalization) -> tuple[np.ndarray, bool]:
    """Return transformed values and an exact constant-objective diagnostic.

    Min-max uses the observed range; z-score uses population standard deviation (ddof=0).
    Constant samples map to zero in either normalized mode. No epsilon is added. Halving
    opposite-sign extremes avoids overflow of the range; centering before division retains
    close representable values when the original observations have a large common offset.
    """
    normalization_definition(mode)
    low, high = np.min(y), np.max(y)
    constant = bool(low == high)
    if mode == "none":
        return y, constant
    if constant:
        values = np.zeros_like(y)
    else:
        with np.errstate(over="ignore"):
            span = high - low
        values = (y - low) / span if np.isfinite(span) else (y / 2 - low / 2) / (high / 2 - low / 2)
        if mode == "zscore":
            values -= np.mean(values)
            values /= np.sqrt(np.mean(values * values))
    values.flags.writeable = False
    return values, constant
