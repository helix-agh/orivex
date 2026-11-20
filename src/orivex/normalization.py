"""Versioned objective preprocessing, applied after objective-sense canonicalization."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

YNormalization: TypeAlias = Literal["minmax", "rank"] | None


def normalization_definition(mode: YNormalization) -> str:
    if mode is None:
        return "objective-none-v1"
    if mode == "minmax":
        return "objective-minmax-v1"
    if mode == "rank":
        return "objective-rank-v1"
    raise ValueError(f"unsupported y_normalization: {mode!r}; expected 'minmax', 'rank' or None")


def normalize_objectives(y: np.ndarray, mode: YNormalization) -> tuple[np.ndarray, bool]:
    """Return transformed values and an exact constant-objective diagnostic.

    Min-max uses the observed range, following Prager and Trautmann (2023), Section 5,
    https://doi.org/10.1007/978-3-031-30229-9_27. Rank maps each value to its min-max
    normalized average rank, ``(r - 1) / (n - 1)``, following the rank-based extraction of
    Tanaka, Tanaka and Hatanaka (2026), Section 4.1, https://arxiv.org/abs/2611.20440.
    None preserves raw objectives. Constant samples map to zero in both modes. No epsilon
    is added. Halving opposite-sign extremes avoids overflow of the range; centering before
    division retains close representable values when the original observations have a
    large common offset.
    """
    normalization_definition(mode)
    low, high = np.min(y), np.max(y)
    constant = bool(low == high)
    if mode is None:
        return y, constant
    if constant:
        values = np.zeros_like(y)
    elif mode == "rank":
        values = rank_positions(y) / (y.size - 1)
    else:
        with np.errstate(over="ignore"):
            span = high - low
        values = (y - low) / span if np.isfinite(span) else (y / 2 - low / 2) / (high / 2 - low / 2)
    values.flags.writeable = False
    return values, constant


def rank_positions(y: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, ``rankdata(y, method="average") - 1``.

    ``less + (equal - 1) / 2`` is computed as ``(less + less_or_equal - 1) / 2`` from
    integer counts, so the Torch backend can reproduce it exactly.
    """
    ordered = np.sort(y)
    less = np.searchsorted(ordered, y, side="left")
    less_or_equal = np.searchsorted(ordered, y, side="right")
    return (less + less_or_equal - 1).astype(y.dtype) / 2
