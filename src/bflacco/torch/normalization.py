"""Tensor-native objective preprocessing with preserved autograd history."""

from __future__ import annotations

import torch

from bflacco.normalization import YNormalization, normalization_definition


def normalize_objectives(y: torch.Tensor, mode: YNormalization) -> tuple[torch.Tensor, bool]:
    """Use observed min/max or population z-scores; constants map to connected zeros.

    Min-max is piecewise differentiable, with nonsmooth boundaries at tied extrema.
    The z-score expression is mathematically smooth on nonconstant samples, despite using
    a range-scaled intermediate to avoid overflow. No tensors leave their original device.
    """
    normalization_definition(mode)
    low, high = torch.amin(y), torch.amax(y)
    constant = bool((low == high).detach().item())
    if mode == "none":
        return y, constant
    if constant:
        return y - y, constant
    span = high - low
    if bool(torch.isfinite(span).detach().item()):
        values = (y - low) / span
    else:
        values = (y / 2 - low / 2) / (high / 2 - low / 2)
    if mode == "zscore":
        values = values - torch.mean(values)
        values = values / torch.sqrt(torch.mean(values.square()))
    return values, constant
