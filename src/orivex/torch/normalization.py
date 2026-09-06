"""Tensor-native objective preprocessing with preserved autograd history."""

from __future__ import annotations

import torch

from orivex.normalization import YNormalization, normalization_definition


def normalize_objectives(y: torch.Tensor, mode: YNormalization) -> tuple[torch.Tensor, bool]:
    """Use observed min/max; None preserves raw objectives.

    Min-max is piecewise differentiable, with nonsmooth boundaries at tied extrema.
    Constants map to connected zeros. No tensors leave their original device.
    """
    normalization_definition(mode)
    low, high = torch.amin(y), torch.amax(y)
    constant = bool((low == high).detach().item())
    if mode is None:
        return y, constant
    if constant:
        return y - y, constant
    span = high - low
    if bool(torch.isfinite(span).detach().item()):
        values = (y - low) / span
    else:
        values = (y / 2 - low / 2) / (high / 2 - low / 2)
    return values, constant
