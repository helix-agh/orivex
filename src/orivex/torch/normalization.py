"""Tensor-native objective preprocessing with preserved autograd history."""

from __future__ import annotations

import torch

from orivex.normalization import YNormalization, normalization_definition


def normalize_objectives(y: torch.Tensor, mode: YNormalization) -> tuple[torch.Tensor, bool]:
    """Use observed min/max or average ranks; None preserves raw objectives.

    Min-max is piecewise differentiable, with nonsmooth boundaries at tied extrema.
    Rank values are piecewise constant in ``y``: they stay connected to the graph, but their
    gradient with respect to ``y`` is zero. Constants map to connected zeros. No tensors
    leave their original device.
    """
    normalization_definition(mode)
    low, high = torch.amin(y), torch.amax(y)
    constant = bool((low == high).detach().item())
    if mode is None:
        return y, constant
    if constant:
        return y - y, constant
    if mode == "rank":
        ordered = torch.sort(y.detach()).values
        less = torch.searchsorted(ordered, y.detach(), right=False)
        less_or_equal = torch.searchsorted(ordered, y.detach(), right=True)
        positions = (less + less_or_equal - 1).to(y.dtype) / 2
        return (y - y) + positions / (y.numel() - 1), constant
    span = high - low
    if bool(torch.isfinite(span).detach().item()):
        values = (y - low) / span
    else:
        values = (y / 2 - low / 2) / (high / 2 - low / 2)
    return values, constant
