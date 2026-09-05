"""Tensor-native objective-distribution features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from orivex.capabilities import FeatureCapability
from orivex.engine import FeatureUnavailable
from orivex.features.distribution import KURTOSIS as NUMPY_KURTOSIS
from orivex.features.distribution import SKEWNESS as NUMPY_SKEWNESS
from orivex.planner import IntermediateSpec
from orivex.specs import InputRequirement
from orivex.torch.engine import (
    TensorComputationContext,
    TensorFeatureDefinition,
    TensorIntermediateDefinition,
)


@dataclass(frozen=True, slots=True)
class CenteredObjectives:
    observations: int
    values: torch.Tensor
    scale: torch.Tensor


def centered_objectives(context: TensorComputationContext) -> CenteredObjectives:
    """Mean-centered objectives rescaled to unit maximum magnitude.

    The deviations are divided by a *detached* ``max|y - mean|`` so the moment sums stay bounded at
    any finite objective scale. The scale cancels analytically, so detaching it keeps the feature's
    value and its gradient with respect to ``y`` unchanged while avoiding overflow of the raw
    powers. ``scale`` is ``0`` exactly for a constant objective.
    """
    y = context.y
    deviations = y - torch.mean(y, dim=-1, keepdim=True)
    # Constancy is decided on the objectives directly: a genuinely constant objective can still
    # acquire tiny nonzero deviations through mean rounding, and rescaling would otherwise amplify
    # that noise into apparently valid moments.
    constant = torch.amax(y, dim=-1, keepdim=True) == torch.amin(y, dim=-1, keepdim=True)
    raw_scale = torch.amax(torch.abs(deviations), dim=-1, keepdim=True)
    scale = torch.where(constant, torch.zeros_like(raw_scale), raw_scale)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale)).detach()
    return CenteredObjectives(y.shape[-1], deviations / divisor, scale.detach())


def _centered(context: TensorComputationContext) -> CenteredObjectives:
    value = context.intermediate("y.centered")
    if not isinstance(value, CenteredObjectives):
        raise TypeError("y.centered has an invalid runtime type")
    return value


def sum2(context: TensorComputationContext) -> torch.Tensor:
    centered = _centered(context).values
    return torch.sum(centered * centered, dim=-1)


def sum3(context: TensorComputationContext) -> torch.Tensor:
    return torch.sum(_centered(context).values ** 3, dim=-1)


def sum4(context: TensorComputationContext) -> torch.Tensor:
    return torch.sum(_centered(context).values ** 4, dim=-1)


def _sum(context: TensorComputationContext, name: str) -> torch.Tensor:
    value = context.intermediate(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def _is_zero(value: torch.Tensor) -> bool:
    return bool(value.detach().eq(0).item())


def skewness_type3(context: TensorComputationContext) -> torch.Tensor:
    centered = _centered(context)
    second = _sum(context, "y.sum2")
    third = _sum(context, "y.sum3")
    if centered.observations < 3:
        raise FeatureUnavailable("type-3 skewness requires at least 3 observations")
    if _is_zero(centered.scale):
        raise FeatureUnavailable("skewness is undefined for constant objective values")
    n = centered.observations
    type1 = math.sqrt(n) * third / second**1.5
    return type1 * ((n - 1) / n) ** 1.5


def kurtosis_type3(context: TensorComputationContext) -> torch.Tensor:
    centered = _centered(context)
    second = _sum(context, "y.sum2")
    fourth = _sum(context, "y.sum4")
    if centered.observations < 4:
        raise FeatureUnavailable("type-3 kurtosis requires at least 4 observations")
    if _is_zero(centered.scale):
        raise FeatureUnavailable("kurtosis is undefined for constant objective values")
    n = centered.observations
    ratio = n * fourth / second**2
    return ratio * (1.0 - 1.0 / n) ** 2 - 3.0


INTERMEDIATES = (
    TensorIntermediateDefinition(
        IntermediateSpec("y.centered", (), frozenset({InputRequirement.Y})),
        centered_objectives,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec("y.sum2", ("y.centered",), frozenset()),
        sum2,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec("y.sum3", ("y.centered",), frozenset()),
        sum3,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec("y.sum4", ("y.centered",), frozenset()),
        sum4,
    ),
)

FEATURES = (
    TensorFeatureDefinition(NUMPY_SKEWNESS.spec, skewness_type3),
    TensorFeatureDefinition(NUMPY_KURTOSIS.spec, kurtosis_type3),
)

CAPABILITIES = (
    FeatureCapability(
        feature_name=NUMPY_SKEWNESS.spec.name,
        backend="torch",
        autograd="smooth",
        devices=("cpu", "cuda", "mps"),
        dtypes=("float32", "float64"),
        notes=("Differentiable where the second central moment is non-zero.",),
    ),
    FeatureCapability(
        feature_name=NUMPY_KURTOSIS.spec.name,
        backend="torch",
        autograd="smooth",
        devices=("cpu", "cuda", "mps"),
        dtypes=("float32", "float64"),
        notes=("Differentiable where the second central moment is non-zero.",),
    ),
)
