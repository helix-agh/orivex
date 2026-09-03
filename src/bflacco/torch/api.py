"""Public API for the optional tensor-native backend."""

from __future__ import annotations

import torch

from bflacco.api import DEFAULT_ENGINE as NUMPY_ENGINE
from bflacco.capabilities import FeatureCapability
from bflacco.result import ComputationResult
from bflacco.specs import FeatureSpec
from bflacco.torch.engine import TensorEngine
from bflacco.torch.features.distribution import CAPABILITIES, FEATURES, INTERMEDIATES
from bflacco.torch.sample import TensorLandscapeSample


class UnsupportedFeatureError(ValueError):
    """The requested feature exists in bflacco but not in the Torch backend."""


DEFAULT_ENGINE = TensorEngine(FEATURES, INTERMEDIATES)


def _supported_feature_names(
    selectors: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    selected = NUMPY_ENGINE.registry.select(selectors)
    supported = set(DEFAULT_ENGINE.registry.names())
    unavailable = tuple(spec.name for spec in selected if spec.name not in supported)
    if unavailable:
        joined = ", ".join(unavailable)
        raise UnsupportedFeatureError(f"features are not available in the Torch backend: {joined}")
    return tuple(spec.name for spec in selected)


def compute(
    sample: TensorLandscapeSample,
    features: str | tuple[str, ...] | list[str],
) -> ComputationResult[torch.Tensor]:
    """Compute selected tensor-native features without detaching their outputs."""

    if not isinstance(sample, TensorLandscapeSample):
        raise TypeError("sample must be a bflacco.torch.TensorLandscapeSample")
    return DEFAULT_ENGINE.compute(sample, _supported_feature_names(features))


def list_features() -> tuple[FeatureSpec, ...]:
    """Return the mathematical specifications implemented by this backend."""

    return tuple(DEFAULT_ENGINE.registry.get(name) for name in DEFAULT_ENGINE.registry.names())


def list_capabilities() -> tuple[FeatureCapability, ...]:
    """Return implementation-specific device, dtype, and autograd declarations."""

    return tuple(sorted(CAPABILITIES, key=lambda capability: capability.feature_name))
