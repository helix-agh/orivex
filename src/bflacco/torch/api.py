"""Public API for the optional tensor-native backend."""

from __future__ import annotations

import torch

from bflacco.api import DEFAULT_ENGINE as NUMPY_ENGINE
from bflacco.capabilities import FeatureCapability
from bflacco.result import ComputationResult
from bflacco.specs import FeatureSpec
from bflacco.torch.engine import TensorEngine
from bflacco.torch.features.distribution import CAPABILITIES as DISTRIBUTION_CAPABILITIES
from bflacco.torch.features.distribution import FEATURES as DISTRIBUTION_FEATURES
from bflacco.torch.features.distribution import INTERMEDIATES as DISTRIBUTION_INTERMEDIATES
from bflacco.torch.features.meta_model import CAPABILITIES as META_MODEL_CAPABILITIES
from bflacco.torch.features.meta_model import FEATURES as META_MODEL_FEATURES
from bflacco.torch.features.meta_model import INTERMEDIATES as META_MODEL_INTERMEDIATES
from bflacco.torch.sample import TensorLandscapeSample


class UnsupportedFeatureError(ValueError):
    """The requested feature exists in bflacco but not in the Torch backend."""


class UnsupportedFeatureDeviceError(ValueError):
    """The requested Torch feature is unavailable on the sample's device type."""


DEFAULT_ENGINE = TensorEngine(
    DISTRIBUTION_FEATURES + META_MODEL_FEATURES,
    DISTRIBUTION_INTERMEDIATES + META_MODEL_INTERMEDIATES,
)
CAPABILITIES = DISTRIBUTION_CAPABILITIES + META_MODEL_CAPABILITIES


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
    feature_names = _supported_feature_names(features)
    capabilities = {capability.feature_name: capability for capability in CAPABILITIES}
    unavailable = tuple(
        name for name in feature_names if sample.device_type not in capabilities[name].devices
    )
    if unavailable:
        joined = ", ".join(unavailable)
        raise UnsupportedFeatureDeviceError(
            f"features are not available on device type {sample.device_type!r}: {joined}"
        )
    return DEFAULT_ENGINE.compute(sample, feature_names)


def list_features() -> tuple[FeatureSpec, ...]:
    """Return the mathematical specifications implemented by this backend."""

    return tuple(DEFAULT_ENGINE.registry.get(name) for name in DEFAULT_ENGINE.registry.names())


def list_capabilities() -> tuple[FeatureCapability, ...]:
    """Return implementation-specific device, dtype, and autograd declarations."""

    return tuple(sorted(CAPABILITIES, key=lambda capability: capability.feature_name))
