"""Public API for the optional tensor-native backend."""

from __future__ import annotations

from dataclasses import replace

import torch

from orivex.api import DEFAULT_ENGINE as NUMPY_ENGINE
from orivex.capabilities import FeatureCapability
from orivex.normalization import YNormalization, normalization_definition
from orivex.result import ComputationResult
from orivex.specs import FeatureSpec
from orivex.torch.engine import TensorEngine
from orivex.torch.features.distribution import CAPABILITIES as DISTRIBUTION_CAPABILITIES
from orivex.torch.features.distribution import FEATURES as DISTRIBUTION_FEATURES
from orivex.torch.features.distribution import INTERMEDIATES as DISTRIBUTION_INTERMEDIATES
from orivex.torch.features.meta_model import CAPABILITIES as META_MODEL_CAPABILITIES
from orivex.torch.features.meta_model import FEATURES as META_MODEL_FEATURES
from orivex.torch.features.meta_model import INTERMEDIATES as META_MODEL_INTERMEDIATES
from orivex.torch.sample import TensorLandscapeSample


class UnsupportedFeatureError(ValueError):
    """The requested feature exists in orivex but not in the Torch backend."""


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
    *,
    y_normalization: YNormalization = "minmax",
) -> ComputationResult[torch.Tensor]:
    """Compute tensor features with min-max objective normalization by default.

    ``y_normalization="none"`` preserves raw canonical objectives; ``"zscore"`` uses
    population standard deviation. Min-max preprocessing is piecewise differentiable.
    All modes preserve the sample's tensors, dtype, device, and autograd history.
    """

    if not isinstance(sample, TensorLandscapeSample):
        raise TypeError("sample must be a orivex.torch.TensorLandscapeSample")
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
    return DEFAULT_ENGINE.compute(sample, feature_names, y_normalization=y_normalization)


def list_features() -> tuple[FeatureSpec, ...]:
    """Return the mathematical specifications implemented by this backend."""

    return tuple(DEFAULT_ENGINE.registry.get(name) for name in DEFAULT_ENGINE.registry.names())


def list_capabilities(
    *,
    y_normalization: YNormalization = "minmax",
) -> tuple[FeatureCapability, ...]:
    """Return capabilities including a conservative preprocessing autograd guarantee."""

    normalization_definition(y_normalization)
    capabilities = CAPABILITIES
    if y_normalization == "minmax":
        capabilities = tuple(
            replace(
                item,
                autograd="piecewise",
                notes=(
                    *item.notes,
                    "Min-max preprocessing is piecewise differentiable at extrema.",
                ),
            )
            for item in capabilities
        )
    return tuple(sorted(capabilities, key=lambda capability: capability.feature_name))
