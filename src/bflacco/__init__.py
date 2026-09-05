"""Better FLACCO: selective and verified exploratory landscape analysis."""

from importlib.metadata import PackageNotFoundError, version

from bflacco.api import compute, list_features
from bflacco.capabilities import AutogradSupport, FeatureCapability
from bflacco.normalization import YNormalization
from bflacco.result import BackendName, DeviceType, FloatingDType
from bflacco.sample import LandscapeSample, ObjectiveSense, ObjectiveSenseName
from bflacco.specs import (
    CostModel,
    CostTier,
    FeatureSpec,
    InputRequirement,
    InvarianceBehavior,
    InvarianceClaim,
    MetricKind,
    Reference,
    Transformation,
)

try:
    __version__ = version("bflacco")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.0.0"

__all__ = [
    "AutogradSupport",
    "BackendName",
    "CostModel",
    "CostTier",
    "DeviceType",
    "FeatureCapability",
    "FeatureSpec",
    "FloatingDType",
    "InputRequirement",
    "InvarianceBehavior",
    "InvarianceClaim",
    "LandscapeSample",
    "MetricKind",
    "ObjectiveSense",
    "ObjectiveSenseName",
    "Reference",
    "Transformation",
    "YNormalization",
    "__version__",
    "compute",
    "list_features",
]
