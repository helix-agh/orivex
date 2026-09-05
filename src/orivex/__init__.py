"""Better FLACCO: selective and verified exploratory landscape analysis."""

from importlib.metadata import PackageNotFoundError, version

from orivex.api import compute, list_features
from orivex.capabilities import AutogradSupport, FeatureCapability
from orivex.normalization import YNormalization
from orivex.result import BackendName, DeviceType, FloatingDType
from orivex.sample import LandscapeSample, ObjectiveSense, ObjectiveSenseName
from orivex.specs import (
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
    __version__ = version("orivex")
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
