"""Better FLACCO: selective and verified exploratory landscape analysis."""

from importlib.metadata import PackageNotFoundError, version

from .api import compute, list_features
from .sample import LandscapeSample, ObjectiveSense
from .specs import (
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
    "CostModel",
    "CostTier",
    "FeatureSpec",
    "InputRequirement",
    "InvarianceBehavior",
    "InvarianceClaim",
    "LandscapeSample",
    "MetricKind",
    "ObjectiveSense",
    "Reference",
    "Transformation",
    "__version__",
    "compute",
    "list_features",
]
