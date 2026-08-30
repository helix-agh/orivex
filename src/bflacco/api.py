"""Public convenience API backed by the built-in feature engine."""

from __future__ import annotations

import numpy as np

from .engine import Engine
from .features.distribution import FEATURES as DISTRIBUTION_FEATURES
from .features.distribution import INTERMEDIATES as DISTRIBUTION_INTERMEDIATES
from .features.meta_model import FEATURES as META_MODEL_FEATURES
from .features.meta_model import INTERMEDIATES as META_MODEL_INTERMEDIATES
from .result import ComputationResult
from .sample import LandscapeSample
from .specs import FeatureSpec

DEFAULT_ENGINE = Engine(
    DISTRIBUTION_FEATURES + META_MODEL_FEATURES,
    DISTRIBUTION_INTERMEDIATES + META_MODEL_INTERMEDIATES,
)


def compute(
    sample: LandscapeSample,
    features: str | tuple[str, ...] | list[str],
    *,
    rng: np.random.Generator | None = None,
) -> ComputationResult:
    return DEFAULT_ENGINE.compute(sample, features, rng=rng)


def list_features() -> tuple[FeatureSpec, ...]:
    return tuple(DEFAULT_ENGINE.registry.get(name) for name in DEFAULT_ENGINE.registry.names())
