"""Public convenience API backed by the built-in feature engine."""

from __future__ import annotations

import numpy as np

from orivex.engine import Engine
from orivex.features.distribution import FEATURES as DISTRIBUTION_FEATURES
from orivex.features.distribution import INTERMEDIATES as DISTRIBUTION_INTERMEDIATES
from orivex.features.information_content import FEATURES as INFORMATION_CONTENT_FEATURES
from orivex.features.information_content import INTERMEDIATES as INFORMATION_CONTENT_INTERMEDIATES
from orivex.features.meta_model import FEATURES as META_MODEL_FEATURES
from orivex.features.meta_model import INTERMEDIATES as META_MODEL_INTERMEDIATES
from orivex.features.nearest_better import FEATURES as NEAREST_BETTER_FEATURES
from orivex.features.nearest_better import INTERMEDIATES as NEAREST_BETTER_INTERMEDIATES
from orivex.normalization import YNormalization
from orivex.result import ComputationResult
from orivex.sample import LandscapeSample
from orivex.specs import FeatureSpec

DEFAULT_ENGINE = Engine(
    DISTRIBUTION_FEATURES
    + META_MODEL_FEATURES
    + INFORMATION_CONTENT_FEATURES
    + NEAREST_BETTER_FEATURES,
    DISTRIBUTION_INTERMEDIATES
    + META_MODEL_INTERMEDIATES
    + INFORMATION_CONTENT_INTERMEDIATES
    + NEAREST_BETTER_INTERMEDIATES,
)


def compute(
    sample: LandscapeSample,
    features: str | tuple[str, ...] | list[str],
    *,
    rng: np.random.Generator | None = None,
    workers: int = 1,
    y_normalization: YNormalization = "minmax",
) -> ComputationResult:
    """Compute features from canonical objectives, min-max normalized by default.

    ``y_normalization`` is ``"minmax"`` (observed range), ``"zscore"`` (population
    standard deviation), or ``"none"`` (raw canonical objectives). Normalized constant
    objectives map to zero. The original sample is preserved; preprocessing is recorded
    separately from mathematical feature definitions in result metadata.

    ``workers=1`` is the predictable default. Pass ``-1`` to let supporting kernels use all
    available CPUs, or a positive integer to set an upper worker count. Kernels that do not
    support parallel execution ignore this setting.
    """
    return DEFAULT_ENGINE.compute(
        sample, features, rng=rng, workers=workers, y_normalization=y_normalization
    )


def list_features() -> tuple[FeatureSpec, ...]:
    return tuple(DEFAULT_ENGINE.registry.get(name) for name in DEFAULT_ENGINE.registry.names())
