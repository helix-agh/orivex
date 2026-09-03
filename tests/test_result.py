from typing import cast

import pytest

from bflacco.result import (
    BackendName,
    ComputationResult,
    DeviceType,
    ExecutionMetadata,
    FeatureStatus,
    FeatureValue,
    FloatingDType,
)


def test_values_and_execution_costs_are_separate_and_immutable() -> None:
    source = {"ela_distr.skewness": FeatureValue(0.5, FeatureStatus.OK, "corrected-v1")}
    result = ComputationResult(
        source,
        ExecutionMetadata("abc", ("ela_distr.skewness",), ("y.moments",), 0.01, 0),
    )
    source.clear()

    assert result.values["ela_distr.skewness"].value == 0.5
    assert result.metadata.runtime_seconds == 0.01
    with pytest.raises(TypeError):
        result.values["new"] = FeatureValue(1.0, FeatureStatus.OK, "corrected-v1")


def test_metadata_rejects_negative_costs() -> None:
    with pytest.raises(ValueError, match="runtime"):
        ExecutionMetadata("abc", (), (), -1.0, 0)
    with pytest.raises(ValueError, match="objective evaluations"):
        ExecutionMetadata("abc", (), (), 0.0, -1)


@pytest.mark.parametrize("workers", [0, -2])
def test_metadata_rejects_invalid_workers(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        ExecutionMetadata("abc", (), (), 0.0, 0, workers=workers)


def test_metadata_rejects_values_outside_literal_domains() -> None:
    with pytest.raises(ValueError, match="backend"):
        ExecutionMetadata("abc", (), (), 0.0, 0, backend=cast(BackendName, "jax"))
    with pytest.raises(ValueError, match="device"):
        ExecutionMetadata("abc", (), (), 0.0, 0, device=cast(DeviceType, "tpu"))
    with pytest.raises(ValueError, match="dtype"):
        ExecutionMetadata("abc", (), (), 0.0, 0, dtype=cast(FloatingDType, "float16"))
