import pytest

from bflacco.result import (
    ComputationResult,
    ExecutionMetadata,
    FeatureStatus,
    FeatureValue,
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
