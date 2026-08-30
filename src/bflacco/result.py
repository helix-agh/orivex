"""Feature values and execution metadata are intentionally separate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class FeatureStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FeatureValue:
    value: float | int | None
    status: FeatureStatus
    definition: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionMetadata:
    sample_fingerprint: str
    requested_features: tuple[str, ...]
    computed_intermediates: tuple[str, ...]
    runtime_seconds: float
    additional_objective_evaluations: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must not be negative")
        if self.additional_objective_evaluations < 0:
            raise ValueError("additional objective evaluations must not be negative")


@dataclass(frozen=True, slots=True)
class ComputationResult:
    values: Mapping[str, FeatureValue]
    metadata: ExecutionMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
