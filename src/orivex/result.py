"""Feature values and execution metadata are intentionally separate."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar

from orivex.normalization import YNormalization, normalization_definition
from orivex.options import FeatureOptions, resolve_options

FeatureScalar = TypeVar("FeatureScalar")
BackendName: TypeAlias = Literal["numpy", "torch"]
DeviceType: TypeAlias = Literal["cpu", "cuda", "mps"]
FloatingDType: TypeAlias = Literal["float32", "float64"]


class FeatureStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FeatureValue(Generic[FeatureScalar]):
    value: FeatureScalar | None
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
    workers: int = 1
    backend: BackendName = "numpy"
    device: DeviceType = "cpu"
    device_index: int | None = None
    dtype: FloatingDType = "float64"
    y_normalization: YNormalization = "none"
    constant_objective: bool = False
    options: FeatureOptions = field(default_factory=resolve_options)

    @property
    def y_normalization_definition(self) -> str:
        return normalization_definition(self.y_normalization)

    @property
    def preprocessing_fingerprint(self) -> str:
        """Cache-key component identifying raw input, backend, dtype, and preprocessing.

        A full result cache must additionally include feature definitions and execution options.
        """
        identity = (
            self.sample_fingerprint,
            self.backend,
            self.dtype,
            self.y_normalization_definition,
        )
        return hashlib.sha256("\0".join(identity).encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        normalization_definition(self.y_normalization)
        object.__setattr__(self, "options", resolve_options(self.options))
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must not be negative")
        if self.additional_objective_evaluations < 0:
            raise ValueError("additional objective evaluations must not be negative")
        if self.workers == 0 or self.workers < -1:
            raise ValueError("workers must be -1 or a positive integer")
        if self.backend not in ("numpy", "torch"):
            raise ValueError(f"unsupported backend: {self.backend!r}")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"unsupported device type: {self.device!r}")
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("device index must not be negative")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"unsupported floating dtype: {self.dtype!r}")


@dataclass(frozen=True, slots=True)
class ComputationResult(Generic[FeatureScalar]):
    values: Mapping[str, FeatureValue[FeatureScalar]]
    metadata: ExecutionMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
