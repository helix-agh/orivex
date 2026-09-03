"""Backend-specific feature implementation capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from bflacco.result import BackendName, DeviceType, FloatingDType

AutogradSupport: TypeAlias = Literal["smooth", "piecewise", "none"]


@dataclass(frozen=True, slots=True)
class FeatureCapability:
    """Execution properties that belong to an implementation, not its mathematics."""

    feature_name: str
    backend: BackendName
    autograd: AutogradSupport
    devices: tuple[DeviceType, ...]
    dtypes: tuple[FloatingDType, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("feature name must not be empty")
        if self.backend not in ("numpy", "torch"):
            raise ValueError(f"unsupported backend: {self.backend!r}")
        if self.autograd not in ("smooth", "piecewise", "none"):
            raise ValueError(f"unsupported autograd support: {self.autograd!r}")
        if not self.devices:
            raise ValueError("at least one device type is required")
        if not self.dtypes:
            raise ValueError("at least one floating dtype is required")
        if any(device not in ("cpu", "cuda", "mps") for device in self.devices):
            raise ValueError(f"unsupported device types: {self.devices!r}")
        if any(dtype not in ("float32", "float64") for dtype in self.dtypes):
            raise ValueError(f"unsupported floating dtypes: {self.dtypes!r}")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("device types must be unique")
        if len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError("floating dtypes must be unique")
