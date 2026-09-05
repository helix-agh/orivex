"""Tensor-native landscape observations that preserve autograd history."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch

from orivex.result import DeviceType, FloatingDType
from orivex.sample import ObjectiveSense, ObjectiveSenseName

TensorValue: TypeAlias = torch.Tensor | int | float | Sequence[int | float]
_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SUPPORTED_DEVICES: tuple[DeviceType, ...] = ("cpu", "cuda", "mps")


def _clone_bound(value: TensorValue, *, reference: torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != reference.device:
            raise ValueError(f"{name} must be on device {reference.device}")
        if value.dtype != reference.dtype:
            raise ValueError(f"{name} must have dtype {reference.dtype}")
        return value.clone()
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device).clone()


def _fingerprint(
    tensors: tuple[torch.Tensor, ...],
    sense: ObjectiveSense,
) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        snapshot = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(snapshot.shape)).encode("ascii"))
        digest.update(str(snapshot.dtype).encode("ascii"))
        digest.update(snapshot.numpy().tobytes(order="C"))
    digest.update(sense.value.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class TensorLandscapeSample:
    """A validated single-landscape tensor sample with mutation detection."""

    x: torch.Tensor
    y: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    sense: ObjectiveSense
    _minimization_y: torch.Tensor
    _fingerprint: str
    _versions: tuple[int, ...]

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lower: TensorValue,
        upper: TensorValue,
        sense: ObjectiveSense | ObjectiveSenseName = ObjectiveSense.MINIMIZE,
    ) -> None:
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("X and y must be torch tensors")
        if x.dtype not in _SUPPORTED_DTYPES or y.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("X and y must have dtype torch.float32 or torch.float64")
        if x.dtype != y.dtype:
            raise ValueError("X and y must have the same dtype")
        if x.device != y.device:
            raise ValueError("X and y must be on the same device")
        if x.device.type not in _SUPPORTED_DEVICES:
            raise ValueError(f"unsupported device type: {x.device.type!r}")

        x_array = x.clone()
        y_array = y.clone()
        lower_array = _clone_bound(lower, reference=x_array, name="lower")
        upper_array = _clone_bound(upper, reference=x_array, name="upper")
        sense_value = ObjectiveSense(sense)

        if x_array.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {tuple(x_array.shape)}")
        if y_array.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {tuple(y_array.shape)}")
        observations, dimension = x_array.shape
        if observations == 0 or dimension == 0:
            raise ValueError("X must contain at least one observation and one variable")
        if y_array.shape != (observations,):
            raise ValueError(f"y must have shape ({observations},), got {tuple(y_array.shape)}")
        if lower_array.shape != (dimension,) or upper_array.shape != (dimension,):
            raise ValueError(f"bounds must both have shape ({dimension},)")
        if not bool(torch.isfinite(x_array).all().item()):
            raise ValueError("X must contain only finite values")
        if not bool(torch.isfinite(y_array).all().item()):
            raise ValueError("y must contain only finite values")
        if not bool(torch.isfinite(lower_array).all().item()):
            raise ValueError("lower must contain only finite values")
        if not bool(torch.isfinite(upper_array).all().item()):
            raise ValueError("upper must contain only finite values")
        if not bool((lower_array < upper_array).all().item()):
            raise ValueError("every lower bound must be strictly smaller than its upper bound")
        inside = (x_array >= lower_array) & (x_array <= upper_array)
        if not bool(inside.all().item()):
            raise ValueError("all observations must lie within the inclusive box bounds")

        minimization_y = y_array if sense_value is ObjectiveSense.MINIMIZE else -y_array
        tensors = (x_array, y_array, lower_array, upper_array, minimization_y)
        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "lower", lower_array)
        object.__setattr__(self, "upper", upper_array)
        object.__setattr__(self, "sense", sense_value)
        object.__setattr__(self, "_minimization_y", minimization_y)
        object.__setattr__(self, "_fingerprint", _fingerprint(tensors[:4], sense_value))
        object.__setattr__(self, "_versions", tuple(tensor._version for tensor in tensors))

    @property
    def n_observations(self) -> int:
        return self.x.shape[0]

    @property
    def dimension(self) -> int:
        return self.x.shape[1]

    @property
    def minimization_y(self) -> torch.Tensor:
        return self._minimization_y

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def device_type(self) -> DeviceType:
        device_type = self.x.device.type
        if device_type == "cpu":
            return "cpu"
        if device_type == "cuda":
            return "cuda"
        return "mps"

    @property
    def device_index(self) -> int | None:
        return self.x.device.index

    @property
    def dtype_name(self) -> FloatingDType:
        return "float32" if self.x.dtype is torch.float32 else "float64"

    def validate_unchanged(self) -> None:
        tensors = (self.x, self.y, self.lower, self.upper, self._minimization_y)
        versions = tuple(tensor._version for tensor in tensors)
        if versions != self._versions:
            raise RuntimeError("TensorLandscapeSample tensors must not be modified in place")
