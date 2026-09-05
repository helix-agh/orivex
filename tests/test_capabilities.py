from typing import cast

import pytest

from orivex.capabilities import AutogradSupport, FeatureCapability
from orivex.result import BackendName, DeviceType, FloatingDType


def valid_capability(**overrides: object) -> FeatureCapability:
    values = {
        "feature_name": "ela_distr.skewness",
        "backend": "torch",
        "autograd": "smooth",
        "devices": ("cpu", "cuda"),
        "dtypes": ("float32", "float64"),
    }
    values.update(overrides)
    return FeatureCapability(**values)  # type: ignore[arg-type]


def test_capability_accepts_literal_domains() -> None:
    capability = valid_capability()

    assert capability.backend == "torch"
    assert capability.autograd == "smooth"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", cast(BackendName, "jax"), "backend"),
        ("autograd", cast(AutogradSupport, "approximate"), "autograd"),
        ("devices", (cast(DeviceType, "tpu"),), "device"),
        ("dtypes", (cast(FloatingDType, "float16"),), "dtype"),
    ],
)
def test_capability_rejects_values_outside_literal_domains(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        valid_capability(**{field: value})
