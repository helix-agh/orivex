"""Validation and immutable snapshots of options keyed by feature group."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

FeatureOptions: TypeAlias = Mapping[str, Mapping[str, float]]


def _path(group: str, name: str) -> str:
    return f"options[{group!r}][{name!r}]"


def _as_number(value: object) -> float | None:
    """Real, non-boolean scalars only: ``True`` is a valid ``Real`` but never a valid option."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _unit_interval(group: str, name: str, value: object) -> float:
    number = _as_number(value)
    if number is None or not 0 < number <= 1:
        raise ValueError(f"{_path(group, name)} must be finite in (0, 1]")
    return number


def _positive(group: str, name: str, value: object) -> float:
    number = _as_number(value)
    if number is None or not 0 < number < math.inf:
        raise ValueError(f"{_path(group, name)} must be finite and positive")
    return number


_Validator: TypeAlias = Callable[[str, str, object], float]

DEFAULTS: Mapping[str, Mapping[str, float]] = {
    "fitness_distance": {"proportion_of_best": 0.1},
    "nbc_soft": {"temperature": 0.003, "rank_temperature": 0.5},
}

_VALIDATORS: Mapping[str, Mapping[str, _Validator]] = {
    "fitness_distance": {"proportion_of_best": _unit_interval},
    "nbc_soft": {"temperature": _positive, "rank_temperature": _positive},
}


def resolve_options(options: FeatureOptions | None = None) -> FeatureOptions:
    """Copy, validate, and fill defaults without retaining mutable caller dictionaries."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping of feature groups to option mappings, or None")
    available_groups = ", ".join(repr(group) for group in DEFAULTS)
    for group in options:
        if group not in DEFAULTS:
            raise ValueError(
                f"unknown options group {group!r}; available groups: {available_groups}"
            )

    resolved: dict[str, Mapping[str, float]] = {}
    for group, defaults in DEFAULTS.items():
        provided = options.get(group, {})
        if not isinstance(provided, Mapping):
            raise TypeError(f"options[{group!r}] must be a mapping")
        validators = _VALIDATORS[group]
        available_names = ", ".join(repr(name) for name in defaults)
        for name in provided:
            if name not in defaults:
                raise ValueError(
                    f"unknown option {name!r} for {group!r}; available options: {available_names}"
                )
        values = {
            name: validators[name](group, name, provided[name]) if name in provided else default
            for name, default in defaults.items()
        }
        resolved[group] = MappingProxyType(values)
    return MappingProxyType(resolved)
