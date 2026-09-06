"""Validation and immutable snapshots of options keyed by feature group."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

FeatureOptions: TypeAlias = Mapping[str, Mapping[str, float]]


def resolve_options(options: FeatureOptions | None = None) -> FeatureOptions:
    """Copy, validate, and fill defaults without retaining mutable caller dictionaries."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping of feature groups to option mappings, or None")
    for group in options:
        if group != "fitness_distance":
            raise ValueError(
                f"unknown options group {group!r}; available group: 'fitness_distance'"
            )

    fitness_distance = options.get("fitness_distance", {})
    if not isinstance(fitness_distance, Mapping):
        raise TypeError("options['fitness_distance'] must be a mapping")
    for name in fitness_distance:
        if name != "proportion_of_best":
            raise ValueError(
                f"unknown option {name!r} for 'fitness_distance'; "
                "available option: 'proportion_of_best'"
            )
    proportion = fitness_distance.get("proportion_of_best", 0.1)
    if isinstance(proportion, bool) or not isinstance(proportion, Real) or not 0 < proportion <= 1:
        raise ValueError(
            "options['fitness_distance']['proportion_of_best'] must be finite in (0, 1]"
        )
    return MappingProxyType(
        {"fitness_distance": MappingProxyType({"proportion_of_best": float(proportion)})}
    )
