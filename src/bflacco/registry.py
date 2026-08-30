"""Feature discovery and exact/glob selection."""

from __future__ import annotations

from fnmatch import fnmatchcase

from .specs import FeatureSpec


class RegistryError(ValueError):
    """Base class for feature-registry errors."""


class DuplicateFeatureError(RegistryError):
    pass


class UnknownFeatureSelection(RegistryError):
    pass


class FeatureRegistry:
    """Mutable during construction, deterministic during discovery and selection."""

    def __init__(self, specs: tuple[FeatureSpec, ...] = ()) -> None:
        self._specs: dict[str, FeatureSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise DuplicateFeatureError(f"feature already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise UnknownFeatureSelection(f"unknown feature: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def select(self, selectors: str | tuple[str, ...] | list[str]) -> tuple[FeatureSpec, ...]:
        selector_values = (selectors,) if isinstance(selectors, str) else tuple(selectors)
        if not selector_values:
            raise UnknownFeatureSelection("at least one feature selector is required")

        selected: dict[str, FeatureSpec] = {}
        available = self.names()
        for selector in selector_values:
            matches = [name for name in available if fnmatchcase(name, selector)]
            if not matches:
                raise UnknownFeatureSelection(f"selector matched no features: {selector}")
            for name in matches:
                selected.setdefault(name, self._specs[name])
        return tuple(selected.values())
