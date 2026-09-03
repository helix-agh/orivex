"""Dependency planning for shared numerical intermediates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bflacco.registry import FeatureRegistry
from bflacco.specs import FeatureSpec, InputRequirement

_INTERMEDIATE_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


class PlanningError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IntermediateSpec:
    name: str
    dependencies: tuple[str, ...]
    requirements: frozenset[InputRequirement]

    def __post_init__(self) -> None:
        if _INTERMEDIATE_ID.fullmatch(self.name) is None:
            raise ValueError(f"invalid intermediate name: {self.name!r}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("intermediate dependencies must be unique")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    features: tuple[FeatureSpec, ...]
    intermediates: tuple[IntermediateSpec, ...]
    requirements: frozenset[InputRequirement]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.features)

    @property
    def intermediate_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.intermediates)


class Planner:
    def __init__(
        self,
        features: FeatureRegistry,
        intermediates: tuple[IntermediateSpec, ...] = (),
    ) -> None:
        self._features = features
        self._intermediates: dict[str, IntermediateSpec] = {}
        for intermediate in intermediates:
            if intermediate.name in self._intermediates:
                raise PlanningError(f"intermediate already registered: {intermediate.name}")
            self._intermediates[intermediate.name] = intermediate

    def plan(self, selectors: str | tuple[str, ...] | list[str]) -> ExecutionPlan:
        features = self._features.select(selectors)
        ordered: list[IntermediateSpec] = []
        permanent: set[str] = set()
        temporary: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise PlanningError(f"intermediate dependency cycle contains: {name}")
            try:
                intermediate = self._intermediates[name]
            except KeyError as error:
                raise PlanningError(f"unknown intermediate dependency: {name}") from error
            temporary.add(name)
            for dependency in intermediate.dependencies:
                visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(intermediate)

        for feature in features:
            for intermediate in feature.intermediates:
                visit(intermediate)

        requirements = set().union(*(feature.requirements for feature in features))
        requirements.update(
            requirement for intermediate in ordered for requirement in intermediate.requirements
        )
        return ExecutionPlan(features, tuple(ordered), frozenset(requirements))
