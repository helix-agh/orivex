"""Execution of planned feature definitions with shared intermediate caching."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bflacco.planner import IntermediateSpec, Planner
from bflacco.registry import FeatureRegistry
from bflacco.result import ComputationResult, ExecutionMetadata, FeatureStatus, FeatureValue
from bflacco.sample import LandscapeSample
from bflacco.specs import FeatureSpec, InputRequirement

IntermediateValue = object


class FeatureUnavailable(ValueError):
    """A feature is mathematically undefined for an otherwise valid sample."""


@dataclass(frozen=True, slots=True)
class ComputationContext:
    sample: LandscapeSample
    intermediates: Mapping[str, IntermediateValue]
    rng: np.random.Generator | None
    workers: int = 1

    def intermediate(self, name: str) -> IntermediateValue:
        try:
            return self.intermediates[name]
        except KeyError as error:
            raise RuntimeError(f"intermediate was not planned: {name}") from error


FeatureCalculator = Callable[[ComputationContext], float | int]
IntermediateCalculator = Callable[[ComputationContext], IntermediateValue]


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    spec: FeatureSpec
    calculate: FeatureCalculator


@dataclass(frozen=True, slots=True)
class IntermediateDefinition:
    spec: IntermediateSpec
    calculate: IntermediateCalculator


class Engine:
    def __init__(
        self,
        features: tuple[FeatureDefinition, ...],
        intermediates: tuple[IntermediateDefinition, ...],
    ) -> None:
        self._feature_definitions: dict[str, FeatureDefinition] = {}
        for definition in features:
            if definition.spec.name in self._feature_definitions:
                raise ValueError(f"feature definition already registered: {definition.spec.name}")
            self._feature_definitions[definition.spec.name] = definition

        self._intermediate_definitions: dict[str, IntermediateDefinition] = {}
        for definition in intermediates:
            if definition.spec.name in self._intermediate_definitions:
                raise ValueError(
                    f"intermediate definition already registered: {definition.spec.name}"
                )
            self._intermediate_definitions[definition.spec.name] = definition

        self.registry = FeatureRegistry(tuple(item.spec for item in features))
        self.planner = Planner(
            self.registry,
            tuple(item.spec for item in intermediates),
        )

    def compute(
        self,
        sample: LandscapeSample,
        features: str | tuple[str, ...] | list[str],
        *,
        rng: np.random.Generator | None = None,
        workers: int = 1,
    ) -> ComputationResult:
        if workers == 0 or workers < -1:
            raise ValueError("workers must be -1 or a positive integer")
        started = time.perf_counter()
        plan = self.planner.plan(features)
        if InputRequirement.RNG in plan.requirements and rng is None:
            raise ValueError("the requested feature plan requires an explicit numpy Generator")

        cache: dict[str, IntermediateValue] = {}
        for intermediate in plan.intermediates:
            context = ComputationContext(sample, MappingProxyType(cache), rng, workers)
            cache[intermediate.name] = self._intermediate_definitions[intermediate.name].calculate(
                context
            )

        context = ComputationContext(sample, MappingProxyType(cache), rng, workers)
        values: dict[str, FeatureValue] = {}
        for spec in plan.features:
            definition = self._feature_definitions[spec.name]
            try:
                value = definition.calculate(context)
                if not np.isfinite(value):
                    raise FeatureUnavailable("definition produced a non-finite value")
                values[spec.name] = FeatureValue(
                    value=value,
                    status=FeatureStatus.OK,
                    definition=spec.definition,
                )
            except FeatureUnavailable as error:
                values[spec.name] = FeatureValue(
                    value=None,
                    status=FeatureStatus.INVALID,
                    definition=spec.definition,
                    message=str(error),
                )

        metadata = ExecutionMetadata(
            sample_fingerprint=sample.fingerprint,
            requested_features=plan.feature_names,
            computed_intermediates=plan.intermediate_names,
            runtime_seconds=time.perf_counter() - started,
            additional_objective_evaluations=0,
            workers=workers,
        )
        return ComputationResult(values, metadata)
