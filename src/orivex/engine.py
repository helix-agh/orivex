"""Execution of planned feature definitions with shared intermediate caching."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from orivex.normalization import YNormalization, normalize_objectives
from orivex.options import FeatureOptions, resolve_options
from orivex.planner import IntermediateSpec, Planner
from orivex.registry import FeatureRegistry
from orivex.result import ComputationResult, ExecutionMetadata, FeatureStatus, FeatureValue
from orivex.sample import LandscapeSample
from orivex.specs import FeatureSpec, InputRequirement

IntermediateValue = object


class FeatureUnavailable(ValueError):
    """A feature is mathematically undefined for an otherwise valid sample."""


# Numerical failures that are expected for otherwise valid samples (overflow, underflow, division
# by an underflowed quantity, singular designs). These are isolated per feature as INVALID results.
# Programming errors -- TypeError, KeyError, IndexError, AttributeError, and the like -- are not
# listed and therefore still propagate so that bugs remain distinguishable from numerical failures.
EXPECTED_NUMERICAL_ERRORS: tuple[type[Exception], ...] = (
    FeatureUnavailable,
    ArithmeticError,
    np.linalg.LinAlgError,
)


@dataclass(frozen=True, slots=True)
class _IntermediateFailure:
    """Marks an intermediate that raised an expected numerical failure.

    Stored in place of a value so that features depending on it (directly or transitively) surface
    the same failure as an INVALID status instead of terminating the whole request.
    """

    message: str


@dataclass(frozen=True, slots=True)
class ComputationContext:
    sample: LandscapeSample
    intermediates: Mapping[str, IntermediateValue]
    rng: np.random.Generator | None
    workers: int = 1
    objective_y: np.ndarray | None = None
    options: FeatureOptions = field(default_factory=resolve_options)

    @property
    def y(self) -> np.ndarray:
        """Canonical objectives after the computation's declared preprocessing."""
        return self.sample.minimization_y if self.objective_y is None else self.objective_y

    def intermediate(self, name: str) -> IntermediateValue:
        try:
            value = self.intermediates[name]
        except KeyError as error:
            raise RuntimeError(f"intermediate was not planned: {name}") from error
        if isinstance(value, _IntermediateFailure):
            raise FeatureUnavailable(value.message)
        return value


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
        y_normalization: YNormalization = "minmax",
        options: FeatureOptions | None = None,
    ) -> ComputationResult:
        options = resolve_options(options)
        if workers == 0 or workers < -1:
            raise ValueError("workers must be -1 or a positive integer")
        sample.validate_unchanged()
        started = time.perf_counter()
        plan = self.planner.plan(features)
        if InputRequirement.RNG in plan.requirements and rng is None:
            raise ValueError("the requested feature plan requires an explicit numpy Generator")

        objective_y, constant = normalize_objectives(sample.minimization_y, y_normalization)

        cache: dict[str, IntermediateValue] = {}
        for intermediate in plan.intermediates:
            context = ComputationContext(
                sample,
                MappingProxyType(cache),
                rng,
                workers,
                objective_y,
                options,
            )
            try:
                cache[intermediate.name] = self._intermediate_definitions[
                    intermediate.name
                ].calculate(context)
            except EXPECTED_NUMERICAL_ERRORS as error:
                cache[intermediate.name] = _IntermediateFailure(str(error))

        context = ComputationContext(
            sample,
            MappingProxyType(cache),
            rng,
            workers,
            objective_y,
            options,
        )
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
            except EXPECTED_NUMERICAL_ERRORS as error:
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
            y_normalization=y_normalization,
            constant_objective=constant,
            options=options,
        )
        return ComputationResult(values, metadata)
