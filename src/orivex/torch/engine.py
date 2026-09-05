"""Execution engine for tensor-native feature definitions."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from orivex.engine import FeatureUnavailable
from orivex.normalization import YNormalization
from orivex.planner import IntermediateSpec, Planner
from orivex.registry import FeatureRegistry
from orivex.result import ComputationResult, ExecutionMetadata, FeatureStatus, FeatureValue
from orivex.specs import FeatureSpec
from orivex.torch.normalization import normalize_objectives
from orivex.torch.sample import TensorLandscapeSample

IntermediateValue = object

# Numerical failures that are expected for otherwise valid samples. These are isolated per feature
# as INVALID results; programming errors are not listed and still propagate. Torch surfaces
# singular designs through ``torch.linalg.LinAlgError`` (a ``RuntimeError`` subclass); only that
# specific type is treated as numerical so unrelated ``RuntimeError`` bugs stay distinguishable.
EXPECTED_NUMERICAL_ERRORS: tuple[type[Exception], ...] = (
    FeatureUnavailable,
    ArithmeticError,
    torch.linalg.LinAlgError,
)


@dataclass(frozen=True, slots=True)
class _IntermediateFailure:
    """Marks an intermediate that raised an expected numerical failure, propagated to dependents."""

    message: str


@dataclass(frozen=True, slots=True)
class TensorComputationContext:
    sample: TensorLandscapeSample
    intermediates: Mapping[str, IntermediateValue]
    objective_y: torch.Tensor | None = None

    @property
    def y(self) -> torch.Tensor:
        return self.sample.minimization_y if self.objective_y is None else self.objective_y

    def intermediate(self, name: str) -> IntermediateValue:
        try:
            value = self.intermediates[name]
        except KeyError as error:
            raise RuntimeError(f"intermediate was not planned: {name}") from error
        if isinstance(value, _IntermediateFailure):
            raise FeatureUnavailable(value.message)
        return value


TensorFeatureCalculator = Callable[[TensorComputationContext], torch.Tensor]
TensorIntermediateCalculator = Callable[[TensorComputationContext], IntermediateValue]


@dataclass(frozen=True, slots=True)
class TensorFeatureDefinition:
    spec: FeatureSpec
    calculate: TensorFeatureCalculator


@dataclass(frozen=True, slots=True)
class TensorIntermediateDefinition:
    spec: IntermediateSpec
    calculate: TensorIntermediateCalculator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class TensorEngine:
    def __init__(
        self,
        features: tuple[TensorFeatureDefinition, ...],
        intermediates: tuple[TensorIntermediateDefinition, ...],
    ) -> None:
        self._features = {definition.spec.name: definition for definition in features}
        if len(self._features) != len(features):
            raise ValueError("tensor feature definitions must have unique names")
        self._intermediates = {definition.spec.name: definition for definition in intermediates}
        if len(self._intermediates) != len(intermediates):
            raise ValueError("tensor intermediate definitions must have unique names")
        self.registry = FeatureRegistry(tuple(definition.spec for definition in features))
        self.planner = Planner(
            self.registry,
            tuple(definition.spec for definition in intermediates),
        )

    def compute(
        self,
        sample: TensorLandscapeSample,
        features: str | tuple[str, ...] | list[str],
        *,
        y_normalization: YNormalization = "minmax",
    ) -> ComputationResult[torch.Tensor]:
        sample.validate_unchanged()
        _synchronize(sample.x.device)
        started = time.perf_counter()
        plan = self.planner.plan(features)
        objective_y, constant = normalize_objectives(sample.minimization_y, y_normalization)

        cache: dict[str, IntermediateValue] = {}
        for intermediate in plan.intermediates:
            context = TensorComputationContext(sample, MappingProxyType(cache), objective_y)
            try:
                cache[intermediate.name] = self._intermediates[intermediate.name].calculate(context)
            except EXPECTED_NUMERICAL_ERRORS as error:
                cache[intermediate.name] = _IntermediateFailure(str(error))

        context = TensorComputationContext(sample, MappingProxyType(cache), objective_y)
        values: dict[str, FeatureValue[torch.Tensor]] = {}
        for spec in plan.features:
            try:
                value = self._features[spec.name].calculate(context)
                if value.ndim != 0:
                    raise TypeError(f"{spec.name} must produce a scalar tensor")
                if not bool(torch.isfinite(value).detach().item()):
                    raise FeatureUnavailable("definition produced a non-finite value")
                values[spec.name] = FeatureValue(value, FeatureStatus.OK, spec.definition)
            except EXPECTED_NUMERICAL_ERRORS as error:
                values[spec.name] = FeatureValue(
                    None,
                    FeatureStatus.INVALID,
                    spec.definition,
                    str(error),
                )

        _synchronize(sample.x.device)
        metadata = ExecutionMetadata(
            sample_fingerprint=sample.fingerprint,
            requested_features=plan.feature_names,
            computed_intermediates=plan.intermediate_names,
            runtime_seconds=time.perf_counter() - started,
            additional_objective_evaluations=0,
            backend="torch",
            device=sample.device_type,
            device_index=sample.device_index,
            dtype=sample.dtype_name,
            y_normalization=y_normalization,
            constant_objective=constant,
        )
        return ComputationResult(values, metadata)
