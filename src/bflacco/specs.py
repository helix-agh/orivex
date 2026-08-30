"""Typed, immutable feature-definition metadata.

This module describes feature semantics. It intentionally contains no feature execution logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

_FEATURE_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_DEFINITION_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class InputRequirement(str, Enum):
    """Inputs that a feature or its intermediates require."""

    X = "x"
    Y = "y"
    BOUNDS = "bounds"
    OBJECTIVE = "objective"
    RNG = "rng"


class CostTier(str, Enum):
    """Coarse cost class used for discovery and safe defaults."""

    SAMPLE_ONLY = "sample_only"
    ADDITIONAL_EVALUATIONS = "additional_evaluations"
    OPTIMIZATION = "optimization"


class MetricKind(str, Enum):
    """Whether an output describes a landscape or only its sampling design."""

    LANDSCAPE = "landscape"
    DESIGN = "design"


class Transformation(str, Enum):
    """Transformations considered by metamorphic verification."""

    ROW_PERMUTATION = "row_permutation"
    VARIABLE_PERMUTATION = "variable_permutation"
    X_TRANSLATION = "x_translation"
    X_POSITIVE_SCALING = "x_positive_scaling"
    X_ORTHOGONAL_ROTATION = "x_orthogonal_rotation"
    Y_TRANSLATION = "y_translation"
    Y_POSITIVE_SCALING = "y_positive_scaling"
    OBJECTIVE_SENSE_REVERSAL = "objective_sense_reversal"


class InvarianceBehavior(str, Enum):
    """Declared behavior under a transformation."""

    INVARIANT = "invariant"
    EQUIVARIANT = "equivariant"
    NON_INVARIANT = "non_invariant"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class InvarianceClaim:
    """A precise, testable transformation claim."""

    transformation: Transformation
    behavior: InvarianceBehavior
    conditions: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Reference:
    """Literature or software reference supporting a feature definition."""

    citation: str
    doi: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        if not self.citation.strip():
            raise ValueError("reference citation must not be empty")
        if self.doi is None and self.url is None:
            raise ValueError("reference must provide a DOI or URL")


@dataclass(frozen=True, slots=True)
class CostModel:
    """Auditable symbolic cost model for a single feature request."""

    tier: CostTier
    cpu: str
    memory: str
    additional_objective_evaluations: str = "0"

    def __post_init__(self) -> None:
        if not self.cpu.strip() or not self.memory.strip():
            raise ValueError("CPU and memory cost descriptions must not be empty")
        if not self.additional_objective_evaluations.strip():
            raise ValueError("objective-evaluation cost must not be empty")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Complete metadata contract for one individually selectable output."""

    name: str
    group: str
    kind: MetricKind
    definition: str
    summary: str
    requirements: frozenset[InputRequirement]
    intermediates: tuple[str, ...]
    cost: CostModel
    deterministic: bool
    invariances: tuple[InvarianceClaim, ...] = ()
    references: tuple[Reference, ...] = ()
    legacy_names: tuple[str, ...] = ()
    minimum_observations: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if _FEATURE_ID.fullmatch(self.name) is None:
            raise ValueError(f"invalid feature name: {self.name!r}")
        if not self.group or self.name.split(".", 1)[0] != self.group:
            raise ValueError("feature group must equal the first component of the feature name")
        if _DEFINITION_ID.fullmatch(self.definition) is None:
            raise ValueError(f"invalid definition identifier: {self.definition!r}")
        if not self.summary.strip():
            raise ValueError("feature summary must not be empty")
        if self.kind is MetricKind.LANDSCAPE and InputRequirement.Y not in self.requirements:
            raise ValueError("landscape features must explicitly require objective observations y")
        if self.kind is MetricKind.DESIGN and InputRequirement.X not in self.requirements:
            raise ValueError("design descriptors must explicitly require decision observations X")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if len(set(self.intermediates)) != len(self.intermediates):
            raise ValueError("intermediate identifiers must be unique")
        transformations = [claim.transformation for claim in self.invariances]
        if len(set(transformations)) != len(transformations):
            raise ValueError("a feature may declare at most one claim per transformation")
        if len(set(self.legacy_names)) != len(self.legacy_names):
            raise ValueError("legacy feature names must be unique")
