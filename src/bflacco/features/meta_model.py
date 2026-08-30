"""Selected ELA meta-model features implemented with NumPy least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine import (
    ComputationContext,
    FeatureDefinition,
    FeatureUnavailable,
    IntermediateDefinition,
)
from ..planner import IntermediateSpec
from ..specs import (
    CostModel,
    CostTier,
    FeatureSpec,
    InputRequirement,
    InvarianceBehavior,
    InvarianceClaim,
    MetricKind,
    Reference,
    Transformation,
)

REFERENCE = Reference(
    citation="Mersmann et al. (2011), Exploratory Landscape Analysis",
    doi="10.1145/2001576.2001690",
)
R_REFERENCE = Reference(
    citation="flacco 1.8, ELA meta-model reference implementation",
    url="https://github.com/kerschke/flacco/blob/master/R/feature_ela_meta_model.R",
)


@dataclass(frozen=True, slots=True)
class PredictorMatrix:
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class RegressionFit:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    observations: int
    predictors: int
    rank: int
    constant_objective: bool
    total_sum_squares: float


def linear_predictors(context: ComputationContext) -> PredictorMatrix:
    return PredictorMatrix(context.sample.x)


def quadratic_predictors(context: ComputationContext) -> PredictorMatrix:
    x = context.sample.x
    return PredictorMatrix(np.concatenate((x, x * x), axis=1))


def _pairwise_interactions(values: np.ndarray) -> np.ndarray:
    left, right = np.triu_indices(values.shape[1], k=1)
    if left.size == 0:
        return values
    return np.concatenate((values, values[:, left] * values[:, right]), axis=1)


def linear_interaction_predictors(context: ComputationContext) -> PredictorMatrix:
    return PredictorMatrix(_pairwise_interactions(context.sample.x))


def quadratic_interaction_predictors(context: ComputationContext) -> PredictorMatrix:
    x = context.sample.x
    linear_and_interactions = _pairwise_interactions(x)
    return PredictorMatrix(np.concatenate((linear_and_interactions, x * x), axis=1))


def _predictors(context: ComputationContext, name: str) -> PredictorMatrix:
    value = context.intermediate(name)
    if not isinstance(value, PredictorMatrix):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def _fit(context: ComputationContext, predictor_name: str) -> RegressionFit:
    predictors = _predictors(context, predictor_name).values
    y = context.sample.minimization_y
    design = np.empty((y.size, predictors.shape[1] + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:] = predictors
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    centered_y = y - np.mean(y)
    coefficients.flags.writeable = False
    fitted.flags.writeable = False
    residuals.flags.writeable = False
    return RegressionFit(
        coefficients=coefficients,
        fitted=fitted,
        residuals=residuals,
        observations=y.size,
        predictors=predictors.shape[1],
        rank=int(rank),
        constant_objective=bool(np.all(y == y[0])),
        total_sum_squares=float(centered_y @ centered_y),
    )


def fit_linear_simple(context: ComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.lin_simple")


def fit_linear_interactions(context: ComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.lin_w_interact")


def fit_quadratic_simple(context: ComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.quad_simple")


def fit_quadratic_interactions(context: ComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.quad_w_interact")


def _regression(context: ComputationContext, name: str) -> RegressionFit:
    value = context.intermediate(name)
    if not isinstance(value, RegressionFit):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def adjusted_r2(context: ComputationContext, fit_name: str) -> float:
    fit = _regression(context, fit_name)
    columns = fit.predictors + 1
    if fit.rank < columns:
        raise FeatureUnavailable("adjusted R-squared requires a full-rank model matrix")
    residual_degrees = fit.observations - fit.predictors - 1
    if residual_degrees <= 0:
        raise FeatureUnavailable(
            "adjusted R-squared requires more observations than model coefficients"
        )
    if fit.constant_objective:
        raise FeatureUnavailable("adjusted R-squared is undefined for constant objective values")

    residual_sum = float(fit.residuals @ fit.residuals)
    if fit.total_sum_squares == 0.0:
        raise FeatureUnavailable("adjusted R-squared has zero total variation")
    correction = residual_degrees / (fit.observations - 1)
    return float(1.0 - (residual_sum / fit.total_sum_squares) / correction)


def linear_intercept(context: ComputationContext) -> float:
    fit = _regression(context, "ela_meta.fit.lin_simple")
    if fit.rank < fit.predictors + 1:
        raise FeatureUnavailable("linear intercept requires a full-rank model matrix")
    return float(fit.coefficients[0])


PREDICTOR_INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.lin_simple",
            (),
            frozenset({InputRequirement.X}),
        ),
        linear_predictors,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.lin_w_interact",
            (),
            frozenset({InputRequirement.X}),
        ),
        linear_interaction_predictors,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.quad_simple",
            (),
            frozenset({InputRequirement.X}),
        ),
        quadratic_predictors,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.quad_w_interact",
            (),
            frozenset({InputRequirement.X}),
        ),
        quadratic_interaction_predictors,
    ),
)

FIT_INTERMEDIATES = (
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.lin_simple",
            ("ela_meta.predictors.lin_simple",),
            frozenset({InputRequirement.Y}),
        ),
        fit_linear_simple,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.lin_w_interact",
            ("ela_meta.predictors.lin_w_interact",),
            frozenset({InputRequirement.Y}),
        ),
        fit_linear_interactions,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.quad_simple",
            ("ela_meta.predictors.quad_simple",),
            frozenset({InputRequirement.Y}),
        ),
        fit_quadratic_simple,
    ),
    IntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.quad_w_interact",
            ("ela_meta.predictors.quad_w_interact",),
            frozenset({InputRequirement.Y}),
        ),
        fit_quadratic_interactions,
    ),
)

COMMON_REQUIREMENTS = frozenset({InputRequirement.X, InputRequirement.Y})
COMMON_REFERENCES = (REFERENCE, R_REFERENCE)
FIT_INVARIANCES = (
    InvarianceClaim(Transformation.ROW_PERMUTATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(Transformation.VARIABLE_PERMUTATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(Transformation.Y_TRANSLATION, InvarianceBehavior.INVARIANT),
    InvarianceClaim(
        Transformation.Y_POSITIVE_SCALING,
        InvarianceBehavior.INVARIANT,
        conditions="finite positive scale and non-constant objective values",
    ),
    InvarianceClaim(
        Transformation.OBJECTIVE_SENSE_REVERSAL,
        InvarianceBehavior.INVARIANT,
        conditions="negate y and reverse the declared objective sense together",
    ),
)


def _fit_feature(
    name: str,
    fit_name: str,
    summary: str,
    *,
    cpu: str,
    definition: str = "ols-adjusted-r2-v1",
    notes: tuple[str, ...] = (),
) -> FeatureDefinition:
    return FeatureDefinition(
        spec=FeatureSpec(
            name=name,
            group="ela_meta",
            kind=MetricKind.LANDSCAPE,
            definition=definition,
            summary=summary,
            requirements=COMMON_REQUIREMENTS,
            intermediates=(fit_name,),
            cost=CostModel(CostTier.SAMPLE_ONLY, cpu=cpu, memory="O(n p)"),
            deterministic=True,
            invariances=FIT_INVARIANCES,
            references=COMMON_REFERENCES,
            legacy_names=(name,),
            minimum_observations=3,
            notes=notes,
        ),
        calculate=lambda context: adjusted_r2(context, fit_name),
    )


LIN_SIMPLE_ADJ_R2 = _fit_feature(
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.fit.lin_simple",
    "Adjusted R-squared of an ordinary linear model with an intercept.",
    cpu="O(n d^2 + d^3)",
)

LIN_INTERACT_ADJ_R2 = _fit_feature(
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.fit.lin_w_interact",
    "Adjusted R-squared of a linear model with pairwise variable interactions.",
    cpu="O(n d^4 + d^6)",
)

QUAD_SIMPLE_ADJ_R2 = _fit_feature(
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.fit.quad_simple",
    "Adjusted R-squared of a model containing linear and squared variable terms.",
    cpu="O(n d^2 + d^3)",
)

QUAD_INTERACT_ADJ_R2 = _fit_feature(
    "ela_meta.quad_w_interact.adj_r2",
    "ela_meta.fit.quad_w_interact",
    "Adjusted R-squared of a complete degree-2 polynomial model.",
    cpu="O(n d^4 + d^6)",
    definition="complete-quadratic-adjusted-r2-v1",
    notes=(
        "Unlike flacco and pflacco, this corrected definition does not interact already-squared "
        "columns and therefore introduces no cubic or quartic terms.",
    ),
)

LIN_SIMPLE_INTERCEPT = FeatureDefinition(
    spec=FeatureSpec(
        name="ela_meta.lin_simple.intercept",
        group="ela_meta",
        kind=MetricKind.LANDSCAPE,
        definition="ols-intercept-v1",
        summary="Intercept of an ordinary linear model fitted by least squares.",
        requirements=COMMON_REQUIREMENTS,
        intermediates=("ela_meta.fit.lin_simple",),
        cost=CostModel(CostTier.SAMPLE_ONLY, cpu="O(n d^2 + d^3)", memory="O(n d)"),
        deterministic=True,
        invariances=(
            InvarianceClaim(Transformation.ROW_PERMUTATION, InvarianceBehavior.INVARIANT),
            InvarianceClaim(Transformation.VARIABLE_PERMUTATION, InvarianceBehavior.INVARIANT),
            InvarianceClaim(
                Transformation.Y_TRANSLATION,
                InvarianceBehavior.EQUIVARIANT,
                notes="The fitted intercept changes by the same additive constant.",
            ),
            InvarianceClaim(
                Transformation.Y_POSITIVE_SCALING,
                InvarianceBehavior.EQUIVARIANT,
                notes="The fitted intercept changes by the same positive factor.",
            ),
            InvarianceClaim(
                Transformation.OBJECTIVE_SENSE_REVERSAL,
                InvarianceBehavior.INVARIANT,
                conditions="negate y and reverse the declared objective sense together",
            ),
        ),
        references=COMMON_REFERENCES,
        legacy_names=("ela_meta.lin_simple.intercept",),
        minimum_observations=2,
    ),
    calculate=linear_intercept,
)

FEATURES = (
    LIN_SIMPLE_ADJ_R2,
    LIN_SIMPLE_INTERCEPT,
    LIN_INTERACT_ADJ_R2,
    QUAD_SIMPLE_ADJ_R2,
    QUAD_INTERACT_ADJ_R2,
)
INTERMEDIATES = PREDICTOR_INTERMEDIATES + FIT_INTERMEDIATES
