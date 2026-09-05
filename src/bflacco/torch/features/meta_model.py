"""Tensor-native ELA meta-model features."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bflacco.capabilities import FeatureCapability
from bflacco.engine import FeatureUnavailable
from bflacco.features.meta_model import (
    LIN_INTERACT_ADJ_R2,
    LIN_SIMPLE_ADJ_R2,
    LIN_SIMPLE_INTERCEPT,
    QUAD_INTERACT_ADJ_R2,
    QUAD_SIMPLE_ADJ_R2,
)
from bflacco.planner import IntermediateSpec
from bflacco.specs import InputRequirement
from bflacco.torch.engine import (
    TensorComputationContext,
    TensorFeatureDefinition,
    TensorIntermediateDefinition,
)


@dataclass(frozen=True, slots=True)
class PredictorMatrix:
    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class RegressionFit:
    coefficients: torch.Tensor
    residuals: torch.Tensor
    observations: int
    predictors: int
    rank: int
    constant_objective: bool
    total_sum_squares: torch.Tensor


def linear_predictors(context: TensorComputationContext) -> PredictorMatrix:
    return PredictorMatrix(context.sample.x)


def quadratic_predictors(context: TensorComputationContext) -> PredictorMatrix:
    x = context.sample.x
    return PredictorMatrix(torch.cat((x, x.square()), dim=1))


def _pairwise_interactions(values: torch.Tensor) -> torch.Tensor:
    dimension = values.shape[1]
    if dimension < 2:
        return values
    indices = torch.triu_indices(dimension, dimension, offset=1, device=values.device)
    interactions = values[:, indices[0]] * values[:, indices[1]]
    return torch.cat((values, interactions), dim=1)


def linear_interaction_predictors(context: TensorComputationContext) -> PredictorMatrix:
    return PredictorMatrix(_pairwise_interactions(context.sample.x))


def quadratic_interaction_predictors(context: TensorComputationContext) -> PredictorMatrix:
    x = context.sample.x
    return PredictorMatrix(torch.cat((_pairwise_interactions(x), x.square()), dim=1))


def _predictors(context: TensorComputationContext, name: str) -> PredictorMatrix:
    value = context.intermediate(name)
    if not isinstance(value, PredictorMatrix):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def _fit(context: TensorComputationContext, predictor_name: str) -> RegressionFit:
    predictors = _predictors(context, predictor_name).values
    y = context.y
    observations = y.numel()
    columns = predictors.shape[1] + 1

    # Fit on standardized predictor columns for conditioning, then map the coefficients back to the
    # specified raw coordinates. The means and scales are detached: standardizing by an
    # intercept-preserving affine transform leaves the column space -- and therefore the value and
    # its gradient -- unchanged, so holding them constant keeps a large coordinate offset from being
    # mistaken for rank deficiency without perturbing autograd.
    means = predictors.detach().mean(dim=0, keepdim=True)
    scales = predictors.detach().std(dim=0, unbiased=False, keepdim=True)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    standardized = (predictors - means) / scales
    design = torch.cat((torch.ones_like(y).unsqueeze(1), standardized), dim=1)

    # Rank determines whether any feature consuming this fit is defined. It is intentionally
    # outside the autograd graph: rank is discrete, while the fit is smooth wherever rank is
    # constant. Avoid calling lstsq for an underdetermined/rank-deficient CUDA design because
    # CUDA's supported ``gels`` driver assumes full column rank.
    rank = int(torch.linalg.matrix_rank(design.detach()).item())
    if rank < columns:
        coefficients = torch.zeros(columns, dtype=y.dtype, device=y.device)
        residuals = y
    else:
        standardized_coefficients = torch.linalg.lstsq(design, y, driver="gels").solution
        residuals = y - design @ standardized_coefficients
        slopes = standardized_coefficients[1:] / scales.squeeze(0)
        intercept = standardized_coefficients[0] - torch.dot(slopes, means.squeeze(0))
        coefficients = torch.cat((intercept.unsqueeze(0), slopes))

    centered_y = y - torch.mean(y)
    return RegressionFit(
        coefficients=coefficients,
        residuals=residuals,
        observations=observations,
        predictors=predictors.shape[1],
        rank=rank,
        constant_objective=bool(torch.all(y.detach() == y.detach()[0]).item()),
        total_sum_squares=torch.dot(centered_y, centered_y),
    )


def fit_linear_simple(context: TensorComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.lin_simple")


def fit_linear_interactions(context: TensorComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.lin_w_interact")


def fit_quadratic_simple(context: TensorComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.quad_simple")


def fit_quadratic_interactions(context: TensorComputationContext) -> RegressionFit:
    return _fit(context, "ela_meta.predictors.quad_w_interact")


def _regression(context: TensorComputationContext, name: str) -> RegressionFit:
    value = context.intermediate(name)
    if not isinstance(value, RegressionFit):
        raise TypeError(f"{name} has an invalid runtime type")
    return value


def adjusted_r2(context: TensorComputationContext, fit_name: str) -> torch.Tensor:
    fit = _regression(context, fit_name)
    columns = fit.predictors + 1
    if fit.rank < columns:
        raise FeatureUnavailable("adjusted R-squared requires a full-rank model matrix")
    residual_degrees = fit.observations - columns
    if residual_degrees <= 0:
        raise FeatureUnavailable(
            "adjusted R-squared requires more observations than model coefficients"
        )
    if fit.constant_objective:
        raise FeatureUnavailable("adjusted R-squared is undefined for constant objective values")
    if bool(fit.total_sum_squares.detach().eq(0).item()):
        raise FeatureUnavailable("adjusted R-squared has zero total variation")

    residual_sum = torch.dot(fit.residuals, fit.residuals)
    correction = residual_degrees / (fit.observations - 1)
    return 1.0 - (residual_sum / fit.total_sum_squares) / correction


def linear_intercept(context: TensorComputationContext) -> torch.Tensor:
    fit = _regression(context, "ela_meta.fit.lin_simple")
    if fit.rank < fit.predictors + 1:
        raise FeatureUnavailable("linear intercept requires a full-rank model matrix")
    return fit.coefficients[0]


PREDICTOR_INTERMEDIATES = (
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.lin_simple",
            (),
            frozenset({InputRequirement.X}),
        ),
        linear_predictors,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.lin_w_interact",
            (),
            frozenset({InputRequirement.X}),
        ),
        linear_interaction_predictors,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.quad_simple",
            (),
            frozenset({InputRequirement.X}),
        ),
        quadratic_predictors,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.predictors.quad_w_interact",
            (),
            frozenset({InputRequirement.X}),
        ),
        quadratic_interaction_predictors,
    ),
)

FIT_INTERMEDIATES = (
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.lin_simple",
            ("ela_meta.predictors.lin_simple",),
            frozenset({InputRequirement.Y}),
        ),
        fit_linear_simple,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.lin_w_interact",
            ("ela_meta.predictors.lin_w_interact",),
            frozenset({InputRequirement.Y}),
        ),
        fit_linear_interactions,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.quad_simple",
            ("ela_meta.predictors.quad_simple",),
            frozenset({InputRequirement.Y}),
        ),
        fit_quadratic_simple,
    ),
    TensorIntermediateDefinition(
        IntermediateSpec(
            "ela_meta.fit.quad_w_interact",
            ("ela_meta.predictors.quad_w_interact",),
            frozenset({InputRequirement.Y}),
        ),
        fit_quadratic_interactions,
    ),
)

FEATURES = (
    TensorFeatureDefinition(
        LIN_SIMPLE_ADJ_R2.spec,
        lambda context: adjusted_r2(context, "ela_meta.fit.lin_simple"),
    ),
    TensorFeatureDefinition(LIN_SIMPLE_INTERCEPT.spec, linear_intercept),
    TensorFeatureDefinition(
        LIN_INTERACT_ADJ_R2.spec,
        lambda context: adjusted_r2(context, "ela_meta.fit.lin_w_interact"),
    ),
    TensorFeatureDefinition(
        QUAD_SIMPLE_ADJ_R2.spec,
        lambda context: adjusted_r2(context, "ela_meta.fit.quad_simple"),
    ),
    TensorFeatureDefinition(
        QUAD_INTERACT_ADJ_R2.spec,
        lambda context: adjusted_r2(context, "ela_meta.fit.quad_w_interact"),
    ),
)

_ADJUSTED_R2_NOTES = (
    "Differentiable for full-rank model matrices with positive residual degrees of freedom and "
    "non-constant objective values.",
    "PyTorch least squares restricts this implementation to CPU and CUDA.",
)
_INTERCEPT_NOTES = (
    "Differentiable for full-rank linear model matrices.",
    "PyTorch least squares restricts this implementation to CPU and CUDA.",
)

CAPABILITIES = tuple(
    FeatureCapability(
        feature_name=definition.spec.name,
        backend="torch",
        autograd="smooth",
        devices=("cpu", "cuda"),
        dtypes=("float32", "float64"),
        notes=(
            _INTERCEPT_NOTES
            if definition.spec.name == LIN_SIMPLE_INTERCEPT.spec.name
            else _ADJUSTED_R2_NOTES
        ),
    )
    for definition in FEATURES
)

INTERMEDIATES = PREDICTOR_INTERMEDIATES + FIT_INTERMEDIATES
