# Specifications and capabilities

`FeatureSpec` records feature semantics, definition identifiers, input requirements, shared
intermediates, symbolic costs, invariance claims, and references. The
[catalogue](../features/catalogue.md) renders registered instances of this model.

Backend-specific execution properties belong to `FeatureCapability`: supported devices,
dtypes, and the declared autograd behavior. They are separate from mathematical definitions.

::: orivex.FeatureSpec

::: orivex.CostModel

::: orivex.CostTier

::: orivex.InputRequirement

::: orivex.MetricKind

::: orivex.InvarianceClaim

::: orivex.InvarianceBehavior

::: orivex.Transformation

::: orivex.Reference

::: orivex.FeatureCapability

::: orivex.AutogradSupport
