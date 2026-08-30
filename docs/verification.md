# Verification strategy

Agreement with one existing implementation is compatibility evidence, not proof of
correctness. Each feature is verified through independent layers.

## 1. Mathematical specification

Record the formula, preprocessing, estimator conventions, parameters, degenerate cases,
invariances, and source literature before implementation.

## 2. Analytical cases

Use landscapes with known structure, including constant, linear, separable quadratic,
shifted quadratic, rotated ellipsoid, plateau, duplicated-point, and hand-built information
sequences.

## 3. Metamorphic properties

Test transformations only under explicitly recorded preconditions. Candidate transformations
include paired row permutations, decision-variable permutations, translations, positive
objective scaling, orthogonal rotations, and minimize/maximize reversal. A property may be
invariant, equivariant, deliberately non-invariant, or unknown.

## 4. Differential fixtures

Compare against both R `flacco` and `pflacco` 1.2.2 on fixed samples. R agreement is not
automatically canonical: disagreements are resolved using the specification and analytical
tests. Fixtures must be cross-language, versioned, and accompanied by dependency versions and
all feature parameters.

Floating-point outputs use feature-specific tolerances. Platform-specific baselines and a
blanket requirement for cross-platform bit identity are not accepted.

## 5. Stochastic validation

Require reproducibility for a fixed explicit generator. Separately compare estimator
distributions across repeated designs, sampling methods, and budgets.

## 6. Downstream validation

Measure feature usefulness in leakage-resistant algorithm-selection and performance-prediction
experiments. Downstream usefulness does not substitute for numerical correctness.
