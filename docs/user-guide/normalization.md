# Objective normalization

## Motivation

Our default follows the sampled-objective min-max transformation in **Prager and Trautmann
(2023), _Nullifying the Inherent Bias of Non-invariant Exploratory Landscape Analysis Features_**.
The paper examines how absolute objective offsets and scales can bias ELA-based algorithm
selection on BBOB and evaluates normalization before feature computation.
[Sections 5–6](https://doi.org/10.1007/978-3-031-30229-9_27) use each problem's sampled extrema;
the true global minimum and maximum are not required.

The paper also mentions standardization and robust scaling as alternatives. Its selected
transformation and empirical evaluation use min-max; it does not establish that other
normalizations are invalid.

## Behavior

`compute` defaults to `y_normalization="minmax"` in both backends. It first converts the
objective to minimization convention, then applies `(y - min(y)) / (max(y) - min(y))` once per
request. `LandscapeSample.y` and `sample.minimization_y` retain the original observations and
their raw canonical values. Choose preprocessing explicitly when reproducing older results:

```python
compute(sample, "ela_meta.*")  # min-max, the default
compute(sample, "ela_meta.*", y_normalization=None)  # raw canonical objectives
```

The supported choices are `"minmax"` and Python `None`. Restricting the API to the paper's
selected transformation and an explicit opt-out keeps the preprocessing contract focused.
The previous `"none"` spelling and `"zscore"` mode are rejected; update raw computations
to `y_normalization=None`.

For nonconstant canonical minimization objectives $y_i$, orivex applies:

$$
\widehat{y}_i = \frac{y_i - \min_j y_j}{\max_j y_j - \min_j y_j}.
$$

Min-max preprocessing removes positive objective-scale and shift dependence on the
same finite nonconstant observations, up to floating-point accuracy. It does not remove
variation between sampling designs or normalize decision coordinates. Multiplying the objective
by a negative number reverses ordering; declare the corresponding objective sense explicitly
as described in [samples and objective sense](samples.md).

Constant objectives map to zero in min-max mode;
features requiring variation still return `invalid`. No epsilon is added to the denominator.
With `None`, canonical objectives are passed through, including their original offset and scale.

This default changes intercepts and IC thresholds relative to earlier releases. `FeatureSpec`
continues to describe the underlying formula on its input objectives; preprocessing and the
formula definition together identify the computed quantity. Normalization precedes any
family-specific duplicate aggregation and fitness-distance selection. The extrema come from the
full sample for each landscape independently, not from a collection of landscapes or a training
dataset. R/pflacco raw comparisons explicitly use `None`.

Metadata includes `y_normalization`, `y_normalization_definition`, `constant_objective`, and
`preprocessing_fingerprint`. Use the preprocessing fingerprint together with feature definitions
and execution settings for result caching. The raw sample fingerprint alone does not identify
the normalization mode.
Raw preprocessing retains the definition identifier `objective-none-v1`; min-max uses
`objective-minmax-v1`. Disabling normalization records Python `None` in
`metadata.y_normalization` (JSON `null` when serialized).

Torch preprocessing preserves dtype, device, and gradients. Min-max is piecewise differentiable
at changes in the extrema; `orivex.torch.list_capabilities()` conservatively reports the default
pipeline as `piecewise`. Pass `y_normalization=None` to capability discovery to
inspect the raw pipeline. This is objective preprocessing, separate from scaling a feature vector
for a downstream machine-learning model.

The snippets use `compute` and `sample` from the [quick start](../getting-started/quickstart.md).

## Reference

Raphael Patrick Prager and Heike Trautmann (2023).
*Nullifying the Inherent Bias of Non-invariant Exploratory Landscape Analysis Features*.
In *Applications of Evolutionary Computation (EvoApplications 2023)*, pp. 411–425.
DOI: [10.1007/978-3-031-30229-9_27](https://doi.org/10.1007/978-3-031-30229-9_27).
[Download BibTeX](../references.bib).

This citation supports the objective-normalization choice. Our constant-sample convention,
overflow handling, provenance metadata, and Torch autograd behavior are implementation choices.
