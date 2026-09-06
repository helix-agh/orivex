# Samples and objective sense

A sample pairs `x` with shape `(n, d)` and `y` with shape `(n,)`. Bounds have shape `(d,)`.
The NumPy backend stores floating-point arrays as `float64`.

`LandscapeSample` validates its inputs on construction: `X` must be two-dimensional and `y`
one-dimensional with matching length, all values must be finite, every lower bound must be
strictly below its upper bound, and all observations must lie inside the box. The sample is
immutable and its arrays are read-only, so a sample can be reused across many `compute` calls.
The arrays are copied from the caller's inputs, so later changes to the original handles never
affect the sample. Because NumPy arrays that own their storage can have the read-only flag
re-enabled (directly or through an alias), each `compute` call first invokes
`sample.validate_unchanged()`, which recomputes an integrity digest and raises `RuntimeError` if
any exposed array was mutated in place — so a stale `fingerprint` can never be used as a
provenance or cache key. The Torch `TensorLandscapeSample` offers the same guarantee via tensor
version counters.

## Maximization problems

Declare the objective sense instead of negating `y` by hand. Features that claim invariance under
sense reversal return identical values either way.

```python
from orivex import ObjectiveSense

maximizing = LandscapeSample(x, -y, lower=lower, upper=upper, sense=ObjectiveSense.MAXIMIZE)
minimizing = LandscapeSample(x, y, lower=lower, upper=upper)

compute(maximizing, "nbc.*")  # same values as compute(minimizing, "nbc.*")
```

`sample.minimization_y` exposes the objective in minimization convention if you need it directly.

The examples above use `x`, `y`, `lower`, and `upper` from the
[quick start](../getting-started/quickstart.md). See the [sample API](../api/sample.md)
for properties and constructor signatures.
