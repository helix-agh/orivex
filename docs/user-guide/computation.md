# Selecting and computing features

Use an exact feature name, a `fnmatch` glob, or a list of selectors. For example,
`"ela_distr.skewness"` requests one output, `"ela_distr.*"` requests the implemented
distribution family, and `"*"` requests every feature available to the NumPy backend.
Overlapping selectors are deduplicated. Each selector must match at least one feature.

The snippets below continue the [quick start](../getting-started/quickstart.md).

## Selecting individual features avoids unrelated work

The planner computes only the intermediates the requested features actually depend on, and each
shared intermediate is computed once per call. `ComputationResult.metadata` records what happened.

```python
compute(sample, "ela_distr.skewness").metadata.computed_intermediates
# ('y.centered', 'y.sum2', 'y.sum3')

compute(sample, "ela_distr.*").metadata.computed_intermediates
# ('y.centered', 'y.sum2', 'y.sum4', 'y.sum3')
```

```python
metadata = result.metadata
metadata.requested_features  # resolved feature names, in execution order
metadata.computed_intermediates  # shared intermediates actually evaluated
metadata.sample_fingerprint  # SHA-256 over X, y, bounds, and objective sense
metadata.additional_objective_evaluations  # 0 for every currently implemented feature
metadata.runtime_seconds  # wall-clock time of this call
metadata.workers  # explicit worker budget used by supporting kernels
```

## Controlling parallel work

Feature computation defaults to one worker so repeated or externally parallel analyses do not
silently occupy every CPU. Supporting kernels can use a specific positive worker count, or all
available CPUs with `-1`:

```python
fast_single_landscape = compute(sample, "ic.*", workers=-1)
```

Using all CPUs reduces IC wall latency on sufficiently large samples, but increases total CPU
consumption and can be slower for small samples. Keep the default when parallelizing across many
landscapes.

The sample fingerprint identifies the raw numerical input for provenance. Use the preprocessing
fingerprint and feature definitions when identifying cached results.

See [objective normalization](normalization.md) for preprocessing and
[fitness-distance features](fitness-distance.md) for group-specific options.
