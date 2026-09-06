# Fitness-distance features

Both backends support all six pflacco fitness-distance outputs:
`fitness_mean`, `fitness_std`, `distance_mean`, `distance_std`, `fd_cov`, and `fd_correlation`.
They keep the best `round(n * proportion_of_best)` observations. The default fraction is **0.1**;
Python's ties-to-even rounding applies. Maximization selects the largest original objectives.
At least two selected observations are required.

```python
compute(sample, "fitness_distance.*")  # all six, best 10%, min-max normalized
compute(
    sample,
    "fitness_distance.*",
    options={"fitness_distance": {"proportion_of_best": 0.25}},
)
compute(
    sample,
    "fitness_distance.*",
    options={"fitness_distance": {"proportion_of_best": 1.0}},  # full sample
    y_normalization=None,  # raw objective values
)
```

`options` is a dictionary keyed by feature group and works identically in NumPy and Torch.
No options class is needed. Omit it, pass `None`, or use an empty dictionary to keep defaults.
Currently the only supported setting is `fitness_distance.proportion_of_best`, which must be
finite and in `(0, 1]`. Unknown groups, unknown option names, and invalid values raise errors.
Options affect only their named group, including in requests that mix groups.

Objective normalization uses the **full sample**; selection ranks the raw canonical objectives
to avoid artificial ties from rounding during normalization. Distances are always **Euclidean
in raw decision coordinates, measured from the best selected observation**. This reference is
estimated from the sample; no known global optimum is required or accepted. Selection and
reference ties choose the first original row, so distances for tied samples can depend on row
order. This explicit tie rule may differ from pflacco's default unstable sort.

Standard deviations use `ddof=1`; covariance divides by the selected count `k`. Following
pflacco, `fd_correlation` divides this population covariance by the two sample deviations,
so it equals **`(k-1)/k` times Pearson correlation**. Zero fitness or distance variance makes
only correlation invalid; the other statistics remain available. The full-sample case works
in orivex, including reference selection (the inspected pflacco version fails in that path).

Selection is shared across requested outputs. Fitness-only requests need no distances;
distance outputs share one reference-distance vector, with `O(n + k*d)` work and memory
instead of a pairwise distance matrix. Torch preserves dtype, device, and autograd, with
piecewise gradients through selected objectives and reference coordinates. Selection changes
and ties are nonsmooth; zero deviations and coincident distances use zero gradient conventions.

Effective options, including defaults, are copied into an immutable `result.metadata.options`
mapping. For example, read the fraction as
`result.metadata.options["fitness_distance"]["proportion_of_best"]`. Caller dictionaries are
never modified or retained. Include these settings with feature definitions and the
preprocessing fingerprint in result cache keys. Runtime stays in metadata.

These examples use `compute` and `sample` from the
[quick start](../getting-started/quickstart.md). The [feature catalogue](../features/catalogue.md)
records the definitions, references, and conditions for each output.
