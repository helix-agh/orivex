# Computation

## Input and output contract

`compute(sample, features, ...)` returns a `ComputationResult` with one entry per resolved feature
and separate execution metadata. Features use canonical minimization objectives, min-max
normalized by default.

| Argument | Meaning |
| --- | --- |
| `sample` | A validated `LandscapeSample` |
| `features` | An exact name, `fnmatch` glob, or list/tuple of selectors; every selector must match |
| `rng` | Optional NumPy generator for features that declare an RNG requirement |
| `workers` | `1` by default; a positive integer or `-1` for supporting parallel kernels |
| `y_normalization` | `"minmax"` (default) or `None` to disable normalization |
| `options` | Optional mapping of feature groups to settings; see [fitness-distance options](../user-guide/fitness-distance.md) |

An unknown selector raises `UnknownFeatureSelection`. Invalid inputs or options raise errors.
A mathematically undefined feature on a valid sample instead produces an `invalid` value with
an explanation. See [results](../user-guide/results.md).

::: orivex.compute

`list_features()` returns the full `FeatureSpec` tuple for the registered NumPy profile,
ordered by feature name. For a readable listing, see the [catalogue](../features/catalogue.md).

::: orivex.list_features

::: orivex.registry.UnknownFeatureSelection

::: orivex.registry.RegistryError

::: orivex.normalization.YNormalization

::: orivex.options.FeatureOptions
