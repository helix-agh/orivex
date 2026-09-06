# Feature overview

| Group        | Feature slice                                                          |
| ------------ | --------------------------------------------------------------------- |
| `ela_distr`  | individually selectable type-3 skewness and kurtosis                  |
| `ela_meta`   | selected linear and corrected quadratic model intercept/fit statistics |
| `fitness_distance` | all six fitness/distance means, sample deviations, covariance, and correlation |
| `ic`         | all five information-content outputs, deterministic nearest-neighbour tour |
| `nbc`        | all five nearest-better-clustering outputs, deterministic tie handling |

These are the currently implemented slices, not every historical pflacco output.
The [Torch guide](../user-guide/torch.md) describes the smaller differentiable profile.

`list_features()` returns the full specification of every registered feature, not just its name.

```python
from orivex import list_features

for spec in list_features():
    print(
        f"{spec.name:32s} {spec.group:10s} {spec.cost.tier.value:14s} n>={spec.minimum_observations}"
    )
```

The complete current list is generated in the [feature catalogue](catalogue.md).

Each `FeatureSpec` also carries `definition` (the versioned specification identifier), `summary`,
`kind`, `requirements`, `intermediates`, `cost` (tier plus CPU and memory complexity),
`deterministic`, `invariances`, `references`, `legacy_names`, and `notes`.

Minimum observation counts in the catalogue are necessary conditions, not guarantees of a
valid result. Model rank, objective variation, and family-specific rules still apply.
For fitness-distance outputs, at least two **selected** observations are required.

See [fitness-distance conventions](../user-guide/fitness-distance.md) and
[normalization](../user-guide/normalization.md) when reproducing reference results.
