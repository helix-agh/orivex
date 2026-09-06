# Quick start

Wrap paired decision and objective observations in a `LandscapeSample`, then ask for the features
you want. Selectors are exact feature names or `fnmatch` globs, given as a single string or a list.

```python
import numpy as np

from orivex import LandscapeSample, compute

rng = np.random.default_rng(20260830)
lower, upper = np.full(2, -5.0), np.full(2, 5.0)

x = rng.uniform(lower, upper, size=(200, 2))
y = 10.0 * x.shape[1] + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x), axis=1)  # Rastrigin

sample = LandscapeSample(x, y, lower=lower, upper=upper)
result = compute(sample, ["ela_distr.*", "ic.h_max", "nbc.nn_nb.mean_ratio"])

for name, item in result.values.items():
    print(f"{name:24s} {item.value:.6f}  [{item.status.value}]")
```

The output contains a value and status for each selected feature. Check the status before
using a value: mathematically undefined features return `None` with an explanation.

Continue with [samples and objective sense](../user-guide/samples.md),
[feature selection](../user-guide/computation.md), and
[results and metadata](../user-guide/results.md).
