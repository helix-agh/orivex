# Exploratory landscape analysis with orivex

<div class="orivex-hero" markdown>

<img class="orivex-logo-light" src="assets/orivex-lockup-light.png" alt="orivex" width="440">
<img class="orivex-logo-dark" src="assets/orivex-lockup-dark.png" alt="orivex" width="440">

**Compute the landscape features you need, with explicit costs and reproducible definitions.**

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Explore the features](features/index.md){ .md-button }

</div>

orivex computes exploratory landscape analysis (ELA) features from sampled decision vectors
and their objective values. It is a successor to
[pflacco](https://github.com/Reiyan/pflacco), with a NumPy/SciPy core and an optional
differentiable PyTorch backend.

!!! info "Pre-release"
    Install from a source checkout. orivex implements selected feature families and is not yet
    a drop-in replacement for pflacco. See the [feature overview](features/index.md) for coverage.

## Why orivex

- **Selective computation:** request individual outputs; shared intermediates run once per call.
- **Explicit costs:** inspect declared CPU, memory, and additional objective-evaluation costs.
- **Versioned definitions:** trace feature semantics, invariance claims, and source literature.
- **Verification:** analytical, metamorphic, and R flacco differential tests check the implementation.
- **Differentiable features:** use the supported Torch profile with device and autograd awareness.

## Start with a sample

```python
import numpy as np

from orivex import LandscapeSample, compute

rng = np.random.default_rng(42)
x = rng.uniform(-5.0, 5.0, size=(200, 2))
y = np.sum(x**2, axis=1)
sample = LandscapeSample(x, y, lower=[-5.0, -5.0], upper=[5.0, 5.0])

result = compute(sample, "ela_distr.*")
for name, item in result.values.items():
    print(name, item.value, item.status.value)
```

Objective values are min-max normalized by default. Choose preprocessing explicitly when
reproducing reference results; see [objective normalization](user-guide/normalization.md).

## Find your next step

| Task | Documentation |
| --- | --- |
| Install and compute your first features | [Getting started](getting-started/installation.md) |
| Choose outputs and control execution | [Feature selection](user-guide/computation.md) |
| Understand definitions and costs | [Feature catalogue](features/catalogue.md) |
| Differentiate features in a model | [PyTorch and autograd](user-guide/torch.md) |
| Interpret invalid values and record provenance | [Results](user-guide/results.md) |
| Look up signatures and data models | [API reference](api/index.md) |
| Contribute and reproduce benchmarks | [Development](development/contributing.md) |

orivex is distributed under the [MIT License](https://github.com/helix-agh/orivex/blob/main/LICENSE.txt).
