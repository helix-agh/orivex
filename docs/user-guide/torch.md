# PyTorch and autograd

Install the optional backend with `uv sync --extra torch` or
`python -m pip install -e ".[torch]"` from your checkout.

The explicit `orivex.torch` namespace keeps tensors on their existing device, preserves their
floating dtype, and returns scalar tensors connected to the autograd graph. The differentiable
profile contains distribution skewness and kurtosis, the ELA meta-model adjusted R-squared
and linear-intercept features, all six fitness-distance statistics, the five exact
nearest-better clustering features, and their five `nbc_soft.*` surrogates:

```python
import torch

from orivex.torch import TensorLandscapeSample, compute

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(200, 2, device=device, dtype=torch.float32, requires_grad=True)
y = torch.sum(x**2, dim=1)
bounds = torch.full((2,), 5.0, device=x.device, dtype=x.dtype)
sample = TensorLandscapeSample(x, y, lower=-bounds, upper=bounds)

result = compute(sample, "ela_distr.skewness")
feature = result.values["ela_distr.skewness"].value
assert feature is not None
feature.backward()
```

There is no implicit fallback to NumPy: requesting a known feature that has no Torch calculator
raises `UnsupportedFeatureError`. Use `orivex.torch.list_features()` and
`orivex.torch.list_capabilities()` to discover the implemented profile and its declared device,
dtype, and autograd support.

## Input and execution requirements

`x` and `y` must be tensors with matching `float32` or `float64` dtype and device.
Tensor bounds must match them; array-like bounds are converted to the input dtype and device.
Samples clone inputs while preserving autograd history and detect in-place mutation through
tensor version counters.

Sample construction computes a fingerprint using a detached CPU snapshot; feature computation
operates on the sample device. Account for construction separately when benchmarking GPU work.

The Torch `compute` function accepts `y_normalization` and `options`; it has no `workers`
or `rng` arguments. Check [backend capabilities](../api/torch.md) for device support before
requesting features. Information-content features currently have no Torch calculators.

## Nearest-better clustering and its surrogate

The exact `nbc.*` group selects its graph under `no_grad` and then recomputes the selected edge
lengths from `x`, so the four distance features carry a gradient with respect to the design.
With a fixed design they are locally **constant in `y`**: the fitness *ordering* picks the edges,
not the spacing, so the gradient with respect to `y` is zero almost everywhere and those features
cannot drive gradient-based training on their own. `nbc.nb_fitness.cor` is the exception; it
differentiates through `y` at a locally fixed graph, because the indegrees enter as constants
while the objective values do not.

The `nbc_soft.*` group replaces the discrete "first strictly better neighbour" search with a
smooth first-success distribution over distance-ordered candidates, which restores a usable
gradient with respect to `y` for all five statistics. It is a **surrogate**, not an NBC feature:
its value depends on the gate temperature, and only the exact group should be reported.

```python
soft = compute(sample, "nbc_soft.*", options={"nbc_soft": {"temperature": 0.003}})
```

| Option | Default | Meaning |
| --- | --- | --- |
| `temperature` | `0.003` | Gate width in gate-coordinate units. Smaller values track the exact features more closely and concentrate the gradient on fewer observations. |
| `rank_temperature` | `0.5` | Rank blur, measured in median consecutive objective gaps. |

The gates rank the objectives rather than min-max scaling them, and this is fixed rather than
configurable. Plain normalization keeps the original spacing, so a heavy upper tail crowds most of
the sample into an interval no single temperature resolves. Measured on BBOB d=2 with n=500 across
six functions, min-max gates diverged from the exact features 2.4 to 7.8 times further than the
soft ranks, at every temperature tested. Both are invariant to a positive affine change of `y`,
so nothing else was traded away.

The default temperature is the measured balance point for `n = 250d`. On the same BBOB setup the
divergence of `nbc_soft.nn_nb.sd_ratio` from its exact counterpart is about one standard deviation
of the exact feature's own design-to-design sampling noise, while the other four surrogates sit
well below it. A tenth of that temperature pushes the divergence to roughly a third of a standard
deviation, but concentrates 98% of the gradient norm onto 5% of the observations, against 89% at
the default. Divergence *decreases* with sample size, so a temperature calibrated in `d=2`
is conservative for larger designs rather than optimistic.

Two limits are worth stating before training against these features:

* equal objective values gate at `1/2` at every temperature, so plateaus do not reproduce the
  exact equal-fitness fallback;
* the gate coordinates are spaced by `1 / (n - 1)`, so a fixed temperature spans more candidates
  as the sample grows.

Compare a surrogate value against a target computed with the **same** temperature and sample
size, and report the exact `nbc.*` group on fresh designs.

## Preprocessing and gradients

Min-max normalization is the default and introduces piecewise differentiability at extrema.
Use `list_capabilities(y_normalization=None)` to inspect the raw pipeline.
Both nearest-better groups are dense in Torch: they build `O(n^2)` pairwise work and storage,
while the NumPy backend bounds its distance-block memory.
Read [normalization](normalization.md) and
[fitness-distance selection](fitness-distance.md) for details on ties and gradient conventions.
