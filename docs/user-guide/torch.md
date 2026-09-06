# PyTorch and autograd

Install the optional backend with `uv sync --extra torch` or
`python -m pip install -e ".[torch]"` from your checkout.

The explicit `orivex.torch` namespace keeps tensors on their existing device, preserves their
floating dtype, and returns scalar tensors connected to the autograd graph. The differentiable
profile contains distribution skewness and kurtosis, the ELA meta-model adjusted R-squared
and linear-intercept features, and all six fitness-distance statistics:

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
requesting features. IC and NBC currently have no Torch calculators.

## Preprocessing and gradients

Min-max normalization is the default and introduces piecewise differentiability at extrema.
Use `list_capabilities(y_normalization=None)` to inspect the raw pipeline.
Read [normalization](normalization.md) and
[fitness-distance selection](fitness-distance.md) for details on ties and gradient conventions.
