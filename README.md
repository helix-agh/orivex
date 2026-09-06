<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/helix-agh/orivex/main/assets/orivex-lockup-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/helix-agh/orivex/main/assets/orivex-lockup-light.png">
    <img alt="orivex" src="https://raw.githubusercontent.com/helix-agh/orivex/main/assets/orivex-lockup-light.png" width="440">
  </picture>
</p>

<p align="center">
  <strong>A selective and fast engine for exploratory landscape analysis.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <img alt="Status: pre-release" src="https://img.shields.io/badge/status-pre--release-orange.svg">
  <img alt="Backends: NumPy and PyTorch" src="https://img.shields.io/badge/backends-NumPy%20%7C%20PyTorch-informational.svg">
</p>

<p align="center">
  <a href="https://helix-agh.github.io/orivex/">Documentation</a> ·
  <a href="https://helix-agh.github.io/orivex/getting-started/quickstart/">Quick start</a> ·
  <a href="https://helix-agh.github.io/orivex/api/">API reference</a> ·
  <a href="https://helix-agh.github.io/orivex/development/contributing/">Contributing</a>
</p>

**orivex** is a successor to [`pflacco`](https://github.com/Reiyan/pflacco)
for exploratory landscape analysis (ELA). Compute individually selected features from sampled
points and objective values, with shared intermediates, explicit costs, and versioned definitions.
A NumPy/SciPy core is complemented by an optional differentiable PyTorch backend.

## Installation

Python 3.10 or newer is required. Install from PyPI:

```bash
pip install orivex
```

Add the optional differentiable PyTorch backend with `pip install "orivex[torch]"`.

To work from a checkout instead:

```bash
git clone https://github.com/helix-agh/orivex.git
cd orivex
uv sync
```

With pip, run `python -m pip install -e .` (add `".[torch]"` for the PyTorch backend).

## Quick start

```python
import numpy as np

from orivex import LandscapeSample, compute

rng = np.random.default_rng(42)
x = rng.uniform(-5.0, 5.0, size=(200, 2))
y = np.sum(x**2, axis=1)
sample = LandscapeSample(x, y, lower=[-5.0, -5.0], upper=[5.0, 5.0])

result = compute(sample, ["ela_distr.skewness", "nbc.nn_nb.mean_ratio"])
for name, item in result.values.items():
    print(name, item.value, item.status.value)
```

Objective values are min-max normalized by default. Pass `y_normalization=None` to use raw
canonical objectives. Undefined features return an `invalid` status with an explanation.

## Documentation

Read the **[documentation](https://helix-agh.github.io/orivex/)** for installation, feature
selection, normalization, fitness-distance conventions, PyTorch support, and API reference.
The [feature overview](https://helix-agh.github.io/orivex/features/), [contributing guide](https://helix-agh.github.io/orivex/development/contributing/),
and [benchmark guide](https://helix-agh.github.io/orivex/development/benchmarks/) are also available.

Preview the documentation locally:

```bash
uv sync --extra docs
uv run --no-sync mkdocs serve
```

See [documentation development](https://helix-agh.github.io/orivex/development/documentation/) for strict builds and deployment.

## License

orivex is licensed under the [MIT License](https://github.com/helix-agh/orivex/blob/main/LICENSE.txt).
