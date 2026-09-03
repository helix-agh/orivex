# bflacco

**Better FLACCO** is a correctness-first successor to `pflacco` for exploratory
landscape analysis (ELA).

The project is being built around four guarantees:

- individual features are selectable without calculating an entire historical group;
- shared numerical intermediates are calculated once by a dependency planner;
- CPU, memory, and additional objective-evaluation costs are explicit;
- every feature has a versioned mathematical specification and independent verification.

The project is in its initial specification and verification phase. It is not yet a
drop-in replacement for `pflacco`.

Currently implemented feature slices:

- `ela_distr`: individually selectable type-3 skewness and kurtosis;
- `ela_meta`: selected linear and corrected quadratic model intercept/fit statistics;
- `ic`: all five information-content outputs with a deterministic nearest-neighbour tour;
- `nbc`: all five nearest-better-clustering outputs with deterministic tie handling.

## Installation

The package is not published yet, so install it from a source checkout. Python 3.10 or newer,
NumPy, and SciPy are the only runtime requirements.

```bash
uv sync                        # runtime dependencies and the project itself
uv sync --extra dev            # add Ruff, ty, pytest, and pre-commit
uv sync --extra benchmark      # add the benchmark-only dependencies
```

With plain `pip`:

```bash
python -m pip install -e ".[dev]"
```

PyTorch is an optional tensor-native backend and is not imported or installed for NumPy users:

```bash
python -m pip install -e ".[torch]"
```

Every example below can also be run straight from the checkout without installing anything, by
putting the source tree on the import path: `PYTHONPATH=src python your_script.py`.

### Differentiable PyTorch features

The explicit `bflacco.torch` namespace keeps tensors on their existing device, preserves their
floating dtype, and returns scalar tensors connected to the autograd graph. The initial profile
contains the distribution skewness and kurtosis features:

```python
import torch

from bflacco.torch import TensorLandscapeSample, compute

x = torch.rand(200, 2, device="cuda", dtype=torch.float32, requires_grad=True)
y = torch.sum(x**2, dim=1)
bounds = torch.full((2,), 5.0, device=x.device, dtype=x.dtype)
sample = TensorLandscapeSample(x, y, lower=-bounds, upper=bounds)

result = compute(sample, "ela_distr.skewness")
feature = result.values["ela_distr.skewness"].value
feature.backward()
```

There is no implicit fallback to NumPy: requesting a known feature that has no Torch calculator
raises `UnsupportedFeatureError`. Use `bflacco.torch.list_features()` and
`bflacco.torch.list_capabilities()` to discover the implemented profile and its declared device,
dtype, and autograd support.

## Usage

### Compute features for a sample

Wrap paired decision and objective observations in a `LandscapeSample`, then ask for the features
you want. Selectors are exact feature names or `fnmatch` globs, given as a single string or a list.

```python
import numpy as np

from bflacco import LandscapeSample, compute

rng = np.random.default_rng(20260830)
lower, upper = np.full(2, -5.0), np.full(2, 5.0)

x = rng.uniform(lower, upper, size=(200, 2))
y = np.sum(x**2, axis=1) + 10.0 * np.sum(np.cos(2 * np.pi * x), axis=1)  # Rastrigin

sample = LandscapeSample(x, y, lower=lower, upper=upper)
result = compute(sample, ["ela_distr.*", "ic.h_max", "nbc.nn_nb.mean_ratio"])

for name, item in result.values.items():
    print(f"{name:24s} {item.value:.6f}  [{item.status.value}]")
```

```text
ela_distr.kurtosis       -0.211763  [ok]
ela_distr.skewness       0.139713  [ok]
ic.h_max                 0.841578  [ok]
nbc.nn_nb.mean_ratio     0.570166  [ok]
```

`LandscapeSample` validates its inputs on construction: `X` must be two-dimensional and `y`
one-dimensional with matching length, all values must be finite, every lower bound must be
strictly below its upper bound, and all observations must lie inside the box. The sample is
immutable and its arrays are read-only, so a sample can be reused across many `compute` calls.

### Selecting individual features avoids unrelated work

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

### Controlling parallel work

Feature computation defaults to one worker so repeated or externally parallel analyses do not
silently occupy every CPU. Supporting kernels can use a specific positive worker count, or all
available CPUs with `-1`:

```python
fast_single_landscape = compute(sample, "ic.*", workers=-1)
```

Using all CPUs reduces IC wall latency on sufficiently large samples, but increases total CPU
consumption and can be slower for small samples. Keep the default when parallelizing across many
landscapes.

The fingerprint identifies the exact numerical input, which makes it usable as a cache key or as
provenance stored next to an experiment's results.

### Maximization problems

Declare the objective sense instead of negating `y` by hand. Features that claim invariance under
sense reversal return identical values either way.

```python
from bflacco import ObjectiveSense

maximizing = LandscapeSample(x, -y, lower=lower, upper=upper, sense=ObjectiveSense.MAXIMIZE)
minimizing = LandscapeSample(x, y, lower=lower, upper=upper)

compute(maximizing, "nbc.*")  # same values as compute(minimizing, "nbc.*")
```

`sample.minimization_y` exposes the objective in minimization convention if you need it directly.

### Discovering what is available

`list_features()` returns the full specification of every registered feature, not just its name.

```python
from bflacco import list_features

for spec in list_features():
    print(
        f"{spec.name:32s} {spec.group:10s} {spec.cost.tier.value:14s} n>={spec.minimum_observations}"
    )
```

```text
ela_distr.kurtosis               ela_distr  sample_only    n>=4
ela_distr.skewness               ela_distr  sample_only    n>=3
ela_meta.lin_simple.adj_r2       ela_meta   sample_only    n>=3
ela_meta.lin_simple.intercept    ela_meta   sample_only    n>=2
ela_meta.lin_w_interact.adj_r2   ela_meta   sample_only    n>=3
ela_meta.quad_simple.adj_r2      ela_meta   sample_only    n>=3
ela_meta.quad_w_interact.adj_r2  ela_meta   sample_only    n>=3
ic.eps_max                       ic         sample_only    n>=3
ic.eps_ratio                     ic         sample_only    n>=3
ic.eps_s                         ic         sample_only    n>=3
ic.h_max                         ic         sample_only    n>=3
ic.m0                            ic         sample_only    n>=3
nbc.dist_ratio.coeff_var         nbc        sample_only    n>=2
nbc.nb_fitness.cor               nbc        sample_only    n>=2
nbc.nn_nb.cor                    nbc        sample_only    n>=2
nbc.nn_nb.mean_ratio             nbc        sample_only    n>=2
nbc.nn_nb.sd_ratio               nbc        sample_only    n>=2
```

Each `FeatureSpec` also carries `definition` (the versioned specification identifier), `summary`,
`kind`, `requirements`, `intermediates`, `cost` (tier plus CPU and memory complexity),
`deterministic`, `invariances`, `references`, `legacy_names`, and `notes`.

### Mathematically undefined outputs

A feature that is undefined for an otherwise valid sample returns a `FeatureValue` with status
`invalid` and an explanation rather than raising or silently producing `NaN`.

```python
from bflacco.result import FeatureStatus

flat = LandscapeSample(x, np.zeros(len(x)), lower=lower, upper=upper)
item = compute(flat, "ela_distr.skewness").values["ela_distr.skewness"]

item.status is FeatureStatus.INVALID  # True
item.value  # None
item.message  # 'skewness is undefined for constant objective values'
```

A selector that matches no registered feature is a caller error and raises instead:

```python
from bflacco.registry import UnknownFeatureSelection

compute(sample, "ela_meta.nonexistent")
# UnknownFeatureSelection: selector matched no features: ela_meta.nonexistent
```

## Development order

1. Specify feature semantics and catalogue known legacy defects.
2. Establish analytical, metamorphic, and R `flacco` differential tests.
3. Build the sample model, registry, planner, and result metadata.
4. Implement the zero-additional-evaluation core in NumPy/SciPy.
5. Benchmark before introducing native kernels.

See [the roadmap](docs/roadmap.md) and [verification strategy](docs/verification.md).

## Development checks

Install the development dependencies and Git hook once, then run the complete suite as needed:

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
```

The hook applies Ruff linting/formatting, runs ty over the library source, validates project and
data files, and checks common repository hygiene problems. Direct checks are available through
`uv run ruff check .`, `uv run ruff format --check .`, and `uv run ty check`.

The test suite covers analytical, metamorphic, and R `flacco` differential cases. `pyproject.toml`
already puts `src` on the import path, so no install step is required:

```bash
uv run pytest                                        # everything
uv run pytest tests/features/test_information_content.py
uv run pytest -k nearest_better                      # one feature family
uv run pytest tests/verification                     # metamorphic and differential checks
uv run pytest --cov=bflacco --cov-report=term-missing
```

The R differential fixtures are checked in. Regenerate them only when the recorded inputs or the
reference computation change, which requires R with the `flacco` and `jsonlite` packages:

```bash
uv run python tools/generate_fixture_inputs.py
Rscript tools/generate_r_fixtures.R
```

## Benchmarks

Benchmarks are diagnostics, not assertions in the test suite. Run them from the repository root.
The two standalone bflacco benchmarks need nothing beyond the runtime dependencies:

```bash
uv run python benchmarks/benchmark_distribution.py
uv run python benchmarks/benchmark_meta_model.py
```

The comparison and stability scripts need the benchmark extra, and the `compare_pflacco_*` scripts
additionally load a `pflacco` **source checkout** — by default a sibling directory `../pflacco`,
overridable with `--pflacco-root`:

```bash
uv sync --extra benchmark

uv run --extra benchmark python benchmarks/compare_pflacco_distribution.py \
    --sizes 100 1000 --dimension 10 --repeats 5 --calls 10
uv run --extra benchmark python benchmarks/compare_pflacco_families.py \
    --sizes 100 500 --dimensions 2 5 --threads 1 --json report.json
uv run --extra benchmark python benchmarks/analyze_feature_stability.py \
    --dimensions 2 5 --seeds 30 --y-mode standardized
```

Each report records dependency versions, selectors, sample shape, and median CPU and wall-clock
time, so store its output alongside any optimization claim; `--json` retains the full environment
and every individual measurement. See [benchmarks/README.md](benchmarks/README.md) for the scope
and caveats of each script, especially the parts of the `pflacco` comparison that are not yet a
same-instruction-kernel comparison.
