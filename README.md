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

Install `--extra benchmark` to run the reproducible CPU/wall-time comparisons described in
[benchmarks/README.md](benchmarks/README.md).
