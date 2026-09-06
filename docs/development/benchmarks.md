# Benchmarks

Benchmarks are diagnostics, not assertions in the test suite. Run them from the repository root.
The two standalone orivex benchmarks need nothing beyond the runtime dependencies:

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
and every individual measurement. See [benchmarks/README.md](https://github.com/helix-agh/orivex/blob/main/benchmarks/README.md) for the scope
and caveats of each script, especially the parts of the `pflacco` comparison that are not yet a
same-instruction-kernel comparison.
