# Benchmarks

Benchmarks are diagnostics, not timing assertions in the unit-test suite. Run them from the
repository root against the source tree:

```bash
PYTHONPATH=src python benchmarks/benchmark_distribution.py
PYTHONPATH=src python benchmarks/benchmark_meta_model.py
```

Each benchmark reports dependency versions, selectors, requested intermediates, sample shape,
and median wall-clock time over repeated calls. Store benchmark output alongside any optimization
claim so the input and environment remain auditable.

The meta-model benchmark includes individual linear and quadratic selectors alongside the entire
implemented `ela_meta.*` group. This makes the cost avoided by selective execution directly
visible.

Native code is considered only after a profile shows that a corrected NumPy/SciPy implementation
remains a material end-to-end bottleneck.

## Comparing with pflacco

Run the current user-facing distribution comparison with:

```bash
PYTHONPATH=src python benchmarks/compare_pflacco_distribution.py
```

Useful options include `--sizes`, `--dimension`, `--repeats`, `--calls`, and `--json`. The script
verifies the two common numerical outputs before timing and reports both process CPU seconds and
wall-clock seconds. It also distinguishes prepared-input computation from end-to-end input
construction.

Increase `--calls` and `--repeats` for publication-quality runs. The defaults favor a reasonably
stable local comparison without making the script slow to use during development.

The scope is not an identical group comparison yet: pflacco necessarily calculates KDE peak count,
while bflacco currently implements only individually selectable skewness and kurtosis. The report
records this difference so the resulting speedup is not misrepresented as a same-instruction-kernel
comparison.
