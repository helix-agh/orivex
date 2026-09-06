# BBOB correctness benchmark

The benchmark compares all 23 implemented NumPy features with pflacco using identical,
raw float64 samples from all 24 noiseless IOH BBOB functions. Defaults are dimensions
2/5/10, instance 1, five sampling seeds, and `100*d` observations: 360 datasets in total.
Objectives and feature values are not normalized. Timing is a separate experiment.

From the repository root, with a pflacco source checkout at `../pflacco`:

```bash
uv run --extra benchmark python benchmarks/bbob_correctness.py --workers 4
```

Use `--pflacco-root` for another checkout. Independent processes handle datasets, each with
one numerical thread. `--functions`, `--dimensions`, `--instances`, `--seeds`, and
`--sample-multiplier` configure the experiment; `--output` chooses a new result directory.

The report retains absolute/relative errors, nonfinite and failure statuses, warnings,
exact samples, versions, source hashes, and worst-case identifiers. Per-feature error
distributions aggregate functions and sampling seeds, with dimensions shown separately.
Heatmaps identify functions with the largest discrepancies. No timing data is collected.

```bash
uv run --extra benchmark python benchmarks/bbob_correctness.py \
    --plot-only --output benchmark-results/bbob-correctness
```

Numerical agreement uses `abs(a-b) <= 1e-10 + 1e-8*abs(b)` by default. Relative error remains
undefined at reference zero. Undefined outputs never count as passes. The deliberately
different quadratic-interaction definition has an independent polynomial regression check;
its raw pflacco discrepancy is still reported. Other discrepancies require investigation.

Exit status 1 means the saved report contains cases requiring review. Keep analytical,
invariance, edge-case, and Torch tests: reference agreement alone does not prove correctness.

See the [complete benchmark guide](https://github.com/helix-agh/orivex/blob/main/benchmarks/README.md)
for the comparison contracts, CLI examples, output schema, and reproduction instructions.
