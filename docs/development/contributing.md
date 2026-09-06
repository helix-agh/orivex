# Contributing

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
uv run pytest --cov=orivex --cov-report=term-missing
```

The stability-experiment regression tests additionally require the benchmark extra:
`uv run --extra dev --extra benchmark pytest`. Without them, that test module is skipped.

The R differential fixtures are checked in. Regenerate them only when the recorded inputs or the
reference computation change, which requires R with the `flacco` and `jsonlite` packages:

```bash
uv run python tools/generate_fixture_inputs.py
Rscript tools/generate_r_fixtures.R
```

## Adding or changing a feature

1. Specify feature semantics and catalogue known legacy defects.
2. Establish analytical, metamorphic, and R `flacco` differential tests.
3. Build the sample model, registry, planner, and result metadata.
4. Implement the zero-additional-evaluation core in NumPy/SciPy.
5. Benchmark before introducing native kernels.

Keep the implementation, versioned `FeatureSpec`, and verification cases consistent.
The [feature catalogue](../features/catalogue.md) is rebuilt from registered specifications.
Write longer explanations in the user guide and describe public APIs with Google-style
docstrings, consistent with lonkit.

Historical planning and research notes are retained in
[archived_docs](https://github.com/helix-agh/orivex/tree/main/archived_docs).

See [documentation development](documentation.md) for the docs workflow.
