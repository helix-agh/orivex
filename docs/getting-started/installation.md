# Installation

The package is not published yet, so install it from a source checkout:

```bash
git clone https://github.com/helix-agh/orivex.git
cd orivex
```

Python 3.10 or newer, NumPy, and SciPy are the only runtime requirements.

```bash
uv sync                        # runtime dependencies and the project itself
uv sync --extra dev            # add Ruff, ty, pytest, and pre-commit
uv sync --extra benchmark      # add the benchmark-only dependencies
```

With plain `pip`:

```bash
python -m pip install -e .
```

PyTorch is an optional tensor-native backend and is not imported or installed for NumPy users:

```bash
python -m pip install -e ".[torch]"
```

Examples can also be run straight from the checkout without installing anything, by
putting the source tree on the import path: `PYTHONPATH=src python your_script.py`.

For documentation tools, use `uv sync --extra docs`. See
[building the documentation](../development/documentation.md) for preview and build commands.

Continue with the [quick start](quickstart.md).
