"""Generate deterministic cross-language input matrices for differential fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "r_flacco"
INPUT_ROOT = FIXTURE_ROOT / "inputs"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_case(name: str, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    path = INPUT_ROOT / f"{name}.csv"
    columns = [f"x{index}" for index in range(x.shape[1])]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*columns, "y"])
        for row, objective in zip(x, y, strict=True):
            formatted_row = [format(float(value), ".17g") for value in row]
            writer.writerow([*formatted_row, format(float(objective), ".17g")])
    return {
        "name": name,
        "file": str(path.relative_to(FIXTURE_ROOT)),
        "sha256": sha256_file(path),
        "x_columns": columns,
        "lower": [-5.0] * x.shape[1],
        "upper": [5.0] * x.shape[1],
        "blocks": [3] * x.shape[1],
        "seed": 20260830,
    }


def main() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(20260830))
    x = rng.uniform(-5.0, 5.0, size=(128, 2))
    cases = [
        write_case("sphere_d2", x, np.sum(x**2, axis=1)),
        write_case("linear_d2", x, 2.0 * x[:, 0] - 3.0 * x[:, 1] + 4.0),
    ]
    manifest = {
        "schema_version": 1,
        "generator": "numpy.random.PCG64",
        "generator_seed": 20260830,
        "feature_sets": ["ela_distr", "ela_meta", "disp", "nbc", "ic", "pca"],
        "feature_parameters": {
            "ic": {
                "sorting": "nn",
                "nn_neighborhood": 20,
                "nn_start": "lexicographically_smallest_x",
                "epsilon": "0 + logspace(-5, 15, 1000)",
                "settling_sensitivity": 0.05,
                "info_sensitivity": 0.5,
            },
            "nbc": {
                "distance": "euclidean",
                "fast_k": 0.05,
                "distance_tie_breaker": "first",
            },
        },
        "cases": cases,
    }
    (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
