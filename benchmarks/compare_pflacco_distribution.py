"""Compare pflacco and bflacco distribution-feature CPU and wall time.

The comparison is intentionally transparent about scope: bflacco currently computes the two
common outputs (skewness and kurtosis), whereas pflacco's public group call additionally computes
KDE-based peak count. The benchmark therefore measures the user-facing cost of requesting those
common outputs from each current API, not identical internal instruction streams.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import bflacco
from bflacco import LandscapeSample, compute


@dataclass(frozen=True, slots=True)
class Timing:
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Comparison:
    observations: int
    dimension: int
    mode: str
    pflacco: Timing
    bflacco: Timing
    cpu_speedup: float
    wall_speedup: float


def measure(function: Callable[[], object], *, warmups: int, repeats: int, calls: int) -> Timing:
    for _ in range(warmups):
        function()

    cpu_samples = []
    wall_samples = []
    for _ in range(repeats):
        cpu_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        for _ in range(calls):
            function()
        wall_samples.append((time.perf_counter_ns() - wall_started) / calls / 1e9)
        cpu_samples.append((time.process_time_ns() - cpu_started) / calls / 1e9)

    return Timing(
        cpu_seconds=statistics.median(cpu_samples),
        wall_seconds=statistics.median(wall_samples),
    )


def load_pflacco_calculator(pflacco_root: Path) -> tuple[Callable[..., dict], str]:
    resolved = pflacco_root.resolve()
    if not (resolved / "pflacco" / "classical_ela_features.py").is_file():
        raise FileNotFoundError(f"not a pflacco source checkout: {resolved}")
    sys.path.insert(0, str(resolved))
    try:
        module = importlib.import_module("pflacco.classical_ela_features")
    finally:
        sys.path.pop(0)
    calculator = module.calculate_ela_distribution
    return calculator, str(Path(module.__file__).resolve())


def common_values(result: dict[str, object]) -> tuple[float, float]:
    return float(result["ela_distr.skewness"]), float(result["ela_distr.kurtosis"])


def bflacco_values(result: object) -> tuple[float, float]:
    outputs = result.values
    return (
        float(outputs["ela_distr.skewness"].value),
        float(outputs["ela_distr.kurtosis"].value),
    )


def compare_case(
    calculate_pflacco: Callable[..., dict],
    *,
    observations: int,
    dimension: int,
    rng: np.random.Generator,
    warmups: int,
    repeats: int,
    calls: int,
) -> list[Comparison]:
    x = rng.uniform(-5.0, 5.0, size=(observations, dimension))
    y = np.sum(x**2, axis=1)
    lower = np.full(dimension, -5.0)
    upper = np.full(dimension, 5.0)

    frame = pd.DataFrame(x, columns=[f"x{index}" for index in range(dimension)])
    series = pd.Series(y, name="y")
    sample = LandscapeSample(x, y, lower, upper)

    def pflacco_prepared() -> dict:
        return calculate_pflacco(frame, series)

    def bflacco_prepared():
        return compute(sample, "ela_distr.*")

    def pflacco_end_to_end() -> dict:
        return calculate_pflacco(x, y)

    def bflacco_end_to_end():
        return compute(LandscapeSample(x, y, lower, upper), "ela_distr.*")

    expected = common_values(pflacco_prepared())
    actual = bflacco_values(bflacco_prepared())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    comparisons = []
    for mode, pflacco_call, bflacco_call in (
        ("prepared", pflacco_prepared, bflacco_prepared),
        ("end_to_end", pflacco_end_to_end, bflacco_end_to_end),
    ):
        pflacco_timing = measure(pflacco_call, warmups=warmups, repeats=repeats, calls=calls)
        bflacco_timing = measure(bflacco_call, warmups=warmups, repeats=repeats, calls=calls)
        comparisons.append(
            Comparison(
                observations=observations,
                dimension=dimension,
                mode=mode,
                pflacco=pflacco_timing,
                bflacco=bflacco_timing,
                cpu_speedup=pflacco_timing.cpu_seconds / bflacco_timing.cpu_seconds,
                wall_speedup=pflacco_timing.wall_seconds / bflacco_timing.wall_seconds,
            )
        )
    return comparisons


def print_table(comparisons: list[Comparison]) -> None:
    header = (
        "n",
        "mode",
        "pflacco CPU s",
        "bflacco CPU s",
        "CPU speedup",
        "pflacco wall s",
        "bflacco wall s",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for item in comparisons:
        print(
            " | ".join(
                (
                    str(item.observations),
                    item.mode,
                    f"{item.pflacco.cpu_seconds:.9f}",
                    f"{item.bflacco.cpu_seconds:.9f}",
                    f"{item.cpu_speedup:.2f}x",
                    f"{item.pflacco.wall_seconds:.9f}",
                    f"{item.bflacco.wall_seconds:.9f}",
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pflacco-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pflacco",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1_000, 2_000, 5_000])
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()
    if any(size < 4 for size in args.sizes):
        parser.error("every sample size must be at least 4")
    if args.dimension < 1 or args.warmups < 0 or args.repeats < 1 or args.calls < 1:
        parser.error("dimension, repeats, and calls must be positive; warmups must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    calculate_pflacco, source = load_pflacco_calculator(args.pflacco_root)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    comparisons = []
    for observations in args.sizes:
        comparisons.extend(
            compare_case(
                calculate_pflacco,
                observations=observations,
                dimension=args.dimension,
                rng=rng,
                warmups=args.warmups,
                repeats=args.repeats,
                calls=args.calls,
            )
        )

    report = {
        "scope": {
            "common_outputs": ["ela_distr.skewness", "ela_distr.kurtosis"],
            "pflacco_additional_output": "ela_distr.number_of_peaks",
            "objective_evaluation_and_sample_generation_time_included": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "bflacco": bflacco.__version__,
            "pflacco_source": source,
        },
        "settings": {
            "dimension": args.dimension,
            "landscape": "sphere",
            "sampling": "uniform[-5, 5]",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "calls": args.calls,
            "seed": args.seed,
        },
        "comparisons": [asdict(item) for item in comparisons],
    }

    print("Outputs verified equal before timing: skewness, kurtosis")
    print("pflacco additionally computes number_of_peaks; bflacco currently does not.\n")
    print_table(comparisons)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
