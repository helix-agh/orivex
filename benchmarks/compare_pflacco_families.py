"""Compare pflacco and orivex CPU and wall time for every implemented family.

Both implementations receive the same sphere samples. The comparison covers the common
implemented outputs from ``ela_distr``, ``ela_meta``, ``ic``, and ``nbc``. IC uses the same
explicit lexicographic starting observation, epsilon grid, and nearest-neighbour sorting policy.
NBC uses Euclidean distances and deterministic first-index tie handling. Numerical outputs are
verified before any timing result is accepted.
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
import scipy
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

import orivex
from orivex import LandscapeSample, compute


@dataclass(frozen=True, slots=True)
class Timing:
    cpu_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class Comparison:
    observations: int
    dimension: int
    family: str
    mode: str
    pflacco: Timing
    orivex: Timing
    cpu_speedup: float
    wall_speedup: float


def measure_once(function: Callable[[], object], *, calls: int) -> Timing:
    cpu_started = time.process_time_ns()
    wall_started = time.perf_counter_ns()
    for _ in range(calls):
        function()
    wall_seconds = (time.perf_counter_ns() - wall_started) / calls / 1e9
    cpu_seconds = (time.process_time_ns() - cpu_started) / calls / 1e9
    return Timing(cpu_seconds, wall_seconds)


def measure_pair(
    pflacco_call: Callable[[], object],
    orivex_call: Callable[[], object],
    *,
    repeats: int,
    calls: int,
) -> tuple[Timing, Timing]:
    """Measure both implementations in alternating order to limit order bias."""
    samples: dict[str, list[Timing]] = {"pflacco": [], "orivex": []}
    for repeat in range(repeats):
        ordered = (
            (("pflacco", pflacco_call), ("orivex", orivex_call))
            if repeat % 2 == 0
            else (("orivex", orivex_call), ("pflacco", pflacco_call))
        )
        for name, function in ordered:
            samples[name].append(measure_once(function, calls=calls))

    def median(items: list[Timing]) -> Timing:
        return Timing(
            statistics.median(item.cpu_seconds for item in items),
            statistics.median(item.wall_seconds for item in items),
        )

    return median(samples["pflacco"]), median(samples["orivex"])


def load_pflacco_calculators(
    pflacco_root: Path,
) -> tuple[Callable, Callable, Callable, Callable, str]:
    resolved = pflacco_root.resolve()
    source_file = resolved / "pflacco" / "classical_ela_features.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"not a pflacco source checkout: {resolved}")
    sys.path.insert(0, str(resolved))
    try:
        module = importlib.import_module("pflacco.classical_ela_features")
    finally:
        sys.path.pop(0)
    module_file = module.__file__
    if module_file is None:
        raise RuntimeError("the loaded pflacco module has no source path")
    return (
        module.calculate_ela_distribution,
        module.calculate_ela_meta,
        module.calculate_information_content,
        module.calculate_nbc,
        str(Path(module_file).resolve()),
    )


def lexicographic_start(x: np.ndarray) -> int:
    keys = tuple(x[:, column] for column in range(x.shape[1] - 1, -1, -1))
    return int(np.lexsort(keys)[0])


def orivex_values(result) -> dict[str, float]:
    values = {}
    for name, output in result.values.items():
        if output.value is None:
            message = f"orivex returned {output.status.value} for {name}: {output.message}"
            raise AssertionError(message)
        values[name] = float(output.value)
    return values


def verified(
    pflacco_call: Callable[[], dict[str, object]],
    orivex_call: Callable[[], object],
) -> None:
    legacy = pflacco_call()
    current = orivex_values(orivex_call())
    expected = {name: float(legacy[name]) for name in current}
    np.testing.assert_allclose(
        np.array(tuple(current.values())),
        np.array(tuple(expected.values())),
        rtol=2e-12,
        atol=2e-12,
    )


def compare_case(
    calculate_distribution: Callable,
    calculate_meta: Callable,
    calculate_ic: Callable,
    calculate_nbc: Callable,
    *,
    observations: int,
    dimension: int,
    rng: np.random.Generator,
    repeats: int,
    calls: int,
) -> list[Comparison]:
    x = rng.uniform(-5.0, 5.0, size=(observations, dimension))
    y = np.sum(x * x, axis=1)
    lower = np.full(dimension, -5.0)
    upper = np.full(dimension, 5.0)
    frame = pd.DataFrame(x, columns=[f"x{index}" for index in range(dimension)])
    series = pd.Series(y, name="y")
    sample = LandscapeSample(x, y, lower, upper)
    start = lexicographic_start(x)

    def pflacco_distribution_prepared() -> dict[str, object]:
        return calculate_distribution(frame, series)

    def pflacco_distribution_end_to_end() -> dict[str, object]:
        return calculate_distribution(x, y)

    def orivex_distribution_prepared():
        return compute(sample, "ela_distr.*")

    def orivex_distribution_end_to_end():
        return compute(LandscapeSample(x, y, lower, upper), "ela_distr.*")

    def pflacco_meta_prepared() -> dict[str, object]:
        return calculate_meta(frame, series)

    def pflacco_meta_end_to_end() -> dict[str, object]:
        return calculate_meta(x, y)

    def orivex_meta_prepared():
        return compute(sample, "ela_meta.*")

    def orivex_meta_end_to_end():
        return compute(LandscapeSample(x, y, lower, upper), "ela_meta.*")

    def pflacco_ic_prepared() -> dict[str, object]:
        return calculate_ic(frame, series, ic_nn_start=start, seed=20260830)

    def pflacco_ic_end_to_end() -> dict[str, object]:
        return calculate_ic(x, y, ic_nn_start=start, seed=20260830)

    def orivex_ic_prepared():
        return compute(sample, "ic.*")

    def orivex_ic_end_to_end():
        return compute(LandscapeSample(x, y, lower, upper), "ic.*")

    def pflacco_nbc_prepared() -> dict[str, object]:
        return calculate_nbc(frame, series, dist_tie_breaker="first")

    def pflacco_nbc_end_to_end() -> dict[str, object]:
        return calculate_nbc(x, y, dist_tie_breaker="first")

    def orivex_nbc_prepared():
        return compute(sample, "nbc.*")

    def orivex_nbc_end_to_end():
        return compute(LandscapeSample(x, y, lower, upper), "nbc.*")

    call_sets = (
        (
            "ela_distr",
            "prepared",
            pflacco_distribution_prepared,
            orivex_distribution_prepared,
        ),
        (
            "ela_distr",
            "end_to_end",
            pflacco_distribution_end_to_end,
            orivex_distribution_end_to_end,
        ),
        ("ela_meta", "prepared", pflacco_meta_prepared, orivex_meta_prepared),
        ("ela_meta", "end_to_end", pflacco_meta_end_to_end, orivex_meta_end_to_end),
        ("ic", "prepared", pflacco_ic_prepared, orivex_ic_prepared),
        ("ic", "end_to_end", pflacco_ic_end_to_end, orivex_ic_end_to_end),
        ("nbc", "prepared", pflacco_nbc_prepared, orivex_nbc_prepared),
        ("nbc", "end_to_end", pflacco_nbc_end_to_end, orivex_nbc_end_to_end),
    )
    comparisons = []
    for family, mode, pflacco_call, orivex_call in call_sets:
        verified(pflacco_call, orivex_call)
        pflacco_call()
        orivex_call()
        pflacco_timing, orivex_timing = measure_pair(
            pflacco_call,
            orivex_call,
            repeats=repeats,
            calls=calls,
        )
        comparisons.append(
            Comparison(
                observations,
                dimension,
                family,
                mode,
                pflacco_timing,
                orivex_timing,
                pflacco_timing.cpu_seconds / orivex_timing.cpu_seconds,
                pflacco_timing.wall_seconds / orivex_timing.wall_seconds,
            )
        )
    return comparisons


def print_table(comparisons: list[Comparison]) -> None:
    header = (
        "d",
        "n",
        "family",
        "mode",
        "pflacco CPU s",
        "orivex CPU s",
        "CPU speedup",
        "pflacco wall s",
        "orivex wall s",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for item in comparisons:
        print(
            " | ".join(
                (
                    str(item.dimension),
                    str(item.observations),
                    item.family,
                    item.mode,
                    f"{item.pflacco.cpu_seconds:.9f}",
                    f"{item.orivex.cpu_seconds:.9f}",
                    f"{item.cpu_speedup:.2f}x",
                    f"{item.pflacco.wall_seconds:.9f}",
                    f"{item.orivex.wall_seconds:.9f}",
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1_000])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="maximum threads in supported native numerical libraries",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if any(size < 3 for size in args.sizes):
        parser.error("sample sizes must be at least 3")
    if any(dimension < 1 for dimension in args.dimensions):
        parser.error("dimensions must be positive")
    if args.repeats < 1 or args.calls < 1 or args.threads < 1:
        parser.error("repeats, calls, and threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    (
        calculate_distribution,
        calculate_meta,
        calculate_ic,
        calculate_nbc,
        source,
    ) = load_pflacco_calculators(args.pflacco_root)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    comparisons = []
    with threadpool_limits(limits=args.threads):
        native_libraries = threadpool_info()
        for dimension in args.dimensions:
            for observations in args.sizes:
                comparisons.extend(
                    compare_case(
                        calculate_distribution,
                        calculate_meta,
                        calculate_ic,
                        calculate_nbc,
                        observations=observations,
                        dimension=dimension,
                        rng=rng,
                        repeats=args.repeats,
                        calls=args.calls,
                    )
                )

    report = {
        "scope": {
            "families": ["ela_distr", "ela_meta", "ic", "nbc"],
            "orivex_outputs": {
                "ela_distr": 2,
                "ela_meta": 5,
                "ic": 5,
                "nbc": 5,
            },
            "pflacco_additional_outputs": {
                "ela_distr": ["ela_distr.number_of_peaks"],
                "ela_meta": [
                    "ela_meta.lin_simple.coef.min",
                    "ela_meta.lin_simple.coef.max",
                    "ela_meta.lin_simple.coef.max_by_min",
                    "ela_meta.quad_simple.cond",
                ],
                "ic": [],
                "nbc": [],
            },
            "objective": "sphere",
            "sample_generation_timed": False,
            "outputs_verified_before_timing": True,
            "ic_start": "lexicographically_smallest_x",
            "nbc_distance": "euclidean",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "orivex": orivex.__version__,
            "pflacco_source": source,
        },
        "settings": {
            "sizes": args.sizes,
            "dimensions": args.dimensions,
            "repeats": args.repeats,
            "calls": args.calls,
            "threads": args.threads,
            "seed": args.seed,
        },
        "native_libraries": native_libraries,
        "comparisons": [asdict(item) for item in comparisons],
    }
    print("All 17 implemented outputs verified against pflacco before timing.")
    print("pflacco also computes 1 distribution and 4 meta-model legacy outputs.\n")
    print_table(comparisons)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
