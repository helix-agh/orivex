"""Benchmark selective bflacco meta-model computation across sample sizes."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass

import numpy as np

import bflacco
from bflacco import LandscapeSample, compute

SELECTORS = (
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.intercept",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
    "ela_meta.*",
)


@dataclass(frozen=True, slots=True)
class Timing:
    cpu_seconds: float
    wall_seconds: float


def measure(sample: LandscapeSample, selector: str, *, repeats: int, calls: int) -> Timing:
    compute(sample, selector)
    cpu_samples = []
    wall_samples = []
    for _ in range(repeats):
        cpu_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        for _ in range(calls):
            compute(sample, selector)
        wall_samples.append((time.perf_counter_ns() - wall_started) / calls / 1e9)
        cpu_samples.append((time.process_time_ns() - cpu_started) / calls / 1e9)
    return Timing(statistics.median(cpu_samples), statistics.median(wall_samples))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 2_000])
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("sample sizes must be positive")
    if args.dimension < 1 or args.repeats < 1 or args.calls < 1:
        parser.error("dimension, repeats, and calls must be positive")
    return args


def main() -> None:
    args = parse_args()
    rng = np.random.Generator(np.random.PCG64(args.seed))
    records = []
    for observations in args.sizes:
        x = rng.uniform(-5.0, 5.0, size=(observations, args.dimension))
        y = np.sum(x * x, axis=1)
        sample = LandscapeSample(
            x,
            y,
            np.full(args.dimension, -5.0),
            np.full(args.dimension, 5.0),
        )
        for selector in SELECTORS:
            result = compute(sample, selector)
            timing = measure(sample, selector, repeats=args.repeats, calls=args.calls)
            records.append(
                {
                    "observations": observations,
                    "dimension": args.dimension,
                    "selector": selector,
                    "intermediates": result.metadata.computed_intermediates,
                    "cpu_seconds": timing.cpu_seconds,
                    "wall_seconds": timing.wall_seconds,
                }
            )

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "bflacco": bflacco.__version__,
            "numpy": np.__version__,
        },
        "settings": {
            "dimension": args.dimension,
            "repeats": args.repeats,
            "calls": args.calls,
            "seed": args.seed,
        },
        "records": records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
