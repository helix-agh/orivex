"""Microbenchmark the first individually selectable feature slice."""

from __future__ import annotations

import json
import platform
import statistics
import timeit

import numpy as np

import orivex
from orivex import LandscapeSample, compute


def benchmark(sample: LandscapeSample, selector: str, *, number: int = 200) -> dict[str, object]:
    result = compute(sample, selector)
    timings = timeit.repeat(lambda: compute(sample, selector), repeat=7, number=number)
    return {
        "observations": sample.n_observations,
        "dimension": sample.dimension,
        "selector": selector,
        "intermediates": result.metadata.computed_intermediates,
        "calls_per_repeat": number,
        "median_seconds_per_call": statistics.median(timings) / number,
        "minimum_seconds_per_call": min(timings) / number,
    }


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(20260830))
    records = []
    for observations in (100, 500, 2_000, 10_000):
        x = rng.uniform(-5.0, 5.0, size=(observations, 10))
        y = np.sum(x**2, axis=1)
        sample = LandscapeSample(x, y, [-5.0] * 10, [5.0] * 10)
        for selector in (
            "ela_distr.skewness",
            "ela_distr.kurtosis",
            "ela_distr.*",
        ):
            records.append(benchmark(sample, selector))

    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "orivex": orivex.__version__,
            "numpy": np.__version__,
        },
        "records": records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
