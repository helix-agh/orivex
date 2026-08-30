"""Measure ELA feature stability under independently seeded random designs.

For each dimension and sample budget, the same uniformly sampled X is reused across a small
suite of analytical landscapes. Only the random-design seed changes between repetitions.
The primary score estimates how much observed variance is landscape signal rather than
within-landscape sampling noise. Seed-to-seed landscape-rank agreement is reported separately.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_info, threadpool_limits

import bflacco
from bflacco import LandscapeSample, compute, list_features

Array = np.ndarray


@dataclass(frozen=True, slots=True)
class Landscape:
    name: str
    evaluate: Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class Observation:
    dimension: int
    observations: int
    sample_multiplier: int
    seed: int
    landscape: str
    feature: str
    value: float | None
    status: str


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    feature: str
    repeatability_median: float | None
    repeatability_minimum: float | None
    rank_agreement_median: float | None
    median_abs_cv: float | None
    successful_observations: int
    failed_observations: int
    rating: str


def sphere(x: Array) -> Array:
    return np.sum(x * x, axis=1)


def ellipsoid(x: Array) -> Array:
    weights = np.logspace(0.0, 6.0, x.shape[1])
    return np.sum(weights * x * x, axis=1)


def linear(x: Array) -> Array:
    weights = np.linspace(0.5, 1.5, x.shape[1])
    return x @ weights + 2.0


def rastrigin(x: Array) -> Array:
    return 10.0 * x.shape[1] + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


def rosenbrock(x: Array) -> Array:
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1.0) ** 2, axis=1)


def ackley(x: Array) -> Array:
    dimension = x.shape[1]
    radial = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x, axis=1) / dimension))
    periodic = -np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / dimension)
    return radial + periodic + 20.0 + np.e


LANDSCAPES = (
    Landscape("linear", linear),
    Landscape("sphere", sphere),
    Landscape("ellipsoid", ellipsoid),
    Landscape("rosenbrock", rosenbrock),
    Landscape("rastrigin", rastrigin),
    Landscape("ackley", ackley),
)


def transform_y(y: Array, mode: str) -> Array:
    if mode == "raw":
        return y
    standard_deviation = np.std(y, ddof=0)
    if standard_deviation == 0.0:
        return y - np.mean(y)
    return (y - np.mean(y)) / standard_deviation


def collect_observations(
    *,
    dimensions: list[int],
    sample_multipliers: list[int],
    seeds: int,
    base_seed: int,
    y_mode: str,
) -> list[Observation]:
    feature_names = tuple(spec.name for spec in list_features())
    collected = []
    for dimension in dimensions:
        lower = np.full(dimension, -5.0)
        upper = np.full(dimension, 5.0)
        for multiplier in sample_multipliers:
            observations = dimension * multiplier
            for seed_index in range(seeds):
                seed_sequence = np.random.SeedSequence(
                    (base_seed, dimension, multiplier, seed_index)
                )
                seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
                rng = np.random.Generator(np.random.PCG64(seed_sequence))
                x = rng.uniform(lower, upper, size=(observations, dimension))
                for landscape in LANDSCAPES:
                    y = transform_y(landscape.evaluate(x), y_mode)
                    result = compute(LandscapeSample(x, y, lower, upper), feature_names)
                    for feature, output in result.values.items():
                        collected.append(
                            Observation(
                                dimension,
                                observations,
                                multiplier,
                                seed,
                                landscape.name,
                                feature,
                                None if output.value is None else float(output.value),
                                output.status.value,
                            )
                        )
    return collected


def repeatability(values_by_landscape: list[Array]) -> float | None:
    """Estimate single-design reliability with sampling variance removed from signal."""
    if not values_by_landscape:
        return None
    repetitions = values_by_landscape[0].size
    if repetitions < 2 or any(values.size != repetitions for values in values_by_landscape):
        return None
    means = np.array([np.mean(values) for values in values_by_landscape])
    within = float(np.mean([np.var(values, ddof=1) for values in values_by_landscape]))
    between_means = float(np.var(means, ddof=1))
    signal = max(between_means - within / repetitions, 0.0)
    if signal == 0.0 and within == 0.0:
        return None
    return signal / (signal + within)


def rank_agreement(values_by_landscape: list[Array]) -> float | None:
    if not values_by_landscape:
        return None
    matrix = np.stack(values_by_landscape, axis=1)
    correlations = []
    for left_index in range(matrix.shape[0] - 1):
        left = rankdata(matrix[left_index])
        if np.ptp(left) == 0.0:
            continue
        for right_index in range(left_index + 1, matrix.shape[0]):
            right = rankdata(matrix[right_index])
            if np.ptp(right) == 0.0:
                continue
            correlation = float(np.corrcoef(left, right)[0, 1])
            if np.isfinite(correlation):
                correlations.append(correlation)
    return statistics.median(correlations) if correlations else None


def absolute_cv(values: Array) -> float | None:
    mean = float(np.mean(values))
    scale = float(np.max(np.abs(values)))
    if abs(mean) <= np.finfo(np.float64).eps * max(1.0, scale) * 100.0:
        return None
    return float(np.std(values, ddof=1) / abs(mean))


def rating(score: float | None) -> str:
    if score is None:
        return "constant_or_unidentifiable"
    if score >= 0.90:
        return "excellent"
    if score >= 0.75:
        return "good"
    if score >= 0.50:
        return "moderate"
    return "poor"


def summarize(
    observations: list[Observation],
    *,
    dimensions: list[int],
    sample_multipliers: list[int],
) -> list[StabilitySummary]:
    feature_names = tuple(spec.name for spec in list_features())
    summaries = []
    for feature in feature_names:
        feature_rows = [row for row in observations if row.feature == feature]
        successful = [row for row in feature_rows if row.value is not None]
        repeatabilities = []
        agreements = []
        cvs = []
        for dimension in dimensions:
            for multiplier in sample_multipliers:
                cell = [
                    row
                    for row in successful
                    if row.dimension == dimension and row.sample_multiplier == multiplier
                ]
                grouped = []
                complete = True
                for landscape in LANDSCAPES:
                    values = np.array(
                        [row.value for row in cell if row.landscape == landscape.name],
                        dtype=np.float64,
                    )
                    if values.size == 0:
                        complete = False
                        break
                    grouped.append(values)
                    cv = absolute_cv(values)
                    if cv is not None:
                        cvs.append(cv)
                if not complete:
                    continue
                reliability = repeatability(grouped)
                if reliability is not None:
                    repeatabilities.append(reliability)
                agreement = rank_agreement(grouped)
                if agreement is not None:
                    agreements.append(agreement)

        median_reliability = statistics.median(repeatabilities) if repeatabilities else None
        summaries.append(
            StabilitySummary(
                feature,
                median_reliability,
                min(repeatabilities) if repeatabilities else None,
                statistics.median(agreements) if agreements else None,
                statistics.median(cvs) if cvs else None,
                len(successful),
                len(feature_rows) - len(successful),
                rating(median_reliability),
            )
        )
    return sorted(
        summaries,
        key=lambda item: (
            item.repeatability_median is not None,
            item.repeatability_median if item.repeatability_median is not None else -1.0,
        ),
        reverse=True,
    )


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_table(summaries: list[StabilitySummary]) -> None:
    header = (
        "feature",
        "repeatability",
        "worst cell",
        "rank agreement",
        "median |CV|",
        "rating",
        "failures",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for item in summaries:
        print(
            " | ".join(
                (
                    item.feature,
                    format_optional(item.repeatability_median),
                    format_optional(item.repeatability_minimum),
                    format_optional(item.rank_agreement_median),
                    format_optional(item.median_abs_cv),
                    item.rating,
                    str(item.failed_observations),
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--sample-multipliers", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument("--y-mode", choices=("raw", "standardized"), default="raw")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    positive = (*args.dimensions, *args.sample_multipliers, args.seeds, args.threads)
    if any(value < 1 for value in positive):
        parser.error("dimensions, sample multipliers, seeds, and threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        native_libraries = threadpool_info()
        observations = collect_observations(
            dimensions=args.dimensions,
            sample_multipliers=args.sample_multipliers,
            seeds=args.seeds,
            base_seed=args.base_seed,
            y_mode=args.y_mode,
        )
    summaries = summarize(
        observations,
        dimensions=args.dimensions,
        sample_multipliers=args.sample_multipliers,
    )
    elapsed = time.perf_counter() - started
    report = {
        "method": {
            "sampling": "independent uniform random designs in [-5, 5]^d",
            "paired_x_across_landscapes": True,
            "landscapes": [landscape.name for landscape in LANDSCAPES],
            "primary_metric": "variance-component single-design repeatability",
            "secondary_metric": "median pairwise Spearman landscape-rank agreement",
            "cv_warning": "CV is diagnostic only and is unreliable near a zero mean",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "bflacco": bflacco.__version__,
            "native_libraries": native_libraries,
        },
        "settings": {
            "dimensions": args.dimensions,
            "sample_multipliers": args.sample_multipliers,
            "seeds": args.seeds,
            "base_seed": args.base_seed,
            "y_mode": args.y_mode,
            "threads": args.threads,
        },
        "runtime_seconds": elapsed,
        "summaries": [asdict(summary) for summary in summaries],
        "observations": [asdict(observation) for observation in observations],
    }
    print(
        f"Random-design stability: {args.seeds} seeds, dimensions={args.dimensions}, "
        f"sample multipliers={args.sample_multipliers}, y={args.y_mode}.\n"
    )
    print_table(summaries)
    print(f"\nCompleted in {elapsed:.2f} seconds.")
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
