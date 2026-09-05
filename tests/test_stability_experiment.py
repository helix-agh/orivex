"""Experimental-design checks; run with the benchmark extra installed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("threadpoolctl")

spec = importlib.util.spec_from_file_location(
    "stability_experiment", Path(__file__).parents[1] / "benchmarks/analyze_feature_stability.py"
)
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def test_missing_observations_are_joined_by_seed():
    rows = []
    for index, landscape in enumerate(experiment.LANDSCAPES):
        for seed in [3, 1, 2, 0]:
            missing = seed == index % 2
            rows.append(
                experiment.Observation(
                    2,
                    500,
                    250,
                    seed,
                    landscape.name,
                    "ela_distr.skewness",
                    None if missing else index + seed / 10,
                    "invalid" if missing else "ok",
                )
            )
    grouped, seeds = experiment.aligned_values(rows)
    assert seeds == [2, 3]
    np.testing.assert_allclose(grouped[0], [0.2, 0.3])
    summaries = experiment.summarize(rows, dimensions=[2], sample_multipliers=[250])
    summary = next(s for s in summaries if s.feature == "ela_distr.skewness")
    assert summary.failed_observations == 6
    assert summary.rank_agreement_median == pytest.approx(1)


def test_lhs_modes_share_identical_raw_samples_and_are_reproducible(monkeypatch):
    captured = []
    original = experiment.compute

    def capture(sample, *args, **kwargs):
        captured.append(sample.x.copy())
        return original(sample, *args, **kwargs)

    monkeypatch.setattr(experiment, "compute", capture)
    settings = {
        "dimensions": [2],
        "sample_multipliers": [5],
        "seeds": 2,
        "base_seed": 42,
        "y_modes": ["raw", "minmax", "zscore"],
        "workers": 1,
        "sampling": "lhs",
    }
    rows = experiment.collect_observations(**settings)
    assert rows == experiment.collect_observations(**settings)
    for column in captured[0].T:
        bins = np.floor((column + 5) / 10 * 10).astype(int)
        np.testing.assert_array_equal(np.sort(bins), np.arange(10))
    first = [r for r in rows if r.seed == rows[0].seed and r.landscape == rows[0].landscape]
    assert len({r.sample_fingerprint for r in first}) == 1
    assert {r.y_mode for r in first} == {"raw", "minmax", "zscore"}


def test_detailed_results_preserve_landscape_and_dimension():
    rows = [
        experiment.Observation(
            2,
            500,
            250,
            seed,
            landscape.name,
            "ela_distr.skewness",
            index + seed / 10,
            "ok",
            "minmax",
        )
        for index, landscape in enumerate(experiment.LANDSCAPES)
        for seed in range(3)
    ]
    cells, landscapes = experiment.detailed_summaries(rows)
    assert cells[0]["complete_seeds"] == 3
    assert len(landscapes) == 6
    assert all(row["valid"] == 3 and row["dimension"] == 2 for row in landscapes)
    assert cells[0]["repeatability_ci95"] is not None
