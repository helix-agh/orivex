"""Guard the diagnostic harness against falsely declaring numerical agreement."""

from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pytest

pytest.importorskip("threadpoolctl")

import bbob_correctness as benchmark
import correctness_report as reporting


def test_relative_zero_is_undefined_but_absolute_error_is_checked():
    equal = benchmark.compare("test", benchmark.scalar(0), benchmark.scalar(0), 1e-10, 1e-8)
    assert equal["verdict"] == "pass"
    assert equal["relative_error"] is None
    assert equal["absolute_error"] == 0
    assert equal["reference_zero"]
    different = benchmark.compare("test", benchmark.scalar(1), benchmark.scalar(0), 1e-10, 1e-8)
    assert different["verdict"] == "review"
    assert different["absolute_error"] == 1
    assert different["relative_error"] is None


def test_near_zero_relative_error_is_not_clipped():
    row = benchmark.compare("test", benchmark.scalar(1e-12), benchmark.scalar(1e-20), 1e-10, 1e-8)
    assert row["verdict"] == "pass"
    assert row["reference_near_zero"]
    assert row["relative_error"] == pytest.approx(1e8 - 1)


@pytest.mark.parametrize(
    "value, status", [(np.nan, "nan"), (np.inf, "posinf"), (-np.inf, "neginf")]
)
def test_nonfinite_values_are_never_passes_and_remain_json_portable(value, status):
    reference = benchmark.scalar(value)
    assert reference["status"] == status
    row = benchmark.compare("test", reference, reference, 1e-10, 1e-8)
    assert row["comparison"] == "both_undefined"
    assert row["verdict"] == "review"
    assert row["absolute_error"] is None
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("status", ["invalid", "unavailable", "error", "missing"])
def test_failure_status_cannot_disappear_into_error_distribution(status):
    row = benchmark.compare(
        "test",
        benchmark.scalar(None, status=status, message="reason"),
        benchmark.scalar(1),
        1e-10,
        1e-8,
    )
    assert row["verdict"] == "review"
    assert row["absolute_error"] is None
    assert row["orivex_status"] == status
    assert row["orivex_message"] == "reason"


def test_arithmetic_overflow_does_not_become_a_match():
    row = benchmark.compare("test", benchmark.scalar(1e308), benchmark.scalar(-1e308), 1e-10, 1e-8)
    assert row["error_overflow"]
    assert row["verdict"] == "review"
    json.dumps(row, allow_nan=False)


def test_reference_receives_copies_and_warnings_survive_exception():
    x = np.ones((4, 2))
    y = np.ones(4)

    def broken(x, y):
        x[:] = 9
        y[:] = 8
        warnings.warn("reference warning", UserWarning, stacklevel=1)
        raise ValueError("reference failure")

    outputs, messages = benchmark.collect_family(broken, x, y, {})
    np.testing.assert_array_equal(x, np.ones((4, 2)))
    np.testing.assert_array_equal(y, np.ones(4))
    assert outputs["__error__"]["status"] == "error"
    assert "reference failure" in outputs["__error__"]["message"]
    assert messages == ["UserWarning: reference warning"]


def test_different_definition_does_not_pass_without_oracle():
    row = benchmark.compare(
        benchmark.QUADRATIC, benchmark.scalar(0.1), benchmark.scalar(0.9), 1e-10, 1e-8
    )
    assert row["comparison"] == "definition_difference"
    assert row["absolute_error"] == pytest.approx(0.8)
    assert row["verdict"] == "review"


def test_oracle_fits_complete_quadratic_and_rejects_rank_deficiency():
    pytest.importorskip("sklearn")
    x = np.random.default_rng(4).uniform(-5, 5, (100, 2))
    y = 13 + x[:, 0] ** 2 - 2 * x[:, 0] * x[:, 1] + x[:, 1]
    assert benchmark.quadratic_oracle(x, y) == pytest.approx(1)
    with pytest.raises(ValueError, match="rank deficient"):
        benchmark.quadratic_oracle(np.column_stack([x[:, 0], x[:, 0]]), y)


@pytest.mark.parametrize(
    "current, reference, oracle, verdict",
    [
        (0.1, 0.9, 0.1, "pass"),
        (0.1, 0.9, 0.2, "review"),
        (0.1, None, 0.1, "review"),
        (None, 0.9, 0.1, "review"),
        (0.1, 0.9, np.nan, "review"),
    ],
)
def test_quadratic_verdict_requires_valid_reference_and_matching_oracle(
    monkeypatch, current, reference, oracle, verdict
):
    monkeypatch.setattr(benchmark, "quadratic_oracle", lambda x, y: oracle)
    row = benchmark.compare(
        benchmark.QUADRATIC,
        benchmark.scalar(current),
        benchmark.scalar(reference, status="error" if reference is None else "ok"),
        1e-10,
        1e-8,
    )
    benchmark.check_quadratic(row, np.zeros((2, 1)), np.zeros(2))
    assert row["verdict"] == verdict
    json.dumps(row, allow_nan=False)


def test_summary_counts_failures_zeros_and_worst_cases():
    feature = benchmark.FEATURES[0]
    rows = []
    for index, (a, b) in enumerate([(0, 0), (4, 2), (None, 1)]):
        row = benchmark.compare(feature, benchmark.scalar(a), benchmark.scalar(b), 1e-10, 1e-8)
        row.update(dimension=2, case_id=f"case{index}")
        rows.append(row)
    summary = reporting.summarize(rows)[0]
    assert summary["cases"] == 3
    assert summary["review_required"] == 2
    assert summary["exact_matches"] == 1
    assert summary["relative_undefined"] == 1
    assert summary["absolute_error"]["count"] == 2
    assert summary["absolute_error"]["worst_case"] == "case1"


@pytest.mark.parametrize(
    "args",
    [
        ["--functions", "0"],
        ["--seeds", "1", "1"],
        ["--seeds", "-1"],
        ["--workers", "0"],
        ["--atol", "nan"],
        ["--rtol", "inf"],
        ["--dimensions", "1"],
    ],
)
def test_invalid_experiments_fail_before_execution(args):
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args(args)
    assert exc.value.code == 2


def test_case_failure_preserves_every_feature(tmp_path):
    case = benchmark.Case(1, 2, 1, 0, 200)
    result = benchmark.run_case((case, tmp_path / "absent", tmp_path, 1e-10, 1e-8))
    assert len(result["rows"]) == len(benchmark.FEATURES)
    assert all(row["comparison"] == "execution_error" for row in result["rows"])


def test_raw_inputs_and_case_replay_with_reference_failure(tmp_path, monkeypatch):
    pytest.importorskip("ioh")
    pytest.importorskip("sklearn")
    (tmp_path / "cases").mkdir()
    observed = []
    original = benchmark.compute

    def track(sample, names, **kwargs):
        assert names == benchmark.FEATURES
        assert kwargs["y_normalization"] is None
        assert kwargs["workers"] == 1
        observed.append((sample.x.copy(), sample.y.copy()))
        return original(sample, names, **kwargs)

    def broken(x, y, **kwargs):
        raise ValueError("isolated reference failure")

    monkeypatch.setattr(benchmark, "compute", track)
    calculators = dict.fromkeys(benchmark.FAMILIES, broken)
    case = benchmark.Case(1, 2, 1, 0, 200)
    first = benchmark.evaluate_case(case, calculators, tmp_path, 1e-10, 1e-8)
    second = benchmark.evaluate_case(case, calculators, tmp_path, 1e-10, 1e-8)
    assert first["rows"] == second["rows"]
    assert len(observed) == 2  # One orivex call per dataset, including replay.
    assert first["sample_fingerprint"] == second["sample_fingerprint"]
    assert len(first["rows"]) == 23
    assert all(row["orivex_status"] == "ok" for row in first["rows"])
    assert all(row["verdict"] == "review" for row in first["rows"])
    with np.load(tmp_path / first["input_file"]) as saved:
        for x, y in observed:
            np.testing.assert_array_equal(x, saved["x"])
            np.testing.assert_array_equal(y, saved["y"])
        assert saved["y"].max() > 1  # Actual BBOB values, not min-max preprocessing.


def test_orivex_exception_preserves_reference_results(tmp_path, monkeypatch):
    pytest.importorskip("ioh")
    (tmp_path / "cases").mkdir()

    def broken(*args, **kwargs):
        warnings.warn("engine warning", UserWarning, stacklevel=1)
        raise RuntimeError("engine failure")

    def reference(x, y, **kwargs):
        return dict.fromkeys(benchmark.FEATURES, 1.0)

    monkeypatch.setattr(benchmark, "compute", broken)
    monkeypatch.setattr(benchmark, "quadratic_oracle", lambda x, y: 1.0)
    result = benchmark.evaluate_case(
        benchmark.Case(1, 2, 1, 0, 200),
        dict.fromkeys(benchmark.FAMILIES, reference),
        tmp_path,
        1e-10,
        1e-8,
    )
    assert len(result["rows"]) == 23
    assert all(row["pflacco_status"] == "ok" for row in result["rows"])
    assert all(row["orivex_status"] == "error" for row in result["rows"])
    assert all(row["verdict"] == "review" for row in result["rows"])
    assert result["warnings"]["orivex"] == ["UserWarning: engine warning"]


@pytest.mark.parametrize("complete", [False, True])
def test_plot_only_rebuilds_from_cases_without_loading_reference(tmp_path, monkeypatch, complete):
    pytest.importorskip("matplotlib")
    row = benchmark.compare(
        benchmark.FEATURES[0], benchmark.scalar(0), benchmark.scalar(0), 1e-10, 1e-8
    )
    row.update(dimension=2, function=1, case_id="test")
    manifest = {"complete": complete, "expected_cases": 1 if complete else 2}
    benchmark.write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "cases").mkdir()
    benchmark.write_json(
        tmp_path / "cases" / "test.json",
        {"rows": [row], "pflacco_only": {"extra.feature": benchmark.scalar(2)}},
    )
    (tmp_path / "summary.csv").write_text("stale summary", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("plot-only attempted feature computation or reference import")

    monkeypatch.setattr(benchmark, "load_reference", forbidden)
    monkeypatch.setattr(benchmark, "compute", forbidden)
    assert benchmark.main(["--plot-only", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "plots" / f"{benchmark.FEATURES[0]}.png").is_file()
    assert (tmp_path / "plots" / "heatmap_d2.pdf").is_file()
    assert (tmp_path / "comparisons.csv").read_text().count("test") == 1
    assert "stale summary" not in (tmp_path / "summary.csv").read_text()
    report = reporting.load_report(tmp_path)
    assert report["summary"][0]["exact_matches"] == 1
    assert report["completed_cases"] == 1
    assert report["pflacco_only_outputs"] == ["extra.feature"]
    assert f"Run complete: {complete}" in (tmp_path / "README.md").read_text()
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert not (tmp_path / "report.json").exists()
    with (tmp_path / "review.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert "feature" in reader.fieldnames
        assert list(reader) == []


def test_reporting_requires_completed_case_files(tmp_path):
    benchmark.write_json(tmp_path / "manifest.json", {"complete": False, "expected_cases": 1})
    with pytest.raises(ValueError, match="No completed case files"):
        reporting.load_report(tmp_path)
