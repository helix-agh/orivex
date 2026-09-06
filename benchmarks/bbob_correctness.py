"""Compare every implemented NumPy feature with pflacco on shared, raw IOH BBOB samples."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from orivex import LandscapeSample, compute, list_features

QUADRATIC = "ela_meta.quad_w_interact.adj_r2"
FAMILIES = {
    "ela_distr": ("skewness", "kurtosis"),
    "ela_meta": (
        "lin_simple.adj_r2",
        "lin_simple.intercept",
        "lin_w_interact.adj_r2",
        "quad_simple.adj_r2",
        "quad_w_interact.adj_r2",
    ),
    "fitness_distance": (
        "fitness_std",
        "fitness_mean",
        "distance_mean",
        "distance_std",
        "fd_cov",
        "fd_correlation",
    ),
    "ic": ("h_max", "eps_s", "eps_max", "eps_ratio", "m0"),
    "nbc": (
        "nn_nb.sd_ratio",
        "nn_nb.mean_ratio",
        "nn_nb.cor",
        "dist_ratio.coeff_var",
        "nb_fitness.cor",
    ),
}
FEATURES = tuple(f"{family}.{name}" for family, names in FAMILIES.items() for name in names)
CALCULATORS = {
    "ela_distr": ("classical_ela_features", "calculate_ela_distribution"),
    "ela_meta": ("classical_ela_features", "calculate_ela_meta"),
    "fitness_distance": ("misc_features", "calculate_fitness_distance_correlation"),
    "ic": ("classical_ela_features", "calculate_information_content"),
    "nbc": ("classical_ela_features", "calculate_nbc"),
}
# A predeclared float64 diagnostic threshold, not a proof of mathematical correctness.
# Absolute tolerance handles reference zeros; relative tolerance handles raw BBOB scales.
DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-8
REFERENCE_SETTINGS = {
    "ela_distr": {"ela_distr_skewness_type": 3, "ela_distr_kurtosis_type": 3},
    "ela_meta": {},
    "fitness_distance": {"proportion_of_best": 0.1, "minimize": True, "minkowski_p": 2},
    "ic": {
        "ic_sorting": "nn",
        "ic_nn_neighborhood": 20,
        "ic_settling_sensitivity": 0.05,
        "ic_info_sensitivity": 0.5,
    },
    "nbc": {"fast_k": 0.05, "dist_tie_breaker": "first", "minimize": True},
}


@dataclass(frozen=True)
class Case:
    function: int
    dimension: int
    instance: int
    seed: int
    observations: int

    @property
    def key(self) -> str:
        return (
            f"f{self.function:02d}_d{self.dimension}_i{self.instance}"
            f"_s{self.seed}_n{self.observations}"
        )


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_reference(root: Path) -> dict:
    """Import unmodified modules and reject an accidentally shadowing installation."""
    root = root.resolve()
    if not (root / "pflacco" / "classical_ela_features.py").is_file():
        raise FileNotFoundError(f"Expected a pflacco source checkout at {root}")
    sys.path.insert(0, str(root))
    try:
        calculators = {}
        for family, (module_name, function_name) in CALCULATORS.items():
            module = importlib.import_module(f"pflacco.{module_name}")
            if Path(module.__file__).resolve().parent != root / "pflacco":
                raise RuntimeError(f"Wrong pflacco module loaded: {module.__file__}")
            calculators[family] = getattr(module, function_name)
        return calculators
    finally:
        sys.path.pop(0)


def scalar(value: object, *, status: str = "ok", message: str | None = None) -> dict:
    """Keep nonfinite categories without emitting nonstandard JSON NaN/Infinity."""
    if value is None:
        return {"value": None, "status": status if status != "ok" else "none", "message": message}
    number = float(value)
    if not math.isfinite(number):
        kind = "nan" if math.isnan(number) else ("posinf" if number > 0 else "neginf")
        return {"value": None, "status": kind, "message": message}
    return {"value": number, "status": status, "message": message}


def error_metrics(a: float, b: float, atol: float, rtol: float) -> dict:
    absolute = abs(a - b)
    tolerance = atol + rtol * abs(b)
    relative = absolute / abs(b) if b != 0 else None
    scaled = absolute / tolerance
    return {
        "absolute_error": absolute if math.isfinite(absolute) else None,
        "relative_error": relative if relative is None or math.isfinite(relative) else None,
        "tolerance_ratio": scaled if math.isfinite(scaled) else None,
        "error_overflow": not math.isfinite(absolute)
        or not math.isfinite(tolerance)
        or not math.isfinite(scaled)
        or (relative is not None and not math.isfinite(relative)),
        "reference_zero": b == 0,
        "reference_near_zero": abs(b) <= atol,
        "within_tolerance": math.isfinite(absolute)
        and math.isfinite(tolerance)
        and absolute <= tolerance,
    }


def compare(feature: str, current: dict, reference: dict, atol: float, rtol: float) -> dict:
    row = {
        "feature": feature,
        "definition_difference": feature == QUADRATIC,
        "orivex_value": current["value"],
        "orivex_status": current["status"],
        "orivex_message": current.get("message"),
        "pflacco_value": reference["value"],
        "pflacco_status": reference["status"],
        "pflacco_message": reference.get("message"),
        "atol": atol,
        "rtol": rtol,
        "absolute_error": None,
        "relative_error": None,
        "tolerance_ratio": None,
        "reference_zero": None,
        "reference_near_zero": None,
        "within_tolerance": None,
        "error_overflow": False,
    }
    if current["status"] == reference["status"] == "ok":
        row.update(error_metrics(current["value"], reference["value"], atol, rtol))
        row["comparison"] = (
            "definition_difference"
            if feature == QUADRATIC
            else ("pass" if row["within_tolerance"] else "mismatch")
        )
    elif "error" in (current["status"], reference["status"]):
        row["comparison"] = "execution_error"
    elif "missing" in (current["status"], reference["status"]):
        row["comparison"] = "missing_output"
    elif current["value"] is None and reference["value"] is None:
        row["comparison"] = "both_undefined"
    else:
        row["comparison"] = "status_mismatch"
    # Undefined values never establish numerical agreement, even if both sides are undefined.
    row["verdict"] = "pass" if row["comparison"] == "pass" else "review"
    return row


def quadratic_oracle(x: np.ndarray, y: np.ndarray) -> float:
    """Independent sklearn complete-degree-two fit, without orivex intermediates."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    design = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
    if len(y) <= design.shape[1] + 1 or np.var(y) == 0:
        raise ValueError("Adjusted R-squared requires residual degrees of freedom and varying y")
    model = LinearRegression().fit(design, y)
    if model.rank_ != design.shape[1]:
        raise ValueError("Complete quadratic design is rank deficient")
    return float(1 - (1 - model.score(design, y)) * (len(y) - 1) / (len(y) - design.shape[1] - 1))


def collect_family(calculator, x: np.ndarray, y: np.ndarray, settings: dict) -> tuple[dict, list]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = calculator(x.copy(), y.copy(), **settings)
            values = {
                name: scalar(value)
                for name, value in output.items()
                if not name.endswith("costs_runtime")
            }
        except Exception as exc:
            values = {
                "__error__": scalar(None, status="error", message=f"{type(exc).__name__}: {exc}")
            }
    return values, [f"{item.category.__name__}: {item.message}" for item in caught]


def evaluate_case(case: Case, calculators: dict, output: Path, atol: float, rtol: float) -> dict:
    import ioh

    problem = ioh.get_problem(
        case.function,
        instance=case.instance,
        dimension=case.dimension,
        problem_class=ioh.ProblemClass.BBOB,
    )
    lower = np.asarray(problem.bounds.lb, dtype=np.float64)
    upper = np.asarray(problem.bounds.ub, dtype=np.float64)
    entropy = [case.seed, case.function, case.instance, case.dimension, case.observations]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    x = rng.uniform(lower, upper, size=(case.observations, case.dimension))
    y = np.asarray([problem(point) for point in x], dtype=np.float64)
    sample = LandscapeSample(x, y, lower, upper)
    input_path = output / "cases" / f"{case.key}.npz"
    np.savez_compressed(input_path, x=x, y=y, lower=lower, upper=upper)
    details = {
        "case": asdict(case),
        "case_id": case.key,
        "function_name": problem.meta_data.name,
        "seed_sequence_entropy": entropy,
        "input_file": str(input_path.relative_to(output)),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "sample_fingerprint": sample.fingerprint,
        "sample_diagnostics": {
            "unique_x_rows": len(np.unique(x, axis=0)),
            "unique_objectives": len(np.unique(y)),
            "minimum_objective_ties": int(np.count_nonzero(y == y.min())),
        },
        "outputs": {},
        "warnings": {},
        "rows": [],
    }
    start = int(np.lexsort(tuple(x[:, col] for col in reversed(range(case.dimension))))[0])
    for family, suffixes in FAMILIES.items():
        names = tuple(f"{family}.{suffix}" for suffix in suffixes)
        settings = dict(REFERENCE_SETTINGS[family])
        if family == "ic":
            settings.update(
                ic_nn_start=start,
                seed=case.seed,
                ic_epsilon=np.insert(10.0 ** np.linspace(-5.0, 15.0, 1000), 0, 0.0),
            )
        reference, reference_warnings = collect_family(calculators[family], x, y, settings)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = compute(
                    sample,
                    names,
                    workers=1,
                    y_normalization=None,
                    options={"fitness_distance": {"proportion_of_best": 0.1}},
                )
                current = {
                    name: scalar(item.value, status=item.status.value, message=item.message)
                    for name, item in result.values.items()
                }
                engine_warnings = list(result.metadata.warnings)
            except Exception as exc:
                current = {
                    name: scalar(None, status="error", message=f"{type(exc).__name__}: {exc}")
                    for name in names
                }
                engine_warnings = []
        details["outputs"][family] = {"orivex": current, "pflacco": reference}
        details["warnings"][family] = {
            "pflacco": reference_warnings,
            "orivex": engine_warnings
            + [f"{item.category.__name__}: {item.message}" for item in caught],
        }
        for name in names:
            expected = reference.get(
                name, reference.get("__error__", scalar(None, status="missing"))
            )
            row = compare(
                name, current.get(name, scalar(None, status="missing")), expected, atol, rtol
            )
            row.update(asdict(case), case_id=case.key, function_name=problem.meta_data.name)
            if name == QUADRATIC:
                try:
                    oracle = scalar(quadratic_oracle(x, y))
                except Exception as exc:
                    oracle = scalar(None, status="error", message=f"{type(exc).__name__}: {exc}")
                row.update(
                    oracle_value=oracle["value"],
                    oracle_status=oracle["status"],
                    oracle_message=oracle["message"],
                    oracle_absolute_error=None,
                    oracle_relative_error=None,
                    oracle_within_tolerance=None,
                )
                if oracle["status"] == "ok" and row["orivex_status"] == "ok":
                    metrics = error_metrics(row["orivex_value"], oracle["value"], atol, rtol)
                    for key in ("absolute_error", "relative_error", "within_tolerance"):
                        row[f"oracle_{key}"] = metrics[key]
                    if row["comparison"] == "definition_difference" and metrics["within_tolerance"]:
                        row["verdict"] = "pass"
            details["rows"].append(row)
    return details


def run_case(task: tuple) -> dict:
    case, root, output, atol, rtol = task
    try:
        with threadpool_limits(limits=1):
            return evaluate_case(case, load_reference(root), output, atol, rtol)
    except Exception as exc:
        failure = scalar(None, status="error", message=f"{type(exc).__name__}: {exc}")
        rows = []
        for name in FEATURES:
            row = compare(name, failure, failure, atol, rtol)
            row.update(asdict(case), case_id=case.key, function_name=None)
            rows.append(row)
        return {
            "case": asdict(case),
            "case_id": case.key,
            "case_error": failure["message"],
            "rows": rows,
        }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for feature in FEATURES:
        for dimension in [None, *sorted({row["dimension"] for row in rows})]:
            selected = [
                row
                for row in rows
                if row["feature"] == feature
                and (dimension is None or row["dimension"] == dimension)
            ]
            summary = {
                "feature": feature,
                "dimension": dimension,
                "cases": len(selected),
                "counts": dict(Counter(row["comparison"] for row in selected)),
                "review_required": sum(row["verdict"] != "pass" for row in selected),
                "exact_matches": sum(row["absolute_error"] == 0 for row in selected),
                "relative_undefined": sum(row["reference_zero"] is True for row in selected),
                "near_zero_reference": sum(row["reference_near_zero"] is True for row in selected),
                "error_overflows": sum(row["error_overflow"] for row in selected),
            }
            for metric in ("absolute_error", "relative_error", "tolerance_ratio"):
                finite = [row for row in selected if row[metric] is not None]
                values = [row[metric] for row in finite]
                summary[metric] = {
                    "count": len(values),
                    "median": float(np.median(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                    "max": max(values) if values else None,
                    "worst_case": max(finite, key=lambda row: row[metric])["case_id"]
                    if finite
                    else None,
                }
            summaries.append(summary)
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pflacco-root", type=Path, default=Path("../pflacco"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/bbob-correctness"))
    parser.add_argument("--functions", nargs="+", type=int, default=list(range(1, 25)))
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--instances", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample-multiplier", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent processes; -1 uses CPU count"
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-only", action="store_true", help="Regenerate plots from output/report.json"
    )
    args = parser.parse_args(argv)
    if args.plot_only:
        return args
    for name in ("functions", "dimensions", "instances", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} must not contain duplicates")
    if any(not 1 <= f <= 24 for f in args.functions):
        parser.error("BBOB function IDs must be between 1 and 24")
    if min(args.dimensions) < 2 or min(args.instances) < 1:
        parser.error("dimensions must be >= 2 and instances >= 1")
    if any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be integers in [0, 2**32)")
    if args.sample_multiplier < 1:
        parser.error("sample multiplier must be positive")
    if args.workers == 0 or args.workers < -1:
        parser.error("workers must be -1 or positive")
    if not math.isfinite(args.atol) or args.atol <= 0:
        parser.error("atol must be finite and positive")
    if not math.isfinite(args.rtol) or args.rtol < 0:
        parser.error("rtol must be finite and nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    from correctness_report import write_report

    args = parse_args(argv)
    if args.plot_only:
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        write_report(args.output, report, plots=True)
        return 0
    specs = {spec.name: spec for spec in list_features()}
    if set(specs) != set(FEATURES):
        raise RuntimeError(
            "Feature registry changed; update the explicit pflacco comparison mapping"
        )
    # Fail before creating an output directory if the reference/dependencies cannot load.
    load_reference(args.pflacco_root)
    import ioh  # noqa: F401

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases").mkdir()
    root = Path(__file__).resolve().parents[1]
    cases = [
        Case(f, d, i, s, args.sample_multiplier * d)
        for f in args.functions
        for d in args.dimensions
        for i in args.instances
        for s in args.seeds
    ]
    workers = min(len(cases), (os.cpu_count() or 1) if args.workers == -1 else args.workers)
    manifest = {
        "schema_version": 1,
        "complete": False,
        "expected_cases": len(cases),
        "settings": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "workers_effective": workers,
            "numerical_threads_per_worker": 1,
            "backend": "numpy",
            "dtype": "float64",
            "y_normalization": None,
            "feature_normalization": None,
            "sampling": "uniform / PCG64",
            "seed_sequence": "[seed, function, instance, dimension, observations]",
            "pflacco_parameters": REFERENCE_SETTINGS,
            "ic_start": "lexicographically smallest observation",
            "ic_epsilon": "insert(10 ** linspace(-5, 15, 1000), 0, 0)",
            "orivex_options": {"fitness_distance": {"proportion_of_best": 0.1}},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "ioh",
                    "matplotlib",
                    "numdifftools",
                    "SALib",
                    "pyDOE",
                    "threadpoolctl",
                )
            },
            "threadpools_parent": threadpool_info(),
        },
        "source": {
            "orivex_revision": git_revision(root),
            "orivex_sha256": source_hashes(root / "src" / "orivex"),
            "benchmark_sha256": source_hashes(root / "benchmarks"),
            "pflacco_root": str(args.pflacco_root.resolve()),
            "pflacco_revision": git_revision(args.pflacco_root.resolve()),
            "pflacco_sha256": source_hashes(args.pflacco_root.resolve() / "pflacco"),
        },
        "coverage": [
            {
                "feature": name,
                "pflacco_feature": name,
                "definition": specs[name].definition,
                "comparison": "different_definition_with_independent_oracle"
                if name == QUADRATIC
                else "same_definition",
            }
            for name in FEATURES
        ],
        "scope": "All implemented NumPy features and outputs of their five pflacco families; "
        "other pflacco families and Torch are outside this run. No timing measurements.",
    }
    write_json(args.output / "manifest.json", manifest)
    tasks = [
        (case, args.pflacco_root.resolve(), args.output.resolve(), args.atol, args.rtol)
        for case in cases
    ]
    rows = []
    extra_outputs = set()

    def consume(results):
        for index, result in enumerate(results, 1):
            write_json(args.output / "cases" / f"{result['case_id']}.json", result)
            rows.extend(result["rows"])
            for family in result.get("outputs", {}).values():
                extra_outputs.update(set(family["pflacco"]) - set(FEATURES) - {"__error__"})
            reviews = sum(row["verdict"] != "pass" for row in result["rows"])
            print(
                f"[{index}/{len(cases)}] {result['case_id']}: {reviews} feature(s) require review",
                flush=True,
            )

    if workers == 1:
        consume(map(run_case, tasks))
    else:
        # Set before spawned interpreters import numerical libraries; avoid nested pools.
        thread_env = (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        previous = {name: os.environ.get(name) for name in thread_env}
        try:
            for name in thread_env:
                os.environ[name] = "1"
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                consume(pool.map(run_case, tasks))
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    manifest.update(
        complete=True, completed_cases=len(cases), pflacco_only_outputs=sorted(extra_outputs)
    )
    review_count = sum(row["verdict"] != "pass" for row in rows)
    report = {
        "manifest": manifest,
        "rows": rows,
        "summary": summarize(rows),
        "review_required": review_count,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "report.json", report)
    write_report(args.output, report, plots=not args.no_plots)
    print(f"Saved {len(rows)} comparisons to {args.output}; {review_count} require review.")
    return 1 if review_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
