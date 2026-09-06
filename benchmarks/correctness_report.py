"""Render saved correctness results; plotting never recomputes landscape features."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict], *, fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, report: dict, *, plots: bool) -> None:
    rows = report["rows"]
    write_csv(output / "comparisons.csv", rows)
    flattened = []
    for summary in report["summary"]:
        item = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                item.update({f"{key}.{field}": entry for field, entry in value.items()})
            else:
                item[key] = value
        flattened.append(item)
    write_csv(output / "summary.csv", flattened)
    write_csv(
        output / "review.csv",
        [row for row in rows if row["verdict"] != "pass"],
        fields=list(dict.fromkeys(key for row in rows for key in row)),
    )
    lines = [
        "# BBOB correctness report",
        "",
        f"Datasets: {report['manifest']['completed_cases']}. Feature comparisons: {len(rows)}. "
        f"Requiring review: **{report['review_required']}**.",
        "",
        "Raw float64 objectives and feature values; NumPy backend. No timing measurements.",
        "",
        "The quadratic interaction feature has a different definition in pflacco. Its raw "
        "discrepancies are retained; its orivex result is checked against an independent "
        "complete-degree-two sklearn fit. Undefined values never count as numerical passes.",
        "",
        "| Feature | Cases | Review | Median abs. error | Maximum abs. error | Worst case |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["summary"]:
        if item["dimension"] is not None:
            continue
        metric = item["absolute_error"]
        median = "n/a" if metric["median"] is None else f"{metric['median']:.3g}"
        maximum = "n/a" if metric["max"] is None else f"{metric['max']:.3g}"
        worst = metric["worst_case"]
        link = f"[{worst}](cases/{worst}.json)" if worst else "n/a"
        lines.append(
            f"| {item['feature']} | {item['cases']} | {item['review_required']} "
            f"| {median} | {maximum} | {link} |"
        )
    lines += [
        "",
        "Full values and status messages: `comparisons.csv`. Cases requiring review: "
        "`review.csv`. Dimension-specific summaries: `summary.csv`. Exact samples: "
        "`cases/*.npz`. Configuration and source hashes: `manifest.json`.",
        "",
    ]
    if plots:
        plot_errors(output, report)
        lines += [
            "## Error distributions",
            "",
            "Each ECDF aggregates functions, instances, and sampling seeds within one dimension. "
            "The legend counts finite errors, exact zeros, and excluded entries. Exact zeros "
            "contribute to cumulative probability but cannot appear on a logarithmic x-axis. "
            "Relative error is undefined at reference zero; no epsilon is substituted.",
            "",
        ]
        for name in sorted({row["feature"] for row in rows}):
            lines += [f"![{name}](plots/{name}.png)", ""]
        lines += [
            "## Worst discrepancy by function",
            "",
            "Cells show log10(1 + maximum tolerance ratio) over seeds and instances. "
            "A ratio above 1 exceeds tolerance; grey cells have no finite comparison. "
            "The quadratic row (*) compares different definitions and is diagnostic only. "
            "Missing and undefined observations remain visible in the tables.",
            "",
        ]
        for dimension in sorted({row["dimension"] for row in rows}):
            lines += [f"![Dimension {dimension}](plots/heatmap_d{dimension}.png)", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_errors(output: Path, report: dict) -> None:
    # Matplotlib's default user cache may be outside a writable checkout.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "orivex-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    rows = report["rows"]
    features = sorted({row["feature"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    for feature in features:
        selected = [row for row in rows if row["feature"] == feature]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        different = any(row["definition_difference"] for row in selected)
        title = feature + (" — different pflacco definition" if different else "")
        fig.suptitle(title)
        for axis, metric, label in zip(
            axes,
            ("absolute_error", "relative_error"),
            ("Absolute error", "Relative error"),
            strict=True,
        ):
            positive_seen = False
            for dimension in dimensions:
                group = [row for row in selected if row["dimension"] == dimension]
                values = np.sort([row[metric] for row in group if row[metric] is not None])
                zeros = int(np.count_nonzero(values == 0))
                legend = (
                    f"d={dimension}: finite={len(values)}, zero={zeros}, "
                    f"excluded={len(group) - len(values)}"
                )
                positive = values > 0
                if np.any(positive):
                    positive_seen = True
                    cumulative = np.arange(1, len(values) + 1) / len(values)
                    axis.step(
                        values[positive],
                        cumulative[positive],
                        where="post",
                        label=legend,
                        marker=".",
                        markersize=3,
                    )
                else:
                    axis.plot([], [], label=legend)
            axis.set(
                xscale="log",
                xlabel=label,
                ylabel="Cumulative fraction of finite errors",
                ylim=(0, 1.03),
            )
            if not positive_seen:
                axis.set_xlim(1e-16, 1)
                axis.text(
                    0.5, 0.5, "No positive finite errors", ha="center", transform=axis.transAxes
                )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="lower right")
        fig.savefig(plot_dir / f"{feature}.png", dpi=160)
        fig.savefig(plot_dir / f"{feature}.pdf")
        plt.close(fig)
    functions = sorted({row["function"] for row in rows})
    for dimension in dimensions:
        matrix = np.full((len(features), len(functions)), np.nan)
        for i, feature in enumerate(features):
            for j, function in enumerate(functions):
                values = [
                    row["tolerance_ratio"]
                    for row in rows
                    if row["dimension"] == dimension
                    and row["feature"] == feature
                    and row["function"] == function
                    and row["tolerance_ratio"] is not None
                ]
                if values:
                    matrix[i, j] = np.log10(1 + max(values))
        fig, axis = plt.subplots(figsize=(max(9, len(functions) * 0.42), 9), layout="constrained")
        palette = plt.get_cmap("magma").with_extremes(bad="lightgrey")
        finite = matrix[np.isfinite(matrix)]
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0
        heatmap = axis.imshow(matrix, aspect="auto", cmap=palette, vmin=0, vmax=vmax)
        axis.set_xticks(range(len(functions)), labels=[f"f{f}" for f in functions])
        axis.set_yticks(
            range(len(features)),
            labels=[
                name + (" *" if name == "ela_meta.quad_w_interact.adj_r2" else "")
                for name in features
            ],
            fontsize=8,
        )
        axis.set_title(f"Worst discrepancy across seeds and instances — d={dimension}")
        fig.colorbar(heatmap, ax=axis, label="log10(1 + error / tolerance); threshold = 0.301")
        fig.savefig(plot_dir / f"heatmap_d{dimension}.png", dpi=160)
        fig.savefig(plot_dir / f"heatmap_d{dimension}.pdf")
        plt.close(fig)
