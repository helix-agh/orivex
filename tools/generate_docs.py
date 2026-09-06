"""Build the feature catalogue and copy shared branding assets into the docs site."""

from pathlib import Path

import mkdocs_gen_files

from orivex import list_features

ROOT = Path(__file__).resolve().parents[1]


def table_cell(value: str) -> str:
    """Keep metadata containing pipes or line breaks inside a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", "<br>")


with mkdocs_gen_files.open("features/catalogue.md", "w") as page:
    page.write(
        "# Feature catalogue\n\n"
        "This page is generated from the NumPy feature registry on every documentation build. "
        "It records declared semantics and costs; see the "
        "[Torch guide](../user-guide/torch.md) for the supported tensor profile.\n\n"
        "Definition identifiers describe the underlying feature. "
        "[Objective normalization](../user-guide/normalization.md) is a separate preprocessing "
        "step, applied before the feature formula. Minimum counts are necessary conditions; "
        "rank, variation, and selection requirements can impose further restrictions.\n\n"
        "For fitness-distance features the minimum applies to **selected** observations. "
        "Read the [fitness-distance guide](../user-guide/fitness-distance.md) for selection "
        "and estimator conventions.\n\n"
    )
    current_group = None
    for spec in list_features():
        if spec.group != current_group:
            current_group = spec.group
            page.write(f"## {spec.group}\n\n")
        page.write(f"### `{spec.name}`\n\n{spec.summary}\n\n")
        rows = {
            "Definition": f"`{spec.definition}`",
            "Kind": spec.kind.value,
            "Requires": ", ".join(sorted(item.value for item in spec.requirements)),
            "Minimum observations": str(spec.minimum_observations),
            "Cost tier": spec.cost.tier.value,
            "CPU": spec.cost.cpu,
            "Memory": spec.cost.memory,
            "Additional objective evaluations": spec.cost.additional_objective_evaluations,
            "Deterministic": "Yes" if spec.deterministic else "No",
            "Legacy names": ", ".join(f"`{name}`" for name in spec.legacy_names) or "None",
        }
        page.write("| Property | Value |\n| --- | --- |\n")
        for label, value in rows.items():
            page.write(f"| {label} | {table_cell(value)} |\n")
        page.write("\n**Intermediates:** ")
        page.write(", ".join(f"`{name}`" for name in spec.intermediates) or "None")
        page.write(".\n\n")
        if spec.notes:
            page.write("**Conventions and edge cases**\n\n")
            for note in spec.notes:
                page.write(f"- {note}\n")
            page.write("\n")
        if spec.invariances:
            page.write(
                "**Declared transformations**\n\n"
                "| Transformation | Behavior | Conditions | Notes |\n"
                "| --- | --- | --- | --- |\n"
            )
            for claim in spec.invariances:
                cells = (
                    claim.transformation.value,
                    claim.behavior.value,
                    claim.conditions,
                    claim.notes,
                )
                page.write("| " + " | ".join(table_cell(cell) for cell in cells) + " |\n")
            page.write("\n")
        if spec.references:
            page.write("**References**\n\n")
            for reference in spec.references:
                url = f"https://doi.org/{reference.doi}" if reference.doi else reference.url
                page.write(f"- {reference.citation} ([source]({url}))\n")
            page.write("\n")

mkdocs_gen_files.set_edit_path("features/catalogue.md", "../tools/generate_docs.py")

for name in ("orivex-lockup-light.png", "orivex-lockup-dark.png"):
    with mkdocs_gen_files.open(f"assets/{name}", "wb") as asset:
        asset.write((ROOT / "assets" / name).read_bytes())
