from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DifferentialFixture:
    schema_version: int
    case: str
    input_file: str
    input_sha256: str
    r_version: str
    flacco_version: str
    feature_sets: tuple[str, ...]
    feature_parameters: dict[str, Any]
    values: dict[str, float | int | None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_differential_fixture(path: Path) -> DifferentialFixture:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    fixture = DifferentialFixture(
        schema_version=int(raw["schema_version"]),
        case=str(raw["case"]),
        input_file=str(raw["input_file"]),
        input_sha256=str(raw["input_sha256"]),
        r_version=str(raw["r_version"]),
        flacco_version=str(raw["flacco_version"]),
        feature_sets=tuple(raw["feature_sets"]),
        feature_parameters=dict(raw["feature_parameters"]),
        values=dict(raw["values"]),
    )
    if fixture.schema_version != 1:
        raise ValueError(f"unsupported differential-fixture schema {fixture.schema_version}")
    return fixture
