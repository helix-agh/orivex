import json
from pathlib import Path

import pytest

from verification.differential import load_differential_fixture, sha256_file

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "r_flacco"


def expected_fixture_paths() -> list[Path]:
    return sorted((FIXTURE_ROOT / "expected").glob("*.json"))


def test_manifest_inputs_match_recorded_checksums() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))

    for case in manifest["cases"]:
        assert sha256_file(FIXTURE_ROOT / case["file"]) == case["sha256"]


def test_r_fixture_set_is_present() -> None:
    assert [path.name for path in expected_fixture_paths()] == ["linear_d2.json", "sphere_d2.json"]


@pytest.mark.parametrize("fixture_path", expected_fixture_paths(), ids=lambda path: path.stem)
def test_r_fixture_metadata_and_input_checksum(fixture_path: Path) -> None:
    fixture = load_differential_fixture(fixture_path)

    assert fixture.flacco_version
    assert fixture.r_version
    assert fixture.feature_sets == ("ela_distr", "ela_meta", "disp", "nbc", "ic", "pca")
    assert fixture.feature_parameters["ic"]["nn_start"] == "lexicographically_smallest_x"
    assert fixture.feature_parameters["nbc"]["distance_tie_breaker"] == "first"
    assert sha256_file(FIXTURE_ROOT / fixture.input_file) == fixture.input_sha256
    assert fixture.values
