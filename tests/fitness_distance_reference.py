"""Portable outputs from the sibling pflacco implementation; no runtime dependency on it."""

import json
from pathlib import Path

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "pflacco_fitness_distance.json").read_text()
)
