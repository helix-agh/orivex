import numpy as np
import pytest

from orivex import LandscapeSample, compute
from orivex.result import ExecutionMetadata


def test_options_are_copied_without_affecting_other_groups():
    sample = LandscapeSample(np.arange(20.0)[:, None], np.arange(20.0) ** 2, [0], [20])
    options = {"fitness_distance": {"proportion_of_best": 0.5}}
    result = compute(sample, ["fitness_distance.*", "ela_distr.skewness"], options=options)
    assert options == {"fitness_distance": {"proportion_of_best": 0.5}}
    assert result.metadata.options == options
    reference = compute(sample, "ela_distr.skewness")
    assert result.values["ela_distr.skewness"] == reference.values["ela_distr.skewness"]
    options["fitness_distance"]["proportion_of_best"] = 1.0
    assert result.metadata.options["fitness_distance"]["proportion_of_best"] == 0.5
    with pytest.raises(TypeError):
        result.metadata.options["fitness_distance"]["proportion_of_best"] = 0.2
    with pytest.raises(TypeError):
        result.metadata.options["fitness_distance"] = {}


@pytest.mark.parametrize("options", [None, {}, {"fitness_distance": {}}])
def test_omitted_values_use_defaults_without_mutating_input(options):
    sample = LandscapeSample(np.arange(20.0)[:, None], np.arange(20.0) ** 2, [0], [20])
    result = compute(sample, "fitness_distance.*", options=options)
    assert result.metadata.options == {"fitness_distance": {"proportion_of_best": 0.1}}
    assert result.values == compute(sample, "fitness_distance.*").values
    assert options is None or "proportion_of_best" not in options.get("fitness_distance", {})


@pytest.mark.parametrize(
    "options, error, message",
    [
        (0.5, TypeError, "options must be a mapping"),
        ([], TypeError, "options must be a mapping"),
        ({"fitness_distance": 0.5}, TypeError, "must be a mapping"),
        ({"fitness_distance": None}, TypeError, "must be a mapping"),
        ({"fitness_distnace": {}}, ValueError, "unknown options group"),
        ({"proportion_of_best": 0.5}, ValueError, "unknown options group"),
        ({"fitness_distance": {"proportion_of_bset": 0.5}}, ValueError, "unknown option"),
        ({"fitness_distance": {"f_opt": 0}}, ValueError, "unknown option"),
        ({"fitness_distance": {"minkowski_p": 2}}, ValueError, "unknown option"),
    ],
)
def test_options_reject_unknown_names_and_invalid_shapes(options, error, message):
    sample = LandscapeSample([[0], [1]], [0, 1], [0], [1])
    with pytest.raises(error, match=message):
        compute(sample, "fitness_distance.*", options=options)


@pytest.mark.parametrize(
    "name, value",
    [("proportion_of_best", 0.5), ("f_opt", 0), ("minkowski_p", 2), ("fitness_distance", {})],
)
def test_removed_keywords_are_not_accepted(name, value):
    sample = LandscapeSample([[0], [1]], [0, 1], [0], [1])
    with pytest.raises(TypeError, match="unexpected keyword"):
        compute(sample, "fitness_distance.*", **{name: value})


def test_metadata_validates_and_copies_options():
    options = {"fitness_distance": {"proportion_of_best": 0.25}}
    metadata = ExecutionMetadata("abc", (), (), 0, 0, options=options)
    options["fitness_distance"].clear()
    assert metadata.options["fitness_distance"]["proportion_of_best"] == 0.25
    with pytest.raises(ValueError, match="unknown option"):
        ExecutionMetadata("abc", (), (), 0, 0, options={"fitness_distance": {"typo": 1}})
