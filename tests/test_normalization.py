import numpy as np
import pytest
from scipy.stats import rankdata

from orivex import LandscapeSample, compute
from orivex.normalization import normalize_objectives


def sample(y=None, *, sense="minimize"):
    x = np.random.default_rng(17).uniform(-4, 4, (80, 2))
    if y is None:
        y = 17 + np.sin(x[:, 0]) + x[:, 1] ** 2
    return LandscapeSample(x, y, [-5, -5], [5, 5], sense=sense)


def assert_same(left, right):
    for name, item in left.values.items():
        expected = right.values[name]
        assert item.status == expected.status
        if item.value is not None:
            assert item.value == pytest.approx(expected.value, rel=2e-11, abs=2e-12)


def test_preprocessing_matches_manual_raw_pipeline_for_every_feature():
    original = sample()
    y = original.y
    transformed = (y - y.min()) / np.ptp(y)
    expected = compute(sample(transformed), "*", y_normalization=None)
    result = compute(original, "*", y_normalization="minmax")
    assert_same(result, expected)
    np.testing.assert_array_equal(original.y, y)


def test_all_features_inherit_affine_invariance_and_sense_reversal():
    original = sample()
    reference = compute(original, "*", y_normalization="minmax")
    assert_same(reference, compute(sample(8 * original.y - 123), "*", y_normalization="minmax"))
    assert_same(
        reference, compute(sample(-original.y, sense="maximize"), "*", y_normalization="minmax")
    )


def test_default_preserves_raw_sample_and_records_distinct_preprocessing_identity():
    original = sample()
    before = original.y.copy()
    fingerprint = original.fingerprint
    default = compute(original, "ela_meta.lin_simple.intercept")
    explicit = compute(original, "ela_meta.lin_simple.intercept", y_normalization="minmax")
    raw = compute(original, "ela_meta.lin_simple.intercept", y_normalization=None)
    assert_same(default, explicit)
    assert default.values != raw.values
    assert default.metadata.y_normalization == "minmax"
    assert raw.metadata.y_normalization is None
    assert raw.metadata.y_normalization_definition == "objective-none-v1"
    assert default.metadata.y_normalization_definition == "objective-minmax-v1"
    assert default.metadata.preprocessing_fingerprint == explicit.metadata.preprocessing_fingerprint
    assert default.metadata.preprocessing_fingerprint != raw.metadata.preprocessing_fingerprint
    assert default.metadata.sample_fingerprint == raw.metadata.sample_fingerprint == fingerprint
    np.testing.assert_array_equal(original.y, before)


@pytest.mark.parametrize("mode", [None, "minmax"])
def test_constant_objectives_keep_feature_specific_statuses(mode):
    result = compute(sample(np.full(80, 23.0)), ["ela_distr.*", "ela_meta.*"], y_normalization=mode)
    assert result.metadata.constant_objective
    assert result.values["ela_distr.skewness"].status.value == "invalid"
    assert result.values["ela_meta.lin_simple.adj_r2"].status.value == "invalid"
    expected = 23 if mode is None else 0
    assert result.values["ela_meta.lin_simple.intercept"].value == pytest.approx(expected)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_extreme_finite_scales_preserve_distribution_features(scale):
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])
    x = np.arange(len(y))[:, None]
    a = LandscapeSample(x, y, [-1], [7])
    b = LandscapeSample(x, scale * y, [-1], [7])
    assert_same(
        compute(a, "ela_distr.*", y_normalization="minmax"),
        compute(b, "ela_distr.*", y_normalization="minmax"),
    )


def test_overflowing_range_and_close_large_values():
    y = np.array([-1.7e308, -8.5e307, 0, 8.5e307, 1.7e308])
    actual, constant = normalize_objectives(y, "minmax")
    expected, _ = normalize_objectives(np.arange(5.0), "minmax")
    assert not constant
    np.testing.assert_allclose(actual, expected)
    assert not actual.flags.writeable
    close = np.array([1e100, np.nextafter(1e100, np.inf)])
    actual, constant = normalize_objectives(close, "minmax")
    assert not constant
    np.testing.assert_allclose(actual, [0, 1])


@pytest.mark.parametrize("mode", ["zscore", "none", "typo", False, 0])
def test_invalid_normalization_is_rejected(mode):
    with pytest.raises(ValueError, match="y_normalization"):
        compute(sample(), "ela_distr.*", y_normalization=mode)


def test_normalization_is_shared_once_per_request(monkeypatch):
    from orivex import engine

    calls = []
    original = engine.normalize_objectives

    def count(y, mode):
        calls.append(mode)
        return original(y, mode)

    monkeypatch.setattr(engine, "normalize_objectives", count)
    compute(sample(), "*")
    assert calls == ["minmax"]


INCREASING_TRANSFORMS = {
    "affine": lambda y: 8 * y - 123,
    "exp": lambda y: np.exp(y / 5) ** 3 + np.arctan(y),
    "cubic": lambda y: (y - 18) ** 3 + 0.1 * y,
    "log": lambda y: np.log1p(y - y.min()),
    "piecewise": lambda y: np.where(y < 18, y, 1e3 * (y - 18) + 18),
}


def test_rank_preprocessing_uses_normalized_average_ranks():
    y = np.array([4.0, -1.0, 4.0, 9.0, 4.0, 2.0])
    values, constant = normalize_objectives(y, "rank")
    assert not constant
    np.testing.assert_array_equal(values, (rankdata(y, method="average") - 1) / (y.size - 1))
    np.testing.assert_array_equal(values, [0.6, 0.0, 0.6, 1.0, 0.6, 0.2])
    assert not values.flags.writeable
    values, constant = normalize_objectives(np.full(5, 3.0), "rank")
    assert constant
    np.testing.assert_array_equal(values, np.zeros(5))


def test_rank_preprocessing_matches_manual_raw_pipeline_for_every_feature():
    original = sample()
    transformed = (rankdata(original.y) - 1) / (original.y.size - 1)
    rng = np.random.default_rng
    expected = compute(sample(transformed), "*", rng=rng(5), y_normalization=None)
    result = compute(original, "*", rng=rng(5), y_normalization="rank")
    assert_same(result, expected)
    assert result.metadata.y_normalization == "rank"
    assert result.metadata.y_normalization_definition == "objective-rank-v1"


@pytest.mark.parametrize("name", sorted(INCREASING_TRANSFORMS))
def test_every_feature_is_invariant_to_increasing_transforms_under_rank(name):
    original = sample()
    transformed = INCREASING_TRANSFORMS[name](original.y)
    # Precondition: the transform preserves the order of the floating-point values.
    np.testing.assert_array_equal(rankdata(transformed), rankdata(original.y))
    rng = np.random.default_rng
    reference = compute(original, "*", rng=rng(9), y_normalization="rank")
    result = compute(sample(transformed), "*", rng=rng(9), y_normalization="rank")
    assert all(item.status.value == "ok" for item in reference.values.values())
    assert_same(reference, result)
    assert result.metadata.preprocessing_fingerprint != reference.metadata.preprocessing_fingerprint


def test_rank_degenerate_features_depend_only_on_sample_size_without_ties():
    x = np.random.default_rng(17).uniform(-4, 4, (80, 2))
    features = [
        "ela_distr.skewness",
        "ela_distr.kurtosis",
        "fitness_distance.fitness_mean",
        "fitness_distance.fitness_std",
    ]
    first = compute(sample(17 + np.sin(x[:, 0]) + x[:, 1] ** 2), features, y_normalization="rank")
    second = compute(sample(np.abs(x[:, 0] - x[:, 1])), features, y_normalization="rank")
    assert first.values["ela_distr.skewness"].value == pytest.approx(0, abs=1e-12)
    assert_same(first, second)
    plateau = compute(sample(np.round(x[:, 1] ** 2)), features, y_normalization="rank")
    assert plateau.values["ela_distr.skewness"].value != pytest.approx(0, abs=1e-3)


def test_rank_constant_objectives_keep_feature_specific_statuses():
    result = compute(
        sample(np.full(80, 23.0)), ["ela_distr.*", "ela_meta.*"], y_normalization="rank"
    )
    assert result.metadata.constant_objective
    assert result.values["ela_distr.skewness"].status.value == "invalid"
    assert result.values["ela_meta.lin_simple.adj_r2"].status.value == "invalid"
    assert result.values["ela_meta.lin_simple.intercept"].value == pytest.approx(0)


def test_order_only_features_agree_between_minmax_and_rank():
    order_only = {
        "fitness_distance.distance_mean",
        "fitness_distance.distance_std",
        "ic.m0",
        "nbc.dist_ratio.coeff_var",
        "nbc.nn_nb.cor",
        "nbc.nn_nb.mean_ratio",
        "nbc.nn_nb.sd_ratio",
    }
    original = sample()
    minmax = compute(original, "*", rng=np.random.default_rng(3), y_normalization="minmax")
    rank = compute(original, "*", rng=np.random.default_rng(3), y_normalization="rank")
    equal = {
        name
        for name, item in minmax.values.items()
        if item.value == pytest.approx(rank.values[name].value, rel=1e-12, abs=1e-12)
    }
    assert equal == order_only
