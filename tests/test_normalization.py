import numpy as np
import pytest

from bflacco import LandscapeSample, compute
from bflacco.normalization import normalize_objectives


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


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_preprocessing_matches_manual_raw_pipeline_for_every_feature(mode):
    original = sample()
    y = original.y
    transformed = (y - y.min()) / np.ptp(y) if mode == "minmax" else (y - y.mean()) / y.std(ddof=0)
    expected = compute(sample(transformed), "*", y_normalization="none")
    result = compute(original, "*", y_normalization=mode)
    assert_same(result, expected)
    np.testing.assert_array_equal(original.y, y)


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_all_features_inherit_affine_invariance_and_sense_reversal(mode):
    original = sample()
    reference = compute(original, "*", y_normalization=mode)
    assert_same(reference, compute(sample(8 * original.y - 123), "*", y_normalization=mode))
    assert_same(
        reference, compute(sample(-original.y, sense="maximize"), "*", y_normalization=mode)
    )


def test_default_preserves_raw_sample_and_records_distinct_preprocessing_identity():
    original = sample()
    before = original.y.copy()
    fingerprint = original.fingerprint
    default = compute(original, "ela_meta.lin_simple.intercept")
    explicit = compute(original, "ela_meta.lin_simple.intercept", y_normalization="minmax")
    raw = compute(original, "ela_meta.lin_simple.intercept", y_normalization="none")
    assert_same(default, explicit)
    assert default.values != raw.values
    assert default.metadata.y_normalization == "minmax"
    assert default.metadata.y_normalization_definition == "objective-minmax-v1"
    assert default.metadata.preprocessing_fingerprint == explicit.metadata.preprocessing_fingerprint
    assert default.metadata.preprocessing_fingerprint != raw.metadata.preprocessing_fingerprint
    assert default.metadata.sample_fingerprint == raw.metadata.sample_fingerprint == fingerprint
    np.testing.assert_array_equal(original.y, before)


@pytest.mark.parametrize("mode", ["none", "minmax", "zscore"])
def test_constant_objectives_keep_feature_specific_statuses(mode):
    result = compute(sample(np.full(80, 23.0)), ["ela_distr.*", "ela_meta.*"], y_normalization=mode)
    assert result.metadata.constant_objective
    assert result.values["ela_distr.skewness"].status.value == "invalid"
    assert result.values["ela_meta.lin_simple.adj_r2"].status.value == "invalid"
    expected = 23 if mode == "none" else 0
    assert result.values["ela_meta.lin_simple.intercept"].value == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_extreme_finite_scales_preserve_distribution_features(mode, scale):
    y = np.array([-4.0, -1.0, 0.0, 2.0, 8.0, 9.0])
    x = np.arange(len(y))[:, None]
    a = LandscapeSample(x, y, [-1], [7])
    b = LandscapeSample(x, scale * y, [-1], [7])
    assert_same(
        compute(a, "ela_distr.*", y_normalization=mode),
        compute(b, "ela_distr.*", y_normalization=mode),
    )


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_overflowing_range_and_close_large_values(mode):
    y = np.array([-1.7e308, -8.5e307, 0, 8.5e307, 1.7e308])
    actual, constant = normalize_objectives(y, mode)
    expected, _ = normalize_objectives(np.arange(5.0), mode)
    assert not constant
    np.testing.assert_allclose(actual, expected)
    assert not actual.flags.writeable
    close = np.array([1e100, np.nextafter(1e100, np.inf)])
    actual, constant = normalize_objectives(close, mode)
    assert not constant
    np.testing.assert_allclose(actual, [0, 1] if mode == "minmax" else [-1, 1])


def test_invalid_normalization_is_rejected():
    with pytest.raises(ValueError, match="y_normalization"):
        compute(sample(), "ela_distr.*", y_normalization="typo")


def test_normalization_is_shared_once_per_request(monkeypatch):
    from bflacco import engine

    calls = []
    original = engine.normalize_objectives

    def count(y, mode):
        calls.append(mode)
        return original(y, mode)

    monkeypatch.setattr(engine, "normalize_objectives", count)
    compute(sample(), "*")
    assert calls == ["minmax"]
