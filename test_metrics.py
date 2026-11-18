import numpy as np
import pytest

from metrics import all_metrics, fid, inception_score, kid


def test_identical_distributions_have_zero_fid():
    features = np.arange(60, dtype=float).reshape(20, 3)
    assert fid(features, features.copy()) == pytest.approx(0.0, abs=1e-10)


def test_metrics_are_finite_and_have_expected_schema():
    rng = np.random.default_rng(4)
    real = rng.normal(size=(24, 4))
    fake = rng.normal(size=(24, 4))
    logits = rng.normal(size=(24, 10))
    results = all_metrics(real, fake, logits)
    assert tuple(results) == (
        "fid", "kid", "inception_score", "precision", "recall", "density", "coverage"
    )
    assert all(np.isfinite(value) for value in results.values())
    assert 1.0 <= inception_score(logits) <= 10.0


def test_kid_is_deterministic():
    rng = np.random.default_rng(9)
    real, fake = rng.normal(size=(16, 5)), rng.normal(size=(16, 5))
    assert kid(real, fake) == kid(real, fake)
