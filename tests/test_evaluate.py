"""Unit tests for marketlab.evaluate against series with known structure."""

import numpy as np

from marketlab.evaluate import (
    acf,
    excess_kurtosis,
    make_figure,
    mean_path_acf,
    stylized_facts,
)


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _vol_regime_returns(n: int, seed: int = 0) -> np.ndarray:
    """iid signs but regime-switching vol -> zero return-ACF, positive |r|-ACF."""
    rng = np.random.default_rng(seed)
    sigma = np.where((np.arange(n) // 50) % 2 == 0, 0.01, 0.04)
    return rng.normal(0, 1, n) * sigma


def test_acf_matches_ar1_theory():
    x = _ar1(0.6, 50_000)
    est = acf(x, 5)
    for k in range(5):
        assert abs(est[k] - 0.6 ** (k + 1)) < 0.03


def test_acf_of_iid_is_near_zero():
    rng = np.random.default_rng(3)
    est = acf(rng.normal(0, 1, 50_000), 20)
    assert np.abs(est).max() < 0.03


def test_vol_clustering_metric_separates_regimes_from_iid():
    clustered = _vol_regime_returns(20_000)
    iid = np.random.default_rng(4).normal(0, 0.02, 20_000)
    assert np.mean(acf(np.abs(clustered), 20)) > 0.1
    assert abs(np.mean(acf(np.abs(iid), 20))) < 0.02


def test_mean_path_acf_stays_within_paths():
    # Two paths whose concatenation would fake autocorrelation: per-path ACF ~ 0.
    rng = np.random.default_rng(5)
    paths = np.stack([rng.normal(5, 1, 2000), rng.normal(-5, 1, 2000)])
    est = mean_path_acf(paths, 10)
    assert np.abs(est).max() < 0.06


def test_excess_kurtosis():
    rng = np.random.default_rng(6)
    assert abs(excess_kurtosis(rng.normal(0, 1, 200_000))) < 0.1
    assert excess_kurtosis(rng.standard_t(4, 200_000)) > 1.0  # fat tails


def test_stylized_facts_keys_and_figure(tmp_path):
    rng = np.random.default_rng(7)
    real = _vol_regime_returns(3000, seed=8)
    synth = rng.normal(0, 0.02, (20, 500))
    metrics = stylized_facts(real, synth)
    assert set(metrics) == {
        "real_ret_acf_mean_abs", "synth_ret_acf_mean_abs",
        "real_vol_clustering", "synth_vol_clustering",
        "real_excess_kurtosis", "synth_excess_kurtosis",
        "real_daily_vol", "synth_daily_vol",
    }
    out = tmp_path / "fig.png"
    make_figure(real, synth, str(out), "TEST")
    assert out.exists() and out.stat().st_size > 10_000
