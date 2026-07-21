"""Unit tests for marketlab.tokenizer."""

import numpy as np
import pytest

from marketlab.tokenizer import ReturnTokenizer


@pytest.fixture
def fitted():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.02, 10_000)
    return ReturnTokenizer(n_bins=64).fit(returns), returns


def test_edges_monotonic_and_shapes(fitted):
    tok, _ = fitted
    assert tok.edges.shape == (63,)
    assert (np.diff(tok.edges) >= 0).all()
    assert tok.bin_values.shape == (64,)
    assert tok.bin_probs.shape == (64,)
    assert tok.bin_probs.sum() == pytest.approx(1.0)


def test_round_trip_is_tight(fitted):
    tok, returns = fitted
    decoded = tok.decode(tok.encode(returns))
    # Decoded value must be a faithful stand-in for the original return.
    assert np.corrcoef(returns, decoded)[0, 1] > 0.99
    # And each decoded value stays inside its bin's edges (spot-check interior).
    tokens = tok.encode(returns)
    interior = (tokens > 0) & (tokens < 63)
    lo = tok.edges[tokens[interior] - 1]
    hi = tok.edges[tokens[interior]]
    dec = decoded[interior]
    assert ((dec >= lo) & (dec <= hi)).all()


def test_quantile_bins_roughly_uniform(fitted):
    tok, returns = fitted
    counts = np.bincount(tok.encode(returns), minlength=64)
    assert counts.min() > 0.5 * counts.mean()  # no starved bins on the fit data


def test_extremes_clamp_to_end_bins(fitted):
    tok, _ = fitted
    assert tok.encode(np.array([-10.0]))[0] == 0
    assert tok.encode(np.array([+10.0]))[0] == 63


def test_json_round_trip(fitted):
    tok, returns = fitted
    clone = ReturnTokenizer.from_json(tok.to_json())
    np.testing.assert_array_equal(clone.encode(returns), tok.encode(returns))
    np.testing.assert_allclose(clone.bin_values, tok.bin_values)


def test_tied_data_produces_no_nans():
    # Heavy ties collapse quantile edges and leave empty bins; decode values
    # must still be finite everywhere.
    data = np.concatenate([np.zeros(500), np.random.default_rng(0).normal(0, 0.01, 50)])
    tok = ReturnTokenizer(n_bins=16).fit(data)
    assert np.isfinite(tok.bin_values).all()
    assert np.isfinite(tok.decode(np.arange(16))).all()


def test_unfitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ReturnTokenizer().encode(np.array([0.01]))
    with pytest.raises(ValueError, match="at least"):
        ReturnTokenizer(n_bins=64).fit(np.array([0.01, 0.02]))
