"""Unit tests for marketlab.sample (generation) and artifact round-trip."""

import numpy as np
import pytest
import torch

from marketlab.model import MarketGPT, ModelConfig
from marketlab.sample import generate, load_artifacts, to_price_paths
from marketlab.tokenizer import ReturnTokenizer

CFG = ModelConfig(
    vocab_size=16, block_size=32, n_layer=1, n_head=2, n_embd=32, n_tickers=2, dropout=0.0
)


@pytest.fixture
def model_and_tokenizer():
    torch.manual_seed(0)
    model = MarketGPT(CFG).eval()
    rng = np.random.default_rng(1)
    tokenizer = ReturnTokenizer(n_bins=16).fit(rng.normal(0.0, 0.02, 5_000))
    return model, tokenizer


def test_shapes_and_finiteness(model_and_tokenizer):
    model, tok = model_and_tokenizer
    out = generate(model, tok, ticker_id=0, n_paths=5, horizon=12, seed=7)
    assert out.shape == (5, 12)
    assert np.isfinite(out).all()
    # Decoded returns can only take bin values.
    assert set(np.round(out.ravel(), 12)) <= set(np.round(tok.bin_values, 12))


def test_seeded_determinism(model_and_tokenizer):
    model, tok = model_and_tokenizer
    a = generate(model, tok, 0, n_paths=4, horizon=10, seed=123)
    b = generate(model, tok, 0, n_paths=4, horizon=10, seed=123)
    np.testing.assert_array_equal(a, b)
    c = generate(model, tok, 0, n_paths=4, horizon=10, seed=124)
    assert not np.array_equal(a, c)


def test_prefix_conditioning_changes_output(model_and_tokenizer):
    model, tok = model_and_tokenizer
    calm = np.full(20, 0.0005)
    wild = np.array([0.05, -0.06] * 10)
    a = generate(model, tok, 0, 4, 10, prefix_returns=calm, seed=5)
    b = generate(model, tok, 0, 4, 10, prefix_returns=wild, seed=5)
    assert a.shape == b.shape == (4, 10)
    assert not np.array_equal(a, b)


def test_invalid_temperature_raises(model_and_tokenizer):
    model, tok = model_and_tokenizer
    with pytest.raises(ValueError, match="temperature"):
        generate(model, tok, 0, 1, 5, temperature=0.0)


def test_to_price_paths():
    rets = np.array([[np.log(2.0), np.log(2.0)], [0.0, 0.0]])
    prices = to_price_paths(rets, start_price=10.0)
    np.testing.assert_allclose(prices, [[20.0, 40.0], [10.0, 10.0]])


def test_artifact_round_trip(tmp_path, model_and_tokenizer, mini_universe):
    """save_artifacts -> load_artifacts -> identical generation."""
    from marketlab.train import build_windows, save_artifacts, train

    data = build_windows(mini_universe, n_bins=16, block_size=32, train_end="2020-09-30")
    cfg = ModelConfig(vocab_size=16, block_size=32, n_layer=1, n_head=2, n_embd=32,
                      n_tickers=len(data.tickers), dropout=0.0)
    model, metrics = train(data, cfg, steps=5, batch_size=16, device="cpu", log_every=0)
    save_artifacts(str(tmp_path), model, data, "2020-09-30", metrics)

    loaded_model, loaded_tok, meta = load_artifacts(str(tmp_path))
    assert meta["tickers"] == data.tickers
    a = generate(model.cpu().eval(), data.tokenizer, 0, 3, 8, seed=9)
    b = generate(loaded_model, loaded_tok, 0, 3, 8, seed=9)
    np.testing.assert_allclose(a, b)
