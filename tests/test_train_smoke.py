"""Training smoke test: a tiny model on the in-memory fixture learns something.

Also the CI-side proof that the whole pipeline (returns -> tokens -> windows ->
train -> eval) runs end-to-end on CPU in well under a minute.
"""

import numpy as np

from marketlab.model import ModelConfig
from marketlab.train import build_windows, log_returns_by_ticker, train


def test_log_returns_by_ticker(mini_universe):
    per = log_returns_by_ticker(mini_universe)
    assert set(per) == {"AAA", "BBB"}
    dates, rets = per["AAA"]
    assert len(dates) == len(rets) == 299  # 300 closes -> 299 returns
    close = mini_universe[mini_universe["Ticker"] == "AAA"]["Close"].to_numpy()
    np.testing.assert_allclose(rets, np.diff(np.log(close)))


def test_build_windows_split_and_shapes(mini_universe):
    data = build_windows(mini_universe, n_bins=16, block_size=32, train_end="2020-09-30")
    assert data.tickers == ["AAA", "BBB"]
    assert data.x_train.shape[1] == data.x_val.shape[1] == 32
    assert len(data.x_train) > 0 and len(data.x_val) > 0
    # y is x shifted by one within the same window.
    assert (data.x_train[:, 1:] == data.y_train[:, :-1]).all()
    assert data.tid_train.max() == 1
    assert np.isfinite(data.baseline_val_ce)
    # With 16 roughly-uniform bins the marginal baseline sits near ln(16).
    assert abs(data.baseline_val_ce - np.log(16)) < 0.5


def test_build_windows_val_end_excludes_later_data(mini_universe):
    """With val_end set, post-val_end returns appear in NEITHER split."""
    full = build_windows(mini_universe, n_bins=16, block_size=32,
                         train_end="2020-09-30")
    capped = build_windows(mini_universe, n_bins=16, block_size=32,
                           train_end="2020-09-30", val_end="2020-12-31")
    assert len(capped.x_train) == len(full.x_train)   # train split unchanged
    assert len(capped.x_val) < len(full.x_val)        # val split truncated
    # Tokenizer fit on train only in both cases -> identical bins.
    np.testing.assert_allclose(capped.tokenizer.edges, full.tokenizer.edges)

    import pytest
    with pytest.raises(ValueError, match="val_end must be after"):
        build_windows(mini_universe, n_bins=16, block_size=32,
                      train_end="2020-09-30", val_end="2020-06-30")


def test_train_reduces_loss(mini_universe):
    data = build_windows(mini_universe, n_bins=16, block_size=32, train_end="2020-09-30")
    cfg = ModelConfig(vocab_size=16, block_size=32, n_layer=1, n_head=2, n_embd=32,
                      n_tickers=len(data.tickers), dropout=0.0)
    model, metrics = train(data, cfg, steps=120, batch_size=32,
                           device="cpu", seed=7, log_every=0)
    assert metrics["final_train_loss"] < metrics["first_train_loss"] - 0.05
    assert np.isfinite(metrics["val_ce"])
