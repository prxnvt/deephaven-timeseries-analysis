"""Unit tests for marketlab.joint (day-as-sentence joint generation)."""

import numpy as np
import pytest
import torch

from marketlab.joint import (
    align_panel,
    build_joint_windows,
    flatten_days,
    mean_pairwise_corr,
    portfolio_var_joint_vs_independent,
    sample_joint,
    stress_split_corr,
)
from marketlab.model import MarketGPT, ModelConfig
from marketlab.tokenizer import ReturnTokenizer
from marketlab.train import train

N_SLOTS = 2
BLOCK_DAYS = 8


@pytest.fixture(scope="module")
def tiny_setup():
    rng = np.random.default_rng(9)
    tokenizer = ReturnTokenizer(n_bins=16).fit(rng.normal(0, 0.02, 5000))
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=16, block_size=BLOCK_DAYS * N_SLOTS, n_layer=1,
                      n_head=2, n_embd=32, n_tickers=N_SLOTS, dropout=0.0)
    return MarketGPT(cfg).eval(), tokenizer


def test_align_panel(mini_universe):
    dates, rets, tickers = align_panel(mini_universe)
    assert tickers == ["AAA", "BBB"]
    assert rets.shape == (299, 2)
    assert len(dates) == 299
    aaa = mini_universe[mini_universe["Ticker"] == "AAA"]["Close"].to_numpy()
    np.testing.assert_allclose(rets[:, 0], np.diff(np.log(aaa)))


def test_model_accepts_per_position_ticker_ids(tiny_setup):
    model, _ = tiny_setup
    idx = torch.randint(0, 16, (3, 10))
    slots = (torch.arange(10) % N_SLOTS).expand(3, -1)
    logits = model(idx, slots)
    assert logits.shape == (3, 10, 16)
    # Per-position slots must genuinely change the computation vs a single id.
    flat = model(idx, torch.zeros(3, dtype=torch.long))
    assert not torch.allclose(logits, flat)


def test_causality_holds_with_2d_ticker_ids(tiny_setup):
    model, _ = tiny_setup
    t = 6
    a = torch.randint(0, 16, (1, 12))
    b = a.clone()
    b[0, t + 1:] = (b[0, t + 1:] + 5) % 16
    slots = (torch.arange(12) % N_SLOTS).expand(1, -1)
    with torch.no_grad():
        la, lb = model(a, slots), model(b, slots)
    assert torch.allclose(la[:, : t + 1], lb[:, : t + 1], atol=1e-5)


def test_build_joint_windows(mini_universe, tiny_setup):
    _, tokenizer = tiny_setup
    data = build_joint_windows(mini_universe, tokenizer, block_days=BLOCK_DAYS,
                               train_end="2020-09-30", val_end="2020-12-31")
    T = BLOCK_DAYS * N_SLOTS
    assert data.x_train.shape[1] == T
    assert data.tid_train.shape == data.x_train.shape  # per-position slots
    # Slot ids cycle 0,1,0,1,... within every window.
    expected_cycle = np.tile(np.arange(N_SLOTS), BLOCK_DAYS)
    assert (data.tid_train.numpy() == expected_cycle).all()
    # y is the flat stream shifted by one token.
    assert (data.x_train[:, 1:] == data.y_train[:, :-1]).all()
    assert len(data.x_val) > 0
    assert np.isfinite(data.baseline_val_ce)


def test_flatten_days_round_trip(tiny_setup):
    _, tokenizer = tiny_setup
    panel = np.random.default_rng(2).normal(0, 0.02, (5, N_SLOTS))
    flat = flatten_days(panel, tokenizer)
    assert flat.shape == (5 * N_SLOTS,)
    # Day-major order: first N_SLOTS tokens are day 0's slots in order.
    np.testing.assert_array_equal(flat[:N_SLOTS], tokenizer.encode(panel[0]))


def test_sample_joint_shapes_and_determinism(tiny_setup):
    model, tokenizer = tiny_setup
    a = sample_joint(model, tokenizer, N_SLOTS, n_paths=3, horizon_days=6, seed=11)
    b = sample_joint(model, tokenizer, N_SLOTS, n_paths=3, horizon_days=6, seed=11)
    assert a.shape == (3, 6, N_SLOTS)
    np.testing.assert_array_equal(a, b)
    prefix = np.random.default_rng(5).normal(0, 0.02, (4, N_SLOTS))
    c = sample_joint(model, tokenizer, N_SLOTS, 3, 6, prefix_matrix=prefix, seed=11)
    assert not np.array_equal(a, c)


def test_correlation_metrics_on_constructed_panels():
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.02, 4000)
    correlated = np.column_stack([base, base + rng.normal(0, 0.002, 4000)])
    independent = rng.normal(0, 0.02, (4000, 2))
    assert mean_pairwise_corr(correlated) > 0.95
    assert abs(mean_pairwise_corr(independent)) < 0.05

    # Regime panel: high-vol days are strongly correlated, calm days are not.
    hot = rng.normal(0, 0.05, 2000)
    stress_days = np.column_stack([hot, hot * 0.9 + rng.normal(0, 0.005, 2000)])
    calm_days = rng.normal(0, 0.005, (2000, 2))
    panel = np.vstack([stress_days, calm_days])
    calm_c, stress_c = stress_split_corr(panel)
    assert stress_c > calm_c + 0.3


def test_independence_shuffle_understates_portfolio_var():
    rng = np.random.default_rng(8)
    shock = rng.normal(0, 0.02, (10, 300))
    # Perfectly comoving 4-name basket: portfolio tail = single-name tail.
    paths = np.repeat(shock[:, :, None], 4, axis=2)
    var_joint, var_indep = portfolio_var_joint_vs_independent(paths, alpha=0.05)
    assert var_joint < var_indep < 0.0  # shuffling away correlation thins the tail


def test_joint_training_smoke(mini_universe, tiny_setup):
    _, tokenizer = tiny_setup
    data = build_joint_windows(mini_universe, tokenizer, block_days=BLOCK_DAYS,
                               train_end="2020-09-30", val_end="2020-12-31")
    cfg = ModelConfig(vocab_size=16, block_size=BLOCK_DAYS * N_SLOTS, n_layer=1,
                      n_head=2, n_embd=32, n_tickers=N_SLOTS, dropout=0.0)
    model, metrics = train(data, cfg, steps=80, batch_size=16,
                           device="cpu", seed=7, log_every=0)
    assert metrics["final_train_loss"] < metrics["first_train_loss"]
    assert np.isfinite(metrics["val_ce"])
