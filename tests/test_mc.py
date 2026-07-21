"""Unit tests for marketlab.mc on deterministic fake paths."""

import numpy as np
import pytest

from marketlab.mc import mc_backtest, summarize, synthetic_price_frame


@pytest.fixture
def constant_paths():
    """Three 100-day paths with constant daily log-returns: +1%, 0%, -1%."""
    return np.stack([
        np.full(100, 0.01),
        np.full(100, 0.0),
        np.full(100, -0.01),
    ])


def test_synthetic_price_frame_schema():
    frame = synthetic_price_frame(np.full(10, 0.01), start_price=50.0)
    assert list(frame.columns) == ["Date", "Ticker", "Close"]
    assert len(frame) == 10
    assert (frame["Ticker"] == "SYNTH").all()
    assert frame["Close"].iloc[0] == pytest.approx(50.0 * np.exp(0.01))
    assert frame["Date"].is_monotonic_increasing


def test_mc_backtest_orders_outcomes(constant_paths):
    # All-in buy-and-hold via DCA(amount=capital, every day): outcome tracks path.
    per_path = mc_backtest(constant_paths, 100.0, "dca",
                           {"amount": 1000.0, "every_n": 1}, capital=1000.0)
    assert len(per_path) == 3
    finals = per_path.sort_values("path")["final_value"].to_numpy()
    assert finals[0] > finals[1] > finals[2]
    assert finals[1] == pytest.approx(1000.0)
    # +1%/day for 99 more days after entry at the first close.
    assert finals[0] == pytest.approx(1000.0 * np.exp(0.01 * 99))


def test_summarize_percentiles_and_real_position(constant_paths):
    per_path = mc_backtest(constant_paths, 100.0, "dca",
                           {"amount": 1000.0, "every_n": 1}, capital=1000.0)
    real_stats = {"final_value": 1_000_000.0, "total_return": 999.0,
                  "vs_bh": 0.0, "max_drawdown": 0.0}
    table = summarize(per_path, real_stats)
    assert list(table["metric"]) == ["final_value", "total_return", "vs_bh", "max_drawdown"]
    row = table[table["metric"] == "final_value"].iloc[0]
    assert row["p5"] <= row["p50"] <= row["p95"]
    assert row["p50"] == pytest.approx(1000.0)
    assert row["real"] == 1_000_000.0
    assert row["real_pctile"] == 100.0  # real beats every synthetic path

    plain = summarize(per_path)  # no real stats -> no real columns
    assert "real" not in plain.columns
