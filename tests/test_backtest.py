"""Unit tests for marketlab.backtest — hand-computed expectations throughout."""

import numpy as np
import pytest

from marketlab.backtest import (
    STRATEGY_OVERLAYS,
    dip_signals,
    lag_one,
    run_backtest,
    simulate_dca,
    simulate_long_flat,
    sma_signals,
)


def test_lag_one_shifts_forward():
    desired = np.array([0, 1, 1, 0, 1])
    assert lag_one(desired).tolist() == [0, 0, 1, 1, 0]


class TestSma:
    # fast=2 vs slow=3 over this series crosses up at index 4 and down at index 8:
    #   sma2 = [-, 10, 10, 10, 15, 25, 35, 45, 30, 10, 10, 10]
    #   sma3 = [-, -, 10, 10, 13.3, 20, 30, 40, 33.3, 23.3, 10, 10]
    CLOSES = [10, 10, 10, 10, 20, 30, 40, 50, 10, 10, 10, 10]

    def test_desired_positions(self, price_frame):
        desired, overlays = sma_signals(price_frame(self.CLOSES), fast=2, slow=3)
        assert desired.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        assert set(overlays) == {"SMA_fast", "SMA_slow"}

    def test_trades_fill_one_bar_after_signal(self, price_frame):
        df = price_frame(self.CLOSES)
        enriched, _, stats = run_backtest(df, "sma", {"fast": 2, "slow": 3}, capital=100.0)
        trades = enriched["Trade"].tolist()
        assert trades[5] == "BUY" and enriched["TradePrice"][5] == 30.0
        assert trades[9] == "SELL" and enriched["TradePrice"][9] == 10.0
        assert stats["n_trades"] == 1
        # 100 -> buy@30 -> sell@10 = 33.33; a losing round trip.
        assert stats["final_value"] == pytest.approx(100.0 * 10.0 / 30.0)
        assert stats["win_rate"] == 0.0
        # Equity peaks at 50/30*100 = 166.67 (bar 7), troughs at 33.33 -> -80%.
        assert stats["max_drawdown"] == pytest.approx(-0.8)

    def test_no_lookahead(self, price_frame):
        """Signals up to bar k must not depend on any close after bar k."""
        k = 6
        base, _ = sma_signals(price_frame(self.CLOSES), fast=2, slow=3)
        perturbed_closes = list(self.CLOSES)
        for j in range(k + 1, len(perturbed_closes)):
            perturbed_closes[j] = 999.0
        perturbed, _ = sma_signals(price_frame(perturbed_closes), fast=2, slow=3)
        assert base[: k + 1].tolist() == perturbed[: k + 1].tolist()


class TestDip:
    def test_exit_references_fill_price_not_signal_close(self, price_frame):
        # Running high hits 100; bar 2 (89) breaches the -10% trigger -> signal.
        # Fill is bar 3 at 95, so the +15% recovery target is 109.25: bar 6 (110)
        # exits. (Under the old signal-close entry of 89 the target would be
        # 102.35 and bar 5 at 104 would exit -- this test pins the fix.)
        closes = [100, 100, 89, 95, 102, 104, 110, 111]
        desired, overlays = dip_signals(
            price_frame(closes), drop_pct=10.0, recover_pct=15.0, lookback=8
        )
        assert desired.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
        assert "RecentHigh" in overlays

        _, _, stats = run_backtest(
            price_frame(closes), "dip",
            {"drop_pct": 10.0, "recover_pct": 15.0, "lookback": 8}, capital=100.0,
        )
        # Lagged fills: buy bar 3 @95, sell bar 7 @111 -> a winning round trip.
        assert stats["n_trades"] == 1
        assert stats["win_rate"] == 1.0
        assert stats["final_value"] == pytest.approx(100.0 * 111.0 / 95.0)

    def test_signal_on_final_bar_never_fills(self, price_frame):
        closes = [100, 100, 89]  # trigger on the last bar; no next bar to fill on
        _, _, stats = run_backtest(
            price_frame(closes), "dip",
            {"drop_pct": 10.0, "recover_pct": 15.0, "lookback": 3}, capital=100.0,
        )
        assert stats["n_trades"] == 0
        assert stats["final_value"] == 100.0


class TestDca:
    def test_cadence_and_cash_depletion(self, price_frame):
        closes = [10, 10, 20, 20, 40, 40]
        shares, cash, trade, trade_px, round_trips = simulate_dca(
            np.array(closes, dtype=float), capital=1000.0, amount=400.0, every_n=2
        )
        # Buys on bars 0 (40sh @10) and 2 (20sh @20); bar 4 skipped (cash 200 < 400).
        assert shares.tolist() == [40, 40, 60, 60, 60, 60]
        assert cash.tolist() == [600, 600, 200, 200, 200, 200]
        assert [t for t in trade if t] == ["BUY", "BUY"]
        assert round_trips == []

    def test_stats_via_run_backtest(self, price_frame):
        closes = [10, 10, 20, 20, 40, 40]
        _, _, stats = run_backtest(
            price_frame(closes), "dca", {"amount": 400.0, "every_n": 2}, capital=1000.0
        )
        assert stats["final_value"] == pytest.approx(60 * 40 + 200)  # 2600
        assert stats["n_trades"] == 2
        assert stats["win_rate"] is None  # DCA has no closed round trips


class TestAccounting:
    def test_drawdown_and_buy_hold(self, price_frame):
        # All-in on bar 0 (DCA with amount == capital) tracks the price exactly:
        # equity [100, 200, 100, 150, 300]; peak-relative trough is -50% at bar 2.
        closes = [10, 20, 10, 15, 30]
        enriched, _, stats = run_backtest(
            price_frame(closes), "dca", {"amount": 100.0, "every_n": 1}, capital=100.0
        )
        assert stats["max_drawdown"] == pytest.approx(-0.5)
        assert stats["total_return"] == pytest.approx(2.0)
        assert stats["vs_bh"] == pytest.approx(0.0)
        expected_bh = [100.0 * c / closes[0] for c in closes]
        assert enriched["BuyHold_Equity"].tolist() == pytest.approx(expected_bh)

    def test_long_flat_round_trip_values(self):
        close = np.array([10.0, 20.0, 5.0])
        desired = np.array([1, 1, 0])  # already-lagged positions
        shares, cash, trade, trade_px, round_trips = simulate_long_flat(close, desired, 100.0)
        assert trade == ["BUY", None, "SELL"]
        assert cash[-1] == pytest.approx(50.0)  # 10sh bought @10, sold @5
        assert round_trips == [(100.0, pytest.approx(50.0))]


class TestRunBacktestContract:
    def test_empty_frame_raises(self, price_frame):
        with pytest.raises(RuntimeError, match="empty"):
            run_backtest(price_frame([]).iloc[0:0], "sma", {"fast": 2, "slow": 3}, 100.0)

    def test_unknown_strategy_raises(self, price_frame):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_backtest(price_frame([1, 2, 3]), "nope", {}, 100.0)

    def test_slow_window_coerced_past_fast(self, price_frame):
        # slow <= fast silently becomes fast+1 rather than crashing.
        _, _, stats = run_backtest(
            price_frame([10, 11, 12, 13, 14, 15]), "sma", {"fast": 3, "slow": 2}, 100.0
        )
        assert stats["final_value"] > 0

    def test_enriched_schema(self, mini_universe):
        df = mini_universe[mini_universe["Ticker"] == "AAA"].reset_index(drop=True)
        enriched, periodic, stats = run_backtest(df, "sma", {"fast": 5, "slow": 20}, 10_000.0)
        assert len(enriched) == len(df)
        expected = {
            "Date", "Ticker", "Close", "Trade", "TradePrice", "Shares", "Cash",
            "Strategy_Equity", "BuyHold_Equity", "ExpInMarket", "ExpCash", "Drawdown",
        } | set(STRATEGY_OVERLAYS["sma"])
        assert expected <= set(enriched.columns)
        assert (enriched["Ticker"] == "AAA").all()
        assert list(periodic.columns) == ["Period", "PeriodReturn"]
        assert set(stats) == {
            "final_value", "total_return", "vs_bh", "max_drawdown", "n_trades", "win_rate",
        }
