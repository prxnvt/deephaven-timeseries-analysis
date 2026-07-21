"""Backtest engine: strategy signals, portfolio accounting, and run_backtest.

Pure pandas/numpy (no Deephaven imports — see the package docstring). All
path-dependent math lives here so it can be unit-tested headlessly and reused
against both real history and (later milestones) generated synthetic paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Overlay columns each strategy adds to the enriched frame (consumed by charts).
STRATEGY_OVERLAYS = {"sma": ["SMA_fast", "SMA_slow"], "dip": ["RecentHigh"], "dca": []}


# --- Strategies (return a desired 0/1 position series, lookahead handled here) -
def sma_signals(df: pd.DataFrame, fast: int, slow: int):
    sma_fast = df["Close"].rolling(fast, min_periods=fast).mean()
    sma_slow = df["Close"].rolling(slow, min_periods=slow).mean()
    desired = (sma_fast > sma_slow).fillna(False).astype(int).to_numpy()
    overlays = {"SMA_fast": sma_fast.to_numpy(), "SMA_slow": sma_slow.to_numpy()}
    return desired, overlays


def dip_signals(df: pd.DataFrame, drop_pct: float, recover_pct: float, lookback: int):
    close = df["Close"].to_numpy()
    recent_high = df["Close"].rolling(lookback, min_periods=1).max().to_numpy()
    n = len(close)
    desired = np.zeros(n, dtype=int)
    in_mkt = False
    entry = 0.0
    for i in range(n):
        if not in_mkt and close[i] <= recent_high[i] * (1.0 - drop_pct / 100.0):
            in_mkt = True
            # The fill happens one bar after the signal (positions are lagged), so
            # the recovery exit must reference the actual fill price, not the
            # signal-day close. Known to the trader from the fill onward — not
            # lookahead. min() guards a signal on the final bar (never filled).
            entry = close[min(i + 1, n - 1)]
        elif in_mkt and close[i] >= entry * (1.0 + recover_pct / 100.0):
            in_mkt = False
        desired[i] = 1 if in_mkt else 0
    return desired, {"RecentHigh": recent_high}


# --- Portfolio accounting ----------------------------------------------------
def lag_one(desired: np.ndarray) -> np.ndarray:
    """Shift a desired-position series forward one bar (trade AFTER the signal)."""
    return np.concatenate([[0], desired[:-1]]).astype(int)


def simulate_long_flat(close: np.ndarray, desired: np.ndarray, capital: float):
    """All-in / all-out long-flat sim: BUY uses all cash, SELL returns all to cash."""
    n = len(close)
    shares = np.zeros(n)
    cash = np.zeros(n)
    trade = [None] * n
    trade_px = np.full(n, np.nan)
    cur_sh, cur_cash, prev = 0.0, float(capital), 0
    round_trips = []        # (entry_value, exit_value)
    entry_value = None
    for i in range(n):
        pos = int(desired[i])
        if pos == 1 and prev == 0:                      # BUY
            entry_value = cur_cash
            cur_sh, cur_cash = cur_cash / close[i], 0.0
            trade[i], trade_px[i] = "BUY", close[i]
        elif pos == 0 and prev == 1:                    # SELL
            cur_cash, cur_sh = cur_sh * close[i], 0.0
            trade[i], trade_px[i] = "SELL", close[i]
            if entry_value is not None:
                round_trips.append((entry_value, cur_cash))
                entry_value = None
        shares[i], cash[i] = cur_sh, cur_cash
        prev = pos
    if entry_value is not None:  # mark the still-open position to the final close
        round_trips.append((entry_value, cur_sh * close[-1]))
    return shares, cash, trade, trade_px, round_trips


def simulate_dca(close: np.ndarray, capital: float, amount: float, every_n: int):
    n = len(close)
    shares = np.zeros(n)
    cash = np.zeros(n)
    trade = [None] * n
    trade_px = np.full(n, np.nan)
    cur_sh, cur_cash = 0.0, float(capital)
    for i in range(n):
        if i % every_n == 0 and cur_cash >= amount:
            cur_sh += amount / close[i]
            cur_cash -= amount
            trade[i], trade_px[i] = "BUY", close[i]
        shares[i], cash[i] = cur_sh, cur_cash
    return shares, cash, trade, trade_px, []     # DCA has no closed round-trips


# --- Backtest engine ---------------------------------------------------------
def run_backtest(df: pd.DataFrame, strategy: str, params: dict, capital: float,
                 ticker: str | None = None):
    """Backtest one strategy over a pre-sliced per-date frame.

    `df` needs Date and Close columns, date-sorted (see data.slice_universe);
    `ticker` labels the output rows, defaulting to df's Ticker column.
    Returns (enriched_df, periodic_df, stats).
    """
    if df.empty:
        raise RuntimeError("run_backtest received an empty price frame.")
    if ticker is None:
        ticker = str(df["Ticker"].iloc[0]) if "Ticker" in df.columns else "SYNTH"

    close = df["Close"].to_numpy()
    overlays: dict[str, np.ndarray] = {}

    if strategy == "sma":
        fast_i, slow_i = int(params["fast"]), int(params["slow"])
        if slow_i <= fast_i:  # keep windows ordered so the crossover stays meaningful
            slow_i = fast_i + 1
        desired, overlays = sma_signals(df, fast_i, slow_i)
        desired = lag_one(desired)  # execute the bar after the signal (no lookahead)
        shares, cash, trade, trade_px, round_trips = simulate_long_flat(close, desired, capital)
    elif strategy == "dip":
        desired, overlays = dip_signals(
            df, float(params["drop_pct"]), float(params["recover_pct"]), int(params["lookback"])
        )
        desired = lag_one(desired)  # execute the bar after the signal (no lookahead)
        shares, cash, trade, trade_px, round_trips = simulate_long_flat(close, desired, capital)
    elif strategy == "dca":
        shares, cash, trade, trade_px, round_trips = simulate_dca(
            close, capital, float(params["amount"]), max(1, int(params["every_n"]))
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r} (expected sma, dip, or dca).")

    equity = shares * close + cash
    bh_equity = capital * close / close[0]
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    in_market = shares * close

    enriched = pd.DataFrame({
        "Date": df["Date"].to_numpy(),
        "Ticker": ticker,
        "Close": close,
        "Trade": trade,
        "TradePrice": trade_px,
        "Shares": shares,
        "Cash": cash,
        "Strategy_Equity": equity,
        "BuyHold_Equity": bh_equity,
        "ExpInMarket": in_market / equity,
        "ExpCash": cash / equity,
        "Drawdown": drawdown,
    })
    for name, arr in overlays.items():
        enriched[name] = arr

    # Monthly returns (static summary bar chart).
    eq_series = pd.Series(equity, index=pd.DatetimeIndex(df["Date"]))
    monthly = eq_series.resample("ME").last().pct_change().dropna()
    periodic_df = pd.DataFrame(
        {"Period": monthly.index.strftime("%Y-%m"), "PeriodReturn": monthly.to_numpy()}
    )

    wins = sum(1 for ev, xv in round_trips if xv > ev)
    win_rate = (wins / len(round_trips)) if round_trips else None
    stats = {
        "final_value": float(equity[-1]),
        "total_return": float(equity[-1] / capital - 1.0),
        "vs_bh": float(equity[-1] / bh_equity[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "n_trades": int(sum(1 for t in trade if t == "BUY")),
        "win_rate": win_rate,
    }
    return enriched, periodic_df, stats
