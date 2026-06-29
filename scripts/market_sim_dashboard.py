"""INTC-and-friends market simulator — Deephaven what-if trading dashboard.

A stock-market "what-if" backtester: pick one of a basket of ~20 free-data tickers,
choose a strategy, set capital + parameters, and watch the strategy trade against an
*unfolding* historical market (animated with Deephaven's TableReplayer). The dashboard
is a Sunbird-style multi-panel `ui.dashboard`: live KPI cards, an equity curve vs
buy-&-hold, price + signal markers, exposure (in-market vs cash), a live trade log, and
monthly returns.

Architecture (see docs and the project plan):
  * Universe downloaded once via yfinance (multi-ticker) -> tidy long DataFrame.
  * For the selected (ticker, strategy, params, capital, date-range) the FULL backtest is
    precomputed in pandas/numpy (path-dependent math is trivial there), producing one
    enriched per-date frame.
  * A synthetic, compressed `ReplayTime` Instant column maps ~years of bars into a short
    wall-clock window; `TableReplayer` then reveals rows over time. Charts + KPIs bind to
    the replaying table and update live.

Run it from the Deephaven IDE (./scripts is mounted to /data/storage/notebooks): open this
file in the Notebooks panel and run it. It binds `dashboard`, which the IDE renders.

Only runs inside the Deephaven engine (imports deephaven.*), not under a plain `python`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx
from deephaven.replay import TableReplayer
from deephaven.time import to_j_instant

# --- Configuration -----------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "UNH",
    "XOM", "JNJ", "WMT", "PG", "MA", "HD", "KO", "PEP", "COST", "CVX",
]
HISTORY_START = "2014-01-01"
HISTORY_END = "2024-01-01"
MIN_ROWS = 200                       # drop a ticker with less history than this
REPLAY_SECONDS = 90.0                # wall-clock duration of the animated replay
CACHE_PATH = "/data/universe_cache.parquet"   # /data is the container's mounted volume

STRATEGY_LABELS = {
    "sma": "SMA Crossover",
    "dip": "Buy the Dip",
    "dca": "Dollar-Cost Averaging",
}
OVERLAYS = {"sma": ["SMA_fast", "SMA_slow"], "dip": ["RecentHigh"], "dca": []}
RANGE_KEYS = ["Max", "5Y", "3Y", "1Y"]


# --- Data layer (downloaded once at module load) -----------------------------
def _download_universe() -> pd.DataFrame:
    """Download the 20-ticker basket and reshape to tidy long form.

    Columns: Date, Ticker, Open, High, Low, Close, Volume.
    """
    raw = yf.download(
        TICKERS, start=HISTORY_START, end=HISTORY_END,
        auto_adjust=True, group_by="ticker", progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check the container's network.")
    # group_by='ticker' yields a (Ticker, Field) column MultiIndex; stack level 0.
    long_df = (
        raw.stack(level=0, future_stack=True)
        .rename_axis(["Date", "Ticker"])
        .reset_index()
    )
    long_df = long_df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    long_df = long_df.dropna(subset=["Close"])
    long_df["Ticker"] = long_df["Ticker"].astype(str)
    # Drop tickers with too little history (keeps the basket clean / picker honest).
    counts = long_df.groupby("Ticker")["Close"].transform("size")
    long_df = long_df[counts >= MIN_ROWS].reset_index(drop=True)
    return long_df


def _load_universe() -> pd.DataFrame:
    """Load the universe from the local parquet cache, else download and cache it."""
    try:
        cached = pd.read_parquet(CACHE_PATH)
        if not cached.empty:
            return cached
    except Exception:
        pass
    long_df = _download_universe()
    try:
        long_df.to_parquet(CACHE_PATH, index=False)
    except Exception:
        pass  # caching is best-effort; never block on it
    return long_df


PRICES_LONG = _load_universe()
AVAILABLE_TICKERS = sorted(PRICES_LONG["Ticker"].unique())
DATE_MIN = pd.Timestamp(PRICES_LONG["Date"].min())
DATE_MAX = pd.Timestamp(PRICES_LONG["Date"].max())
DEFAULT_TICKER = "AAPL" if "AAPL" in AVAILABLE_TICKERS else AVAILABLE_TICKERS[0]


def _resolve_range(range_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = DATE_MAX
    offsets = {"5Y": 5, "3Y": 3, "1Y": 1}
    if range_key in offsets:
        start = end - pd.DateOffset(years=offsets[range_key])
    else:  # "Max"
        start = DATE_MIN
    return start, end


# --- Strategies (return a desired 0/1 position series, lookahead handled here) -
def _sma_signals(df: pd.DataFrame, fast: int, slow: int):
    sma_fast = df["Close"].rolling(fast, min_periods=fast).mean()
    sma_slow = df["Close"].rolling(slow, min_periods=slow).mean()
    desired = (sma_fast > sma_slow).fillna(False).astype(int).to_numpy()
    overlays = {"SMA_fast": sma_fast.to_numpy(), "SMA_slow": sma_slow.to_numpy()}
    return desired, overlays


def _dip_signals(df: pd.DataFrame, drop_pct: float, recover_pct: float, lookback: int):
    close = df["Close"].to_numpy()
    recent_high = df["Close"].rolling(lookback, min_periods=1).max().to_numpy()
    n = len(close)
    desired = np.zeros(n, dtype=int)
    in_mkt = False
    entry = 0.0
    for i in range(n):
        if not in_mkt and close[i] <= recent_high[i] * (1.0 - drop_pct / 100.0):
            in_mkt = True
            entry = close[i]
        elif in_mkt and close[i] >= entry * (1.0 + recover_pct / 100.0):
            in_mkt = False
        desired[i] = 1 if in_mkt else 0
    return desired, {"RecentHigh": recent_high}


# --- Portfolio accounting ----------------------------------------------------
def _lag_one(desired: np.ndarray) -> np.ndarray:
    """Shift a desired-position series forward one bar (trade AFTER the signal)."""
    return np.concatenate([[0], desired[:-1]]).astype(int)


def _simulate_long_flat(close: np.ndarray, desired: np.ndarray, capital: float):
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


def _simulate_dca(close: np.ndarray, capital: float, amount: float, every_n: int):
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
def run_backtest(ticker, strategy, params, capital, start, end):
    """Return (enriched_df, periodic_df, stats) for one configuration."""
    df = PRICES_LONG[
        (PRICES_LONG["Ticker"] == ticker)
        & (PRICES_LONG["Date"] >= start)
        & (PRICES_LONG["Date"] <= end)
    ].sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start.date()}..{end.date()}.")

    close = df["Close"].to_numpy()
    overlays: dict[str, np.ndarray] = {}

    if strategy == "sma":
        fast_i, slow_i = int(params["fast"]), int(params["slow"])
        if slow_i <= fast_i:  # keep windows ordered so the crossover stays meaningful
            slow_i = fast_i + 1
        desired, overlays = _sma_signals(df, fast_i, slow_i)
        desired = _lag_one(desired)  # execute the bar after the signal (no lookahead)
        shares, cash, trade, trade_px, round_trips = _simulate_long_flat(close, desired, capital)
    elif strategy == "dip":
        desired, overlays = _dip_signals(
            df, float(params["drop_pct"]), float(params["recover_pct"]), int(params["lookback"])
        )
        desired = _lag_one(desired)  # execute the bar after the signal (no lookahead)
        shares, cash, trade, trade_px, round_trips = _simulate_long_flat(close, desired, capital)
    else:  # dca
        shares, cash, trade, trade_px, round_trips = _simulate_dca(
            close, capital, float(params["amount"]), max(1, int(params["every_n"]))
        )

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


def _add_replay_time(df: pd.DataFrame, seconds: float):
    """Append a compressed, tz-aware UTC `ReplayTime` Instant column; return (df, start, end)."""
    n = len(df)
    now_ns = pd.Timestamp.now(tz="UTC").value
    span_ns = int(seconds * 1e9)
    offsets = (np.linspace(0.0, 1.0, n) * span_ns).astype("int64")
    df = df.copy()
    df["ReplayTime"] = pd.to_datetime(now_ns + offsets, utc=True)
    start = to_j_instant(pd.Timestamp(now_ns, tz="UTC"))
    end = to_j_instant(pd.Timestamp(now_ns + span_ns + int(1e9), tz="UTC"))
    return df, start, end


# --- Build one run: enriched table + replayer + figures + KPI tables ----------
def build_run(ticker, strategy, params, capital, start, end):
    capital = float(capital) if capital else 10_000.0
    enriched_df, periodic_df, stats = run_backtest(ticker, strategy, params, capital, start, end)
    enriched_df, r_start, r_end = _add_replay_time(enriched_df, REPLAY_SECONDS)
    enriched_tbl = dhpd.to_table(enriched_df)

    replayer = TableReplayer(r_start, r_end)
    live = replayer.add_table(enriched_tbl, "ReplayTime")
    replayer.start()

    overlays = OVERLAYS[strategy]
    equity_fig = dx.line(
        live, x="Date", y=["Strategy_Equity", "BuyHold_Equity"],
        title="Equity: Strategy vs Buy & Hold",
    )
    price_line = dx.line(live, x="Date", y=["Close"] + overlays, title="Price & Signals")
    try:
        markers = dx.scatter(live.where("Trade != null"), x="Date", y="TradePrice", by="Trade")
        price_fig = dx.layer(price_line, markers)
    except Exception:
        price_fig = price_line
    exposure_fig = dx.area(
        live, x="Date", y=["ExpInMarket", "ExpCash"],
        groupnorm="fraction", title="Exposure: In-Market vs Cash",
    )
    periodic_tbl = dhpd.to_table(periodic_df) if len(periodic_df) else None
    periodic_fig = (
        dx.bar(periodic_tbl, x="Period", y="PeriodReturn", title="Monthly Returns")
        if periodic_tbl is not None else ui.text("Not enough data for monthly returns.")
    )
    trade_log = live.where("Trade != null").view(
        ["Date", "Trade", "TradePrice", "Strategy_Equity"]
    )

    return {
        "live": live,
        "equity_fig": equity_fig,
        "price_fig": price_fig,
        "exposure_fig": exposure_fig,
        "periodic_fig": periodic_fig,
        "trade_log": trade_log,
        "stats": stats,
    }, replayer


# --- Formatting helpers ------------------------------------------------------
def _na(x):
    return x is None or pd.isna(x)


def _money(x):
    return "—" if _na(x) else f"${x:,.0f}"


def _pct(x):
    return "—" if _na(x) else f"{x:+.1%}"


def _pct_plain(x):
    return "—" if _na(x) else f"{x:.1%}"


def _kpi_card(label, value, accent=None):
    return ui.view(
        ui.flex(
            ui.text(label, color="gray-700"),
            ui.heading(value, level=3, color=accent),
            direction="column", gap="size-50",
        ),
        padding="size-150", border_width="thin", border_color="gray-400",
        border_radius="medium", min_width="size-1600", flex_grow=1,
    )


# --- The dashboard component -------------------------------------------------
@ui.component
def market_sim():
    # Control state
    ticker, set_ticker = ui.use_state(DEFAULT_TICKER)
    strategy, set_strategy = ui.use_state("sma")
    capital, set_capital = ui.use_state(10_000.0)
    fast, set_fast = ui.use_state(20)
    slow, set_slow = ui.use_state(100)
    drop_pct, set_drop_pct = ui.use_state(10.0)
    recover_pct, set_recover_pct = ui.use_state(15.0)
    lookback, set_lookback = ui.use_state(60)
    dca_amount, set_dca_amount = ui.use_state(500.0)
    dca_every, set_dca_every = ui.use_state(21)
    range_key, set_range_key = ui.use_state("Max")
    nonce, set_nonce = ui.use_state(0)

    # Run state (holds the live tables / figures published by the effect)
    run, set_run = ui.use_state(None)
    replayer_ref = ui.use_ref(None)
    run_in_context = ui.use_execution_context()

    if strategy == "sma":
        params = {"fast": fast, "slow": slow}
    elif strategy == "dip":
        params = {"drop_pct": drop_pct, "recover_pct": recover_pct, "lookback": lookback}
    else:
        params = {"amount": dca_amount, "every_n": dca_every}

    start, end = _resolve_range(range_key)
    config_key = (ticker, strategy, float(capital), range_key, nonce,
                  tuple(sorted((k, float(v)) for k, v in params.items())))

    def _build_and_publish():
        # Tear down any previous replayer before building a new one.
        if replayer_ref.current is not None:
            try:
                replayer_ref.current.shutdown()
            except Exception:
                pass
            replayer_ref.current = None
        try:
            new_run, replayer = build_run(ticker, strategy, params, capital, start, end)
            replayer_ref.current = replayer
            set_run(new_run)
        except Exception as exc:  # degrade gracefully instead of throwing in render
            set_run({"error": str(exc)})

    def _effect():
        # use_effect's own liveness scope retains the tables created here (the
        # replay + KPI tables) for the render that consumes them.
        run_in_context(_build_and_publish)

        def _cleanup():
            rp = replayer_ref.current
            if rp is not None:
                try:
                    rp.shutdown()
                except Exception:
                    pass
                replayer_ref.current = None

        return _cleanup

    ui.use_effect(_effect, [config_key])

    # Final backtest KPIs: static values from the precomputed run (no per-tick
    # table-listener hooks — those raced the replay and crashed the render).
    ready = run is not None and "error" not in run
    stats = run["stats"] if ready else {}
    final_v = stats.get("final_value")
    ret_v = stats.get("total_return")
    vsbh_v = stats.get("vs_bh")
    dd_v = stats.get("max_drawdown")
    ntr_v = stats.get("n_trades")
    win_rate = stats.get("win_rate")

    # ---- Controls panel ----
    if strategy == "sma":
        param_controls = [
            ui.slider(label="Fast MA (days)", value=fast, on_change=set_fast,
                      min_value=5, max_value=60, step=1),
            ui.slider(label="Slow MA (days)", value=slow, on_change=set_slow,
                      min_value=20, max_value=200, step=5),
        ]
    elif strategy == "dip":
        param_controls = [
            ui.slider(label="Buy after drop %", value=drop_pct, on_change=set_drop_pct,
                      min_value=2, max_value=40, step=1),
            ui.slider(label="Sell after recover %", value=recover_pct, on_change=set_recover_pct,
                      min_value=2, max_value=50, step=1),
            ui.slider(label="High lookback (days)", value=lookback, on_change=set_lookback,
                      min_value=20, max_value=250, step=5),
        ]
    else:
        param_controls = [
            ui.number_field(label="Contribution ($)", value=dca_amount, on_change=set_dca_amount,
                            min_value=50.0, max_value=5000.0, step=50.0,
                            format_options={"style": "currency", "currency": "USD"}),
            ui.slider(label="Every N trading days", value=dca_every, on_change=set_dca_every,
                      min_value=5, max_value=63, step=1),
        ]

    controls = ui.flex(
        ui.picker(
            *[ui.item(t, key=t) for t in AVAILABLE_TICKERS],
            selected_key=ticker, on_change=set_ticker, label="Ticker",
        ),
        ui.picker(
            *[ui.item(STRATEGY_LABELS[k], key=k) for k in ("sma", "dip", "dca")],
            selected_key=strategy, on_change=set_strategy, label="Strategy",
        ),
        ui.number_field(
            label="Initial capital", value=capital, on_change=set_capital,
            min_value=1_000.0, step=1_000.0,
            format_options={"style": "currency", "currency": "USD"},
        ),
        *param_controls,
        ui.picker(
            *[ui.item(k, key=k) for k in RANGE_KEYS],
            selected_key=range_key, on_change=set_range_key, label="Date range",
        ),
        ui.button("Restart replay", on_press=lambda _e: set_nonce(nonce + 1), variant="accent"),
        direction="column", gap="size-100",
    )

    # ---- KPI row ----
    kpi_row = ui.flex(
        _kpi_card("Final value", _money(final_v)),
        _kpi_card("Total return", _pct(ret_v)),
        _kpi_card("vs Buy & Hold", _pct(vsbh_v)),
        _kpi_card("Max drawdown", _pct_plain(dd_v)),
        _kpi_card("# Buys", "—" if _na(ntr_v) else str(int(ntr_v))),
        _kpi_card("Win rate", _pct_plain(win_rate)),
        direction="row", gap="size-100", wrap=True,
    )

    # ---- Panel contents (placeholders until the first run is published) ----
    placeholder = run["error"] if (run and "error" in run) else "Building backtest…"

    def _content(key):
        return run[key] if ready else ui.text(placeholder)

    trade_log_content = ui.table(run["trade_log"]) if ready else ui.text(placeholder)

    return ui.column(
        ui.row(
            ui.column(ui.panel(controls, title="Controls"), width=22),
            ui.panel(kpi_row, title=f"{ticker} · {STRATEGY_LABELS[strategy]}"),
            height=26,
        ),
        ui.row(
            ui.panel(_content("equity_fig"), title="Equity: Strategy vs Buy & Hold"),
            ui.panel(_content("price_fig"), title="Price & Signals"),
            height=40,
        ),
        ui.row(
            ui.panel(_content("exposure_fig"), title="Exposure (In-Market vs Cash)"),
            ui.panel(trade_log_content, title="Trade Log"),
            ui.panel(_content("periodic_fig"), title="Monthly Returns"),
            height=34,
        ),
    )


# Binding the dashboard renders it in the IDE.
dashboard = ui.dashboard(market_sim())
