"""INTC-and-friends market simulator — Deephaven what-if trading dashboard.

A stock-market "what-if" backtester: pick one of a basket of ~20 free-data tickers,
choose a strategy, set capital + parameters, and watch the strategy trade against an
*unfolding* historical market (animated with Deephaven's TableReplayer). The dashboard
is a Sunbird-style multi-panel `ui.dashboard`: live KPI cards, an equity curve vs
buy-&-hold, price + signal markers, exposure (in-market vs cash), a live trade log, and
monthly returns.

Architecture (see docs and the project plan):
  * Universe downloaded once via yfinance (multi-ticker) -> tidy long DataFrame.
  * Data + backtest math live in the `marketlab` package (pure pandas/numpy,
    unit-tested, no Deephaven imports); this script is the thin Deephaven layer.
  * For the selected (ticker, strategy, params, capital, date-range) the FULL backtest is
    precomputed via marketlab.backtest.run_backtest, producing one enriched per-date frame.
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

from marketlab.backtest import STRATEGY_OVERLAYS, run_backtest
from marketlab.data import load_universe, slice_universe

from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx
from deephaven.replay import TableReplayer
from deephaven.time import to_j_instant

# --- Configuration -----------------------------------------------------------
REPLAY_SECONDS = 90.0                # wall-clock duration of the animated replay
CACHE_PATH = "/data/universe_cache.parquet"   # /data is the container's mounted volume

STRATEGY_LABELS = {
    "sma": "SMA Crossover",
    "dip": "Buy the Dip",
    "dca": "Dollar-Cost Averaging",
}
RANGE_KEYS = ["Max", "5Y", "3Y", "1Y"]


# --- Data layer (downloaded once at module load, via marketlab.data) ---------
PRICES_LONG = load_universe(CACHE_PATH)
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
    df = slice_universe(PRICES_LONG, ticker, start, end)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} in {start.date()}..{end.date()}.")
    enriched_df, periodic_df, stats = run_backtest(df, strategy, params, capital)
    enriched_df, r_start, r_end = _add_replay_time(enriched_df, REPLAY_SECONDS)
    enriched_tbl = dhpd.to_table(enriched_df)

    replayer = TableReplayer(r_start, r_end)
    live = replayer.add_table(enriched_tbl, "ReplayTime")
    replayer.start()

    overlays = STRATEGY_OVERLAYS[strategy]
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
