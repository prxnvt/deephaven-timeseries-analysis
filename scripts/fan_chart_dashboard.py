"""Fan-chart dashboard — a probability cone over a ticker's next N trading days.

The MarketGPT checkpoint (see README: "makemore for markets") is prompted with
the last 60 REAL trading days of the chosen ticker, then sampled for hundreds of
continuation paths; the panel shows the 5/25/50/75/95 percentile cone joined to
recent real history, plus Monte-Carlo KPIs for a strategy run over those same
paths. This is a distribution view, not a forecast: the median line is not a
price target, the cone is the model's honest uncertainty.

Temperature note (empirical, documented in the README): for return tokens,
temperature behaves as a regime-persistence dial, not a simple stress dial —
LOW values amplify the model's self-exciting volatility feedback, HIGH values
wash conditionals toward the iid marginal.

Settings apply when you press Generate. Run from the IDE Notebooks panel; it
binds `dashboard`. Only runs inside the Deephaven engine (imports deephaven.*).
"""

import numpy as np
import pandas as pd

from marketlab.data import load_universe
from marketlab.mc import DEFAULT_PARAMS, mc_backtest
from marketlab.sample import generate, load_artifacts, to_price_paths

from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx

# --- Configuration -----------------------------------------------------------
ARTIFACTS_DIR = "/opt/project/artifacts"
CACHE_PATH = "/data/universe_cache.parquet"
PREFIX_DAYS = 60          # real trading days fed to the model as the prompt
REAL_TAIL_DAYS = 120      # real history shown to the left of the cone
CAPITAL = 10_000.0
PATH_CHOICES = ("100", "250", "500")

STRATEGY_LABELS = {
    "sma": "SMA Crossover",
    "dip": "Buy the Dip",
    "dca": "Dollar-Cost Averaging",
}

# --- Module-load state (checkpoint + universe, both cached on disk) ----------
MODEL, TOKENIZER, META = load_artifacts(ARTIFACTS_DIR)
PRICES_LONG = load_universe(CACHE_PATH)
# Only offer tickers the model was trained on AND we have prices for.
AVAILABLE_TICKERS = sorted(set(META["tickers"]) & set(PRICES_LONG["Ticker"].unique()))
DEFAULT_TICKER = "AAPL" if "AAPL" in AVAILABLE_TICKERS else AVAILABLE_TICKERS[0]


# --- One generation run ------------------------------------------------------
def build_fan(ticker, strategy, horizon, temperature, n_paths, seed):
    real = (
        PRICES_LONG[PRICES_LONG["Ticker"] == ticker]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    closes = real["Close"].to_numpy(dtype=np.float64)
    prefix_returns = np.diff(np.log(closes[-(PREFIX_DAYS + 1):]))
    start_price = float(closes[-1])
    last_date = pd.Timestamp(real["Date"].iloc[-1])

    paths = generate(
        MODEL, TOKENIZER, META["tickers"].index(ticker),
        n_paths=n_paths, horizon=horizon, temperature=temperature,
        prefix_returns=prefix_returns, seed=seed,
    )
    prices = to_price_paths(paths, start_price)

    # Percentile cone, prepended with the anchor point so it joins real history.
    levels = [5, 25, 50, 75, 95]
    cone = np.percentile(prices, levels, axis=0)  # (n_levels, horizon)
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    fan_df = pd.DataFrame({"Date": [last_date, *future_dates]})
    for lvl, row in zip(levels, cone):
        fan_df[f"P{lvl}"] = [start_price, *row]

    real_df = real.tail(REAL_TAIL_DAYS)[["Date", "Close"]].rename(
        columns={"Close": "Real"}
    )

    fan_tbl = dhpd.to_table(fan_df)
    real_tbl = dhpd.to_table(real_df)
    fan_lines = dx.line(
        fan_tbl, x="Date", y=[f"P{lvl}" for lvl in levels],
        title=f"{ticker}: {n_paths} sampled continuations, next {horizon} trading days",
    )
    try:
        fig = dx.layer(dx.line(real_tbl, x="Date", y="Real"), fan_lines)
    except Exception:
        fig = fan_lines

    # Monte-Carlo KPIs: the chosen strategy over the SAME sampled paths.
    per_path = mc_backtest(paths, start_price, strategy, DEFAULT_PARAMS[strategy], CAPITAL)
    kpis = {
        "p_beat_bh": float((per_path["vs_bh"] > 0).mean()),
        "median_final": float(per_path["final_value"].median()),
        "p5_final": float(np.percentile(per_path["final_value"], 5)),
        "p5_drawdown": float(np.percentile(per_path["max_drawdown"], 5)),
    }
    return {"fig": fig, "kpis": kpis}


# --- Formatting helpers (kept local: notebook scripts stay self-contained) ---
def _kpi_card(label, value):
    return ui.view(
        ui.flex(
            ui.text(label, color="gray-700"),
            ui.heading(value, level=3),
            direction="column", gap="size-50",
        ),
        padding="size-150", border_width="thin", border_color="gray-400",
        border_radius="medium", min_width="size-1600", flex_grow=1,
    )


# --- The dashboard component -------------------------------------------------
@ui.component
def fan_chart():
    ticker, set_ticker = ui.use_state(DEFAULT_TICKER)
    strategy, set_strategy = ui.use_state("sma")
    # Default to the max horizon: SMA(20/100) needs ~100 bars of MA warmup
    # before it can trade, so short horizons leave the MC cards mostly inert.
    horizon, set_horizon = ui.use_state(250)
    temperature, set_temperature = ui.use_state(1.0)
    n_paths, set_n_paths = ui.use_state("250")
    nonce, set_nonce = ui.use_state(0)

    run, set_run = ui.use_state(None)
    run_in_context = ui.use_execution_context()

    def _build_and_publish():
        try:
            new_run = build_fan(
                ticker, strategy, int(horizon), float(temperature), int(n_paths),
                seed=1000 + nonce,
            )
            set_run(new_run)
        except Exception as exc:  # degrade gracefully instead of throwing in render
            set_run({"error": str(exc)})

    def _effect():
        # Regenerate only when Generate bumps the nonce (and once on mount);
        # the closure reads the control state as of that press.
        run_in_context(_build_and_publish)

    ui.use_effect(_effect, [nonce])

    controls = ui.flex(
        ui.picker(
            *[ui.item(t, key=t) for t in AVAILABLE_TICKERS],
            selected_key=ticker, on_change=set_ticker, label="Ticker",
        ),
        ui.picker(
            *[ui.item(STRATEGY_LABELS[k], key=k) for k in ("sma", "dip", "dca")],
            selected_key=strategy, on_change=set_strategy, label="Strategy (for MC cards)",
        ),
        ui.slider(
            label="Horizon (trading days)", value=horizon, on_change=set_horizon,
            min_value=30, max_value=250, step=5,
        ),
        ui.slider(
            label="Sampling temperature", value=temperature, on_change=set_temperature,
            min_value=0.7, max_value=1.5, step=0.05,
        ),
        ui.picker(
            *[ui.item(p, key=p) for p in PATH_CHOICES],
            selected_key=n_paths, on_change=set_n_paths, label="Paths",
        ),
        ui.button("Generate", on_press=lambda _e: set_nonce(nonce + 1), variant="accent"),
        ui.text("Settings apply on Generate. The cone is a distribution, not a forecast."),
        direction="column", gap="size-100",
    )

    ready = run is not None and "error" not in run
    kpis = run["kpis"] if ready else {}
    placeholder = run["error"] if (run and "error" in run) else "Sampling continuations…"

    def _pct(x):
        return "—" if x is None else f"{x:.0%}"

    def _money(x):
        return "—" if x is None else f"${x:,.0f}"

    kpi_row = ui.flex(
        _kpi_card("P(beats buy & hold)", _pct(kpis.get("p_beat_bh"))),
        _kpi_card("Median final value", _money(kpis.get("median_final"))),
        _kpi_card("5th-pct final value", _money(kpis.get("p5_final"))),
        _kpi_card("5th-pct max drawdown", _pct(kpis.get("p5_drawdown"))),
        direction="row", gap="size-100", wrap=True,
    )

    fig_content = run["fig"] if ready else ui.text(placeholder)

    return ui.column(
        ui.row(
            ui.column(ui.panel(controls, title="Controls"), width=22),
            ui.panel(
                kpi_row,
                title=f"{ticker} · {STRATEGY_LABELS[strategy]} over sampled futures "
                      f"(${CAPITAL:,.0f})",
            ),
            height=30,
        ),
        ui.row(ui.panel(fig_content, title="Probability cone"), height=70),
    )


# Binding the dashboard renders it in the IDE.
dashboard = ui.dashboard(fan_chart())
