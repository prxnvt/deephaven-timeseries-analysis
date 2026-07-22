"""Live VaR monitor — the rung-2 bake-off replayed as a risk desk would see it.

The untouched 2023 test year unfolds through Deephaven's TableReplayer: four
models' 1-day VaR lines against realized returns, a breach blotter that grows
as exceptions happen, and a per-model scorecard (breach counts, rates, Basel
traffic-light zone) that ticks live via engine aggregations. The static pooled
leaderboard — the full-period "answer key" with Kupiec/Christoffersen p-values
— sits alongside.

Leak-free protocol: MarketGPT trained on <=2021 and early-stopped on 2022 only;
the classical models fit through 2022 (strictly more data); 2023 was seen by
nothing. Watching it run animates the Christoffersen story: the unconditional
models' flat VaR lines get caught by vol spikes (e.g. the March 2023 banking
scare), while GARCH and MarketGPT adapt.

All series are precomputed in `marketlab` (pure pandas + one batched MarketGPT
forward) and revealed by the replayer; a later milestone can swap the
precomputed columns for per-tick listener inference with the same panels.

Run from the IDE Notebooks panel; binds `dashboard`. Engine-only (deephaven.*).
"""

import numpy as np
import pandas as pd

from marketlab.data import load_universe
from marketlab.sample import load_artifacts
from marketlab.var_backtest import DEFAULT_FIT_END, leaderboard, monitor_frames

from deephaven import agg
from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx
from deephaven.replay import TableReplayer
from deephaven.time import to_j_instant

# --- Configuration -----------------------------------------------------------
ARTIFACTS_DIR = "/opt/project/artifacts"
CACHE_PATH = "/data/universe_cache.parquet"
LEVELS = {"95": 0.05, "99": 0.01}
DURATIONS = ("30", "60", "90")
MODEL_ORDER = ["iid_gaussian", "block_bootstrap", "garch11", "marketgpt"]

# --- Module-load state --------------------------------------------------------
BUNDLE = load_artifacts(ARTIFACTS_DIR)          # (model, tokenizer, meta)
PRICES_LONG = load_universe(CACHE_PATH)
_META = BUNDLE[2]
AVAILABLE_TICKERS = sorted(set(_META["tickers"]) & set(PRICES_LONG["Ticker"].unique()))
DEFAULT_TICKER = "AAPL" if "AAPL" in AVAILABLE_TICKERS else AVAILABLE_TICKERS[0]

# Full-period pooled coverage results (the "answer key"): computed once at
# module load (~10s: batched MarketGPT forwards over every ticker), so per-run
# rebuilds stay instant.
ANSWER_KEY_TBL = dhpd.to_table(
    leaderboard(PRICES_LONG, ARTIFACTS_DIR).round(
        {"breach_rate": 4, "kupiec_p": 4, "christoffersen_p": 4, "cc_p": 4}
    )
)


# --- Replay-clock helper ------------------------------------------------------
def _add_shared_replay_time(frames: list[pd.DataFrame], seconds: float):
    """Give every frame a ReplayTime keyed by DATE (frames may repeat dates),
    compressing the full date range into `seconds` of wall clock. Returns
    (stamped_frames, start_instant, end_instant)."""
    all_dates = np.array(sorted(pd.concat([f["Date"] for f in frames]).unique()))
    now_ns = pd.Timestamp.now(tz="UTC").value
    span_ns = int(seconds * 1e9)
    offsets = (np.linspace(0.0, 1.0, len(all_dates)) * span_ns).astype("int64")
    lookup = {d: pd.Timestamp(now_ns + o, tz="UTC") for d, o in zip(all_dates, offsets)}
    stamped = []
    for f in frames:
        f = f.copy()
        f["ReplayTime"] = f["Date"].map(lookup)
        stamped.append(f)
    start = to_j_instant(pd.Timestamp(now_ns, tz="UTC"))
    end = to_j_instant(pd.Timestamp(now_ns + span_ns + int(1e9), tz="UTC"))
    return stamped, start, end


# --- One monitoring run -------------------------------------------------------
def build_run(ticker, alpha, replay_seconds):
    wide_df, long_df = monitor_frames(
        PRICES_LONG, ticker, alpha, model_bundle=BUNDLE, fit_end=DEFAULT_FIT_END
    )
    (wide_df, long_df), r_start, r_end = _add_shared_replay_time(
        [wide_df, long_df], replay_seconds
    )
    wide_tbl = dhpd.to_table(wide_df)
    long_tbl = dhpd.to_table(long_df)

    replayer = TableReplayer(r_start, r_end)
    live_wide = replayer.add_table(wide_tbl, "ReplayTime")
    live_long = replayer.add_table(long_tbl, "ReplayTime")
    replayer.start()

    var_cols = [f"VaR_{m}" for m in MODEL_ORDER]
    var_lines = dx.line(
        live_wide, x="Date", y=["Return"] + var_cols,
        title=f"1-day VaR{int((1 - alpha) * 100)} forecasts vs realized returns",
    )
    try:
        markers = dx.scatter(
            live_long.where("Breach = 1 && (Model == `garch11` || Model == `marketgpt`)"),
            x="Date", y="Return", by="Model",
        )
        chart = dx.layer(var_lines, markers)
    except Exception:
        chart = var_lines

    blotter = (
        live_long.where("Breach = 1")
        .view(["Date", "Model", "Return", "VaR"])
        .reverse()  # newest exception on top, like a real blotter
    )

    scorecard = (
        live_long.agg_by(
            [agg.count_("Days"), agg.sum_(["Breaches = Breach"])], by=["Model"]
        )
        .update([
            "BreachRate = Breaches / (double) Days",
            f"Expected = {alpha}",
            "Per250 = Breaches * 250.0 / Days",
            "Zone = Per250 < 5 ? `green` : (Per250 < 10 ? `yellow` : `red`)",
        ])
        .sort("Model")
    )

    return {"chart": chart, "blotter": blotter, "scorecard": scorecard}, replayer


# --- The dashboard component ---------------------------------------------------
@ui.component
def var_monitor():
    ticker, set_ticker = ui.use_state(DEFAULT_TICKER)
    level, set_level = ui.use_state("95")
    duration, set_duration = ui.use_state("60")
    nonce, set_nonce = ui.use_state(0)

    run, set_run = ui.use_state(None)
    replayer_ref = ui.use_ref(None)
    run_in_context = ui.use_execution_context()

    config_key = (ticker, level, duration, nonce)

    def _build_and_publish():
        if replayer_ref.current is not None:
            try:
                replayer_ref.current.shutdown()
            except Exception:
                pass
            replayer_ref.current = None
        try:
            new_run, replayer = build_run(ticker, LEVELS[level], float(duration))
            replayer_ref.current = replayer
            set_run(new_run)
        except Exception as exc:  # degrade gracefully instead of throwing in render
            set_run({"error": str(exc)})

    def _effect():
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

    controls = ui.flex(
        ui.picker(
            *[ui.item(t, key=t) for t in AVAILABLE_TICKERS],
            selected_key=ticker, on_change=set_ticker, label="Ticker",
        ),
        ui.picker(
            *[ui.item(f"{k}% VaR", key=k) for k in LEVELS],
            selected_key=level, on_change=set_level, label="Confidence level",
        ),
        ui.picker(
            *[ui.item(f"{d}s replay", key=d) for d in DURATIONS],
            selected_key=duration, on_change=set_duration, label="Replay duration",
        ),
        ui.button("Restart replay", on_press=lambda _e: set_nonce(nonce + 1),
                  variant="accent"),
        ui.text(
            "Untouched 2023 test year unfolds live. Zone = Basel traffic light "
            "(green < 5 breaches / 250d, yellow < 10, red >= 10) — calibrated "
            "for the 99% level."
        ),
        direction="column", gap="size-100",
    )

    ready = run is not None and "error" not in run
    placeholder = run["error"] if (run and "error" in run) else "Computing VaR series…"

    def _content(key, as_table=False):
        if not ready:
            return ui.text(placeholder)
        return ui.table(run[key]) if as_table else run[key]

    return ui.column(
        ui.row(
            ui.column(ui.panel(controls, title="Controls"), width=22),
            ui.panel(_content("scorecard", as_table=True),
                     title=f"{ticker} · live scorecard ({level}% VaR)"),
            height=32,
        ),
        ui.row(ui.panel(_content("chart"), title="VaR vs realized (replaying)"),
               height=40),
        ui.row(
            ui.panel(_content("blotter", as_table=True), title="Breach blotter"),
            ui.panel(ui.table(ANSWER_KEY_TBL),
                     title="Answer key: full-period pooled coverage (all tickers)"),
            height=28,
        ),
    )


# Binding the dashboard renders it in the IDE.
dashboard = ui.dashboard(var_monitor())
