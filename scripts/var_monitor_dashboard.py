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

THE MODEL IS IN THE LOOP: the replayer streams only raw bars (the "exchange
feed"); a table listener reacts to each tick by advancing every model's state
incrementally (GARCH's variance recursion is an O(1) stream fold; MarketGPT
runs a real torch forward pass) via marketlab.stream.ForecastEngine, and
publishes forecasts + breach events through DynamicTableWriters into the
ticking tables the panels consume. Every VaR number on screen was computed
AFTER its bar arrived, and per-model inference latency is measured live.
Stream/batch parity is pinned by tests/test_stream.py.

Run from the IDE Notebooks panel; binds `dashboard`. Engine-only (deephaven.*).
"""

import numpy as np
import pandas as pd

from marketlab.baselines import BlockBootstrap, Garch11, IIDGaussian
from marketlab.data import load_universe
from marketlab.sample import load_artifacts
from marketlab.stream import (
    ForecastEngine,
    StreamingBootstrap,
    StreamingGarch11,
    StreamingIID,
    StreamingMarketGPT,
)
from marketlab.train import log_returns_by_ticker
from marketlab.var_backtest import DEFAULT_FIT_END, leaderboard

from deephaven import DynamicTableWriter, agg
from deephaven import dtypes as dht
from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx
import deephaven.table_listener as tl
from deephaven.replay import TableReplayer
from deephaven.time import to_j_instant

# --- Configuration -----------------------------------------------------------
ARTIFACTS_DIR = "/opt/project/artifacts"
CACHE_PATH = "/data/universe_cache.parquet"
LEVELS = {"95": 0.05, "99": 0.01}
DURATIONS = ("30", "60", "90")

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
def _to_instant(value):
    """Robustly convert a listener-delivered Date (numpy datetime64) to a Java
    Instant for DynamicTableWriter."""
    try:
        return to_j_instant(value)
    except Exception:
        return to_j_instant(pd.Timestamp(value, tz="UTC"))


def build_run(ticker, alpha, replay_seconds):
    """Stream raw 2023 bars through a replayer; a listener runs every model
    per tick via ForecastEngine and publishes forecasts + breach events.
    Returns (panel dict, cleanup_fn)."""
    dates, rets = log_returns_by_ticker(
        PRICES_LONG[PRICES_LONG["Ticker"] == ticker]
    )[ticker]
    closes = (
        PRICES_LONG[PRICES_LONG["Ticker"] == ticker]
        .sort_values("Date")["Close"].to_numpy(dtype=np.float64)[1:]  # align w/ rets
    )
    cutoff = np.datetime64(pd.Timestamp(DEFAULT_FIT_END))
    test_start = int(np.searchsorted(dates, cutoff, side="right"))
    warm_rets = rets[:test_start]

    # Fit classical models on the warmup window; wrap everything as streams.
    engine = ForecastEngine({
        "iid_gaussian": StreamingIID(IIDGaussian().fit(warm_rets)),
        "block_bootstrap": StreamingBootstrap(BlockBootstrap().fit(warm_rets)),
        "garch11": StreamingGarch11(Garch11().fit(warm_rets)),
        "marketgpt": StreamingMarketGPT(
            BUNDLE[0], BUNDLE[1], BUNDLE[2]["tickers"].index(ticker)
        ),
    })
    engine.warmup(warm_rets)  # GARCH folds history; MarketGPT primes its context

    # The "exchange feed": raw test-period bars on a compressed replay clock.
    raw_df = pd.DataFrame({
        "Date": pd.DatetimeIndex(dates[test_start:]),
        "Return": rets[test_start:],
        "Close": closes[test_start:],
    })
    (raw_df,), r_start, r_end = _add_shared_replay_time([raw_df], replay_seconds)
    replayer = TableReplayer(r_start, r_end)
    live_raw = replayer.add_table(dhpd.to_table(raw_df), "ReplayTime")

    fc_writer = DynamicTableWriter({
        "Date": dht.Instant, "Model": dht.string,
        "VaR95": dht.double, "VaR99": dht.double, "LatencyMs": dht.double,
    })
    br_writer = DynamicTableWriter({
        "Date": dht.Instant, "Model": dht.string,
        "Return": dht.double, "VaR": dht.double, "Level": dht.int64,
    })

    def _on_update(update, is_replay):
        added = update.added()
        if not added or "Return" not in added:
            return
        for i in range(len(added["Return"])):
            date = _to_instant(added["Date"][i])
            forecasts, breach_rows = engine.on_bar(date, float(added["Return"][i]))
            for b in breach_rows:
                br_writer.write_row(b["Date"], b["Model"], b["Return"], b["VaR"],
                                    b["Level"])
            for f in forecasts:
                fc_writer.write_row(f["Date"], f["Model"], f["VaR95"], f["VaR99"],
                                    f["LatencyMs"])

    handle = tl.listen(live_raw, _on_update)
    replayer.start()  # start AFTER the listener is registered: no missed bars

    forecasts_tbl = fc_writer.table
    breach_tbl = br_writer.table
    level = int(round((1.0 - alpha) * 100))
    var_col = f"VaR{level}"

    realized_line = dx.line(
        live_raw, x="Date", y="Return",
        title=f"1-day VaR{level} forecasts vs realized returns (computed per tick)",
    )
    var_lines = dx.line(forecasts_tbl, x="Date", y=var_col, by="Model")
    try:
        markers = dx.scatter(
            breach_tbl.where(f"Level = {level} && "
                             "(Model == `garch11` || Model == `marketgpt`)"),
            x="Date", y="Return", by="Model",
        )
        chart = dx.layer(realized_line, var_lines, markers)
    except Exception:
        chart = dx.layer(realized_line, var_lines)

    blotter = (
        breach_tbl.where(f"Level = {level}")
        .view(["Date", "Model", "Return", "VaR"])
        .reverse()  # newest exception on top, like a real blotter
    )

    scorecard = (
        forecasts_tbl.agg_by(
            [agg.count_("Days"), agg.avg(["AvgLatencyMs = LatencyMs"])],
            by=["Model"],
        )
        .natural_join(
            breach_tbl.where(f"Level = {level}").count_by("Breaches", by=["Model"]),
            on=["Model"],
        )
        .update([
            "Breaches = isNull(Breaches) ? 0 : Breaches",
            "BreachRate = Breaches / (double) Days",
            f"Expected = {alpha}",
            "Per250 = Breaches * 250.0 / Days",
            "Zone = Per250 < 5 ? `green` : (Per250 < 10 ? `yellow` : `red`)",
        ])
        .sort("Model")
    )

    def _cleanup_run():
        for step in (handle.stop, replayer.shutdown, fc_writer.close, br_writer.close):
            try:
                step()
            except Exception:
                pass

    return {"chart": chart, "blotter": blotter, "scorecard": scorecard}, _cleanup_run


# --- The dashboard component ---------------------------------------------------
@ui.component
def var_monitor():
    ticker, set_ticker = ui.use_state(DEFAULT_TICKER)
    level, set_level = ui.use_state("95")
    duration, set_duration = ui.use_state("60")
    nonce, set_nonce = ui.use_state(0)

    run, set_run = ui.use_state(None)
    cleanup_ref = ui.use_ref(None)  # tears down listener + replayer + writers
    run_in_context = ui.use_execution_context()

    config_key = (ticker, level, duration, nonce)

    def _teardown_current():
        if cleanup_ref.current is not None:
            try:
                cleanup_ref.current()
            except Exception:
                pass
            cleanup_ref.current = None

    def _build_and_publish():
        _teardown_current()
        try:
            new_run, cleanup_fn = build_run(ticker, LEVELS[level], float(duration))
            cleanup_ref.current = cleanup_fn
            set_run(new_run)
        except Exception as exc:  # degrade gracefully instead of throwing in render
            set_run({"error": str(exc)})

    def _effect():
        run_in_context(_build_and_publish)
        return _teardown_current

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
