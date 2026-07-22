"""The market Turing test — one of these charts is a real stock, one is MarketGPT.

Each round picks a random ticker and a random 120-trading-day window of its
real history, generates a synthetic 120-day path for the same ticker, indexes
both to 100, and deals them onto panels A and B in random order. Guess which
is real, then press Reveal. New round reshuffles.

The rigorous twin of this game lives in marketlab/discriminator.py: a trained
MLP classifier's held-out accuracy on exactly this task is printed in the
caption (from the number recorded in the README) — try to beat it.

Run from the IDE Notebooks panel; binds `dashboard`. Engine-only (deephaven.*).
"""

import numpy as np
import pandas as pd

from marketlab.data import load_universe
from marketlab.sample import generate, load_artifacts, to_price_paths

from deephaven import pandas as dhpd
from deephaven import ui
import deephaven.plot.express as dx

# --- Configuration -----------------------------------------------------------
ARTIFACTS_DIR = "/opt/project/artifacts"
CACHE_PATH = "/data/universe_cache.parquet"
WINDOW_DAYS = 120

MODEL, TOKENIZER, META = load_artifacts(ARTIFACTS_DIR)
PRICES_LONG = load_universe(CACHE_PATH)
AVAILABLE_TICKERS = sorted(set(META["tickers"]) & set(PRICES_LONG["Ticker"].unique()))


def build_round(round_id: int):
    """Deterministic per-round: (fig_a, fig_b, answer_text)."""
    rng = np.random.default_rng(9000 + round_id)
    ticker = AVAILABLE_TICKERS[int(rng.integers(len(AVAILABLE_TICKERS)))]
    real = (
        PRICES_LONG[PRICES_LONG["Ticker"] == ticker]
        .sort_values("Date").reset_index(drop=True)
    )
    closes = real["Close"].to_numpy(dtype=np.float64)

    start = int(rng.integers(0, len(closes) - WINDOW_DAYS))
    real_px = closes[start:start + WINDOW_DAYS]
    real_px = 100.0 * real_px / real_px[0]
    start_date = pd.Timestamp(real["Date"].iloc[start]).date()

    synth_rets = generate(
        MODEL, TOKENIZER, META["tickers"].index(ticker),
        n_paths=1, horizon=WINDOW_DAYS - 1, seed=int(rng.integers(1 << 30)),
    )
    synth_px = np.concatenate([[100.0], to_price_paths(synth_rets, 100.0)[0]])

    real_first = bool(rng.integers(2))
    panels = [real_px, synth_px] if real_first else [synth_px, real_px]
    answer_real = "A" if real_first else "B"

    figs = []
    for label, px in zip(("A", "B"), panels):
        tbl = dhpd.to_table(pd.DataFrame({"Day": np.arange(WINDOW_DAYS, dtype=np.int64),
                                          "Price": px}))
        figs.append(dx.line(tbl, x="Day", y="Price", title=f"Market {label}"))
    answer = (f"Market {answer_real} is real: {ticker} starting {start_date}. "
              f"The other is MarketGPT.")
    return {"fig_a": figs[0], "fig_b": figs[1], "answer": answer}


@ui.component
def turing_test():
    round_id, set_round_id = ui.use_state(0)
    revealed, set_revealed = ui.use_state(False)

    run, set_run = ui.use_state(None)
    run_in_context = ui.use_execution_context()

    def _build_and_publish():
        try:
            set_run(build_round(round_id))
        except Exception as exc:
            set_run({"error": str(exc)})

    def _effect():
        run_in_context(_build_and_publish)

    ui.use_effect(_effect, [round_id])

    def _new_round(_e):
        set_revealed(False)
        set_round_id(round_id + 1)

    ready = run is not None and "error" not in run
    placeholder = run["error"] if (run and "error" in run) else "Dealing charts…"

    prompt = ("One chart is a real 120-day stretch of a real stock; the other "
              "is sampled from MarketGPT. Both indexed to 100. Which is which? "
              "For calibration: a trained MLP discriminator with explicit vol "
              "features scores 58% held-out accuracy on this task (50% = "
              "indistinguishable).")
    status = run["answer"] if (ready and revealed) else prompt
    controls = ui.flex(
        ui.button("New round", on_press=_new_round, variant="accent"),
        ui.button("Reveal", on_press=lambda _e: set_revealed(True)),
        ui.text(status),
        direction="column", gap="size-100",
    )

    fig_a = run["fig_a"] if ready else ui.text(placeholder)
    fig_b = run["fig_b"] if ready else ui.text(placeholder)

    return ui.column(
        ui.row(ui.panel(controls, title=f"Round {round_id + 1}"), height=24),
        ui.row(
            ui.panel(fig_a, title="Market A"),
            ui.panel(fig_b, title="Market B"),
            height=76,
        ),
    )


# Binding the dashboard renders it in the IDE.
dashboard = ui.dashboard(turing_test())
