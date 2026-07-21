# Deephaven Time-Series Analysis — Stock "What-If" Engine

A local, containerized [Deephaven](https://deephaven.io) stack for stock-market
**"what-if"** analysis over historical price data. Prices are pulled for free from
Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance), loaded into
live Deephaven tables, and explored through reactive `deephaven.ui` dashboards — no
JavaScript or CSS required. Three dashboards ship, backed by a tested pure-Python
engine package (`marketlab`) that also trains a small generative model of returns:

- **[`market_sim_dashboard.py`](scripts/market_sim_dashboard.py)** — a full trading
  **simulator**: pick one of ~20 free-data tickers and a strategy, set capital and
  parameters, and watch it trade against an *unfolding* historical market (animated
  with Deephaven `TableReplayer`) on a Sunbird-style multi-panel board.
- **[`fan_chart_dashboard.py`](scripts/fan_chart_dashboard.py)** — a **probability
  cone** over a ticker's next N trading days: a tiny transformer ("MarketGPT",
  trained on this universe — see below) is prompted with the last 60 real days and
  sampled for hundreds of continuations, with Monte-Carlo strategy KPIs over those
  same sampled futures.
- **[`what_if_dashboard.py`](scripts/what_if_dashboard.py)** — the simpler baseline:
  two price sliders over a single INTC table.

> **Status:** working. The stack builds and runs, and both dashboards are
> implemented and verified end-to-end against the live engine (Deephaven 41.7).

## Stack

| Layer        | Choice                                            |
| ------------ | ------------------------------------------------- |
| Engine / IDE | Deephaven Community Core (`ghcr.io/deephaven/server:latest`) |
| Data source  | `yfinance` (Yahoo Finance) — no API key, free     |
| UI           | `deephaven.ui` (Python-driven reactive components)|
| Backtesting  | `marketlab` — pure-Python engine package, pytest-covered |
| Generative ML| PyTorch — MarketGPT, a 21.6K-param causal transformer over return tokens |
| Runtime      | Docker + Docker Compose; host-side training via `uv` venv |

## Project structure

```
.
├── Dockerfile            # Deephaven server image + project Python deps + marketlab
├── docker-compose.yml    # Local service definition (web IDE on :10000)
├── requirements.txt      # container Python deps (yfinance, pandas)
├── requirements-dev.txt  # host-side dev deps (pytest, ruff, torch, ...)
├── pyproject.toml        # pytest/ruff configuration
├── .github/workflows/    # CI: ruff + pytest on every push
├── .dockerignore         # keeps the build context lean
├── .gitignore
├── docs/SPEC.md          # original project spec / requirements
├── data/                 # Deephaven data root — git-ignored, created at runtime
├── marketlab/            # pure-Python engine package (no deephaven imports)
│   ├── data.py           #   universe download / parquet cache / slicing
│   ├── backtest.py       #   strategy signals + portfolio accounting + run_backtest
│   ├── tokenizer.py      #   daily returns <-> 64 quantile-bin tokens
│   ├── model.py          #   MarketGPT: tiny causal transformer + ticker embedding
│   ├── train.py          #   temporal-split training w/ early stopping (host-side)
│   ├── sample.py         #   path generation: temperature, real-prefix conditioning
│   ├── evaluate.py       #   stylized-facts metrics + comparison figure
│   └── mc.py             #   Monte-Carlo: strategy outcomes over N synthetic paths
├── artifacts/            # committed trained checkpoint (~96 KB) + eval figure
├── tests/                # pytest suite for marketlab (hand-computed expectations)
└── scripts/              # mounted into the IDE "Notebooks" panel
    ├── market_sim_dashboard.py  # trading simulator (20 tickers, strategies, replay)
    └── what_if_dashboard.py     # simpler INTC buy/sell-price what-if
```

The split matters: everything path-dependent (signals, fills, equity, drawdown)
lives in `marketlab`, which never imports `deephaven.*` — so it runs headlessly
under pytest on the host, while the dashboard scripts stay thin Deephaven layers
over it. The container gets the package via `PYTHONPATH=/opt/project` plus a
live bind-mount, so package edits apply without a rebuild.

## Prerequisites

- **Docker Engine + Docker Compose v2** — verify with `docker compose version`.
- No API keys or cloud accounts required.

## Getting started

1. Build the custom image and start the stack:
   ```bash
   docker compose up -d
   ```
2. On first run, grab the auto-generated pre-shared key from the logs:
   ```bash
   docker compose logs deephaven | grep -i "pre-shared key"
   ```
3. Open the IDE at <http://localhost:10000/ide> and paste the key when prompted.
4. Stop the stack when done:
   ```bash
   docker compose down
   ```

> Tip: to skip the per-restart key lookup, pin a fixed PSK by uncommenting the
> `environment` block in `docker-compose.yml`.

## The market simulator

[`scripts/market_sim_dashboard.py`](scripts/market_sim_dashboard.py) is the
flagship — a trading simulator that backtests a strategy against an *unfolding*
historical market. Because `./scripts` is mounted to `/data/storage/notebooks`, it
appears in the IDE **Notebooks** panel; open it and run it to bind `dashboard`, a
multi-panel board.

- **Universe** — ~20 large-cap tickers (`AAPL`, `MSFT`, `NVDA`, …) downloaded once
  via one batched `yfinance` call and cached under `data/`.
- **Strategies** (dropdown) — SMA crossover, buy-the-dip, and dollar-cost
  averaging. Every trade fills at the day's market `Close` (no manual buy price).
- **Unfolding replay** — the chosen run's full backtest is precomputed, then
  revealed over a ~90s wall-clock window via `TableReplayer`; every panel updates
  live as the market "unfolds."
- **Controls** — ticker, strategy, initial capital, per-strategy parameters, date
  range, and a Restart button.
- **Panels** — six KPI cards (final value, total return, vs buy-&-hold, max
  drawdown, # buys, win rate), an equity curve vs buy-&-hold, price + MA overlays
  with buy/sell markers, an exposure (in-market vs cash) area chart, a live trade
  log, and a monthly-returns bar chart.

How it works: the path-dependent backtest is computed in the `marketlab` package
(signals, positions with a one-bar execution lag, equity, drawdown — pure
pandas/numpy, unit-tested), converted to a Deephaven table, given a compressed
`ReplayTime` column, and replayed. The charts
(`deephaven.plot.express`) and trade log bind to the replaying table and animate as
it unfolds; the KPI cards show the final backtest result. The `TableReplayer`
lifecycle is managed inside the `@ui.component` with `use_effect`/`use_ref`.

## The simple what-if dashboard

[`scripts/what_if_dashboard.py`](scripts/what_if_dashboard.py) is the original,
simpler baseline (per the [project spec](docs/SPEC.md)): ~10 years of Intel
(`INTC`) loaded via `deephaven.pandas.to_table`, with two `ui.slider` controls
driving `calculate_profit(buy, sell)` — it keeps days where `Close <= buy_price`
and adds `Simulated_Profit = sell_price - Close`, recomputing reactively.

> Both run entirely on free historical data — no API keys. They need outbound
> internet (from the container) for the one-time `yfinance` download.

## The generative layer: makemore for markets

`marketlab` also ships a small **generative model of daily returns** — the same
autoregressive recipe as Karpathy's makemore, transplanted: names are sequences
of characters, markets are sequences of return buckets. Daily log-returns are
tokenized into 64 quantile bins (fit on training years only), and a tiny causal
transformer (1 layer, 32-dim, 21.6K params, learned per-ticker embedding) is
trained to predict the next day's bucket. It is explicitly **not** a price
predictor — it's a distribution model used to generate *plausible synthetic
market histories* for robustness testing.

**Honesty bar.** Training reports validation cross-entropy against the
*marginal baseline* (predicting every day from the unconditional training
distribution). Bigger or lightly-regularized configs memorize 2014–2021 and
**lose** to that baseline out-of-sample on 2022–2023; the shipped config wins,
4.067 vs 4.159 nats/token — a small, real edge consistent with the fact that
daily returns are mostly noise plus volatility structure.

**Stylized-facts evaluation** (`python -m marketlab.evaluate`), synthetic vs
real AAPL:

![Stylized facts: real vs synthetic](artifacts/stylized_facts.png)

- Raw-return autocorrelation ~0 in both — the generator doesn't hallucinate
  predictability.
- **Volatility clustering is genuinely learned**: mean |return|-ACF +0.073
  synthetic vs +0.153 real (an iid generator scores ~0). Captured, but
  under-strength — stated as-is.
- Tails run thin (excess kurtosis 0.8 vs 5.7): decoding tokens to per-bin
  means caps extreme moves at the outer bins' averages. A known tokenizer
  trade-off, not a modeling win.
- Empirical surprise: sampling temperature acts as a **regime-persistence
  dial here, not a tail dial** — low temperature amplifies the model's
  self-exciting volatility feedback (realized vol rises), high temperature
  washes conditionals toward the iid marginal.

**Monte-Carlo robustness** (`python -m marketlab.mc`): run a strategy over
hundreds of generated histories and see where the real result lands. SMA(20/100)
on AAPL, last ~3y window, $10K, 500 synthetic paths:

| metric        | p5     | p50    | p95    | real   | real %ile |
|---------------|--------|--------|--------|--------|-----------|
| final value   | $8,046 | $19,528| $63,603| $12,159| 21%       |
| vs buy & hold | -59%   | -30%   | +12%   | -19%   | 71%       |
| max drawdown  | -50%   | -30%   | -18%   | -33%   | 39%       |

The takeaway a single backtest can't give you: SMA's underperformance vs
buy-and-hold on real AAPL was **not bad luck** — it underperforms in the large
majority of plausible histories too.

To retrain from scratch (a few minutes on Apple Silicon):

```bash
.venv/bin/python -m marketlab.train --cache data/universe_cache.parquet --out artifacts
```

## The fan-chart dashboard

[`scripts/fan_chart_dashboard.py`](scripts/fan_chart_dashboard.py) puts the
generative model behind an interactive Deephaven dashboard. The committed
checkpoint loads in-container (CPU torch); every press of **Generate** prompts
it with the chosen ticker's last 60 real trading days and samples hundreds of
continuation paths *live*, rendering:

- the 5/25/50/75/95 **percentile cone** joined to recent real history (below,
  the same data rendered for the README — the dashboard version is interactive);
- **Monte-Carlo KPI cards** for a chosen strategy run over those same sampled
  futures: P(beats buy & hold), median and 5th-percentile final value,
  5th-percentile max drawdown.

![Fan chart: AAPL continuations](artifacts/fan_chart.png)

Controls: ticker, strategy, horizon (30–250 days), sampling temperature, path
count. The caption says it plainly: the cone is a distribution, not a forecast —
the median line is not a price target. In-container sampling of 250 paths x 250
days takes a few seconds on Apple Silicon (native arm64 image).

## Development & testing

The engine package is developed and tested on the host — no container needed:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q        # unit tests (hand-computed expectations)
.venv/bin/python -m ruff check .     # lint
```

The same checks run in CI ([.github/workflows/ci.yml](.github/workflows/ci.yml))
on every push once the repo has a GitHub remote.

## Notes

- `data/` is git-ignored: it holds the local Deephaven data root and any
  downloaded data, and is recreated at runtime.
- `pandas` is intentionally left unpinned in `requirements.txt` so it defers to
  the version bundled in the Deephaven base image.

## Roadmap

- **Done:** historical backtests + an animated `TableReplayer` "unfolding market";
  tested `marketlab` engine package + CI; MarketGPT generative layer with
  stylized-facts evaluation and Monte-Carlo strategy robustness.
- **Done:** the fan-chart dashboard — live in-container sampling of prefix-
  conditioned continuations with Monte-Carlo strategy KPIs.
- **Later (optional):** GARCH/bootstrap baselines with VaR coverage backtesting,
  joint multi-ticker generation (correlation under stress), live ticking data
  (e.g. Alpaca, Alpha Vantage) into the engine.
