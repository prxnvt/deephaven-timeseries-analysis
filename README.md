# Deephaven Time-Series Analysis — Stock "What-If" Engine

A local, containerized [Deephaven](https://deephaven.io) stack for stock-market
**"what-if"** analysis over historical price data. Prices are pulled for free from
Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance), loaded into
live Deephaven tables, and explored through reactive `deephaven.ui` dashboards — no
JavaScript or CSS required. Two dashboards ship:

- **[`market_sim_dashboard.py`](scripts/market_sim_dashboard.py)** — a full trading
  **simulator**: pick one of ~20 free-data tickers and a strategy, set capital and
  parameters, and watch it trade against an *unfolding* historical market (animated
  with Deephaven `TableReplayer`) on a Sunbird-style multi-panel board.
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
| Runtime      | Docker + Docker Compose                           |

## Project structure

```
.
├── Dockerfile            # Deephaven server image + project Python deps
├── docker-compose.yml    # Local service definition (web IDE on :10000)
├── requirements.txt      # yfinance, pandas
├── .dockerignore         # keeps the build context lean
├── .gitignore
├── docs/SPEC.md          # original project spec / requirements
├── data/                 # Deephaven data root — git-ignored, created at runtime
└── scripts/              # mounted into the IDE "Notebooks" panel
    ├── market_sim_dashboard.py  # trading simulator (20 tickers, strategies, replay)
    └── what_if_dashboard.py     # simpler INTC buy/sell-price what-if
```

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

How it works: the path-dependent backtest is computed in pandas/numpy (signals,
positions with a one-bar execution lag, equity, drawdown), converted to a Deephaven
table, given a compressed `ReplayTime` column, and replayed. The charts
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

## Notes

- `data/` is git-ignored: it holds the local Deephaven data root and any
  downloaded data, and is recreated at runtime.
- `pandas` is intentionally left unpinned in `requirements.txt` so it defers to
  the version bundled in the Deephaven base image.

## Roadmap

- **Done:** historical backtests + an animated `TableReplayer` "unfolding market."
- **Later (optional):** multi-ticker portfolios, more strategies, and streaming
  live ticking market data (e.g. Alpaca, Alpha Vantage) into the engine.
