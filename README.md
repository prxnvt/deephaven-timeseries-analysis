# Deephaven Time-Series Analysis — Stock "What-If" Engine

A local, containerized [Deephaven](https://deephaven.io) stack for running
stock-market **"what-if"** scenarios over historical price data. Historical
prices are pulled for free from Yahoo Finance via
[`yfinance`](https://github.com/ranaroussi/yfinance), loaded into a live
Deephaven table, and explored through a reactive `deephaven.ui` dashboard —
no JavaScript or CSS required.

> **Status:** working. The full stack builds and runs, and the what-if dashboard
> in [`scripts/what_if_dashboard.py`](scripts/what_if_dashboard.py) is implemented
> and verified end-to-end against the live engine (Deephaven 41.7).

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
    └── what_if_dashboard.py   # the INTC what-if dashboard (run this in the IDE)
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

## Using the dashboard

The what-if dashboard lives in
[`scripts/what_if_dashboard.py`](scripts/what_if_dashboard.py). Because
`./scripts` is mounted to `/data/storage/notebooks`, it appears in the IDE's
**Notebooks** panel. To run it:

1. Open the IDE (<http://localhost:10000/ide>) and authenticate.
2. In the **Notebooks** file browser, open `what_if_dashboard.py` and run it.
3. A `dashboard` panel renders: two sliders (buy / sell price) above a live
   results table. Drag a slider and the table + qualifying-day count recompute
   instantly.

What it does (per the [project spec](docs/SPEC.md)):

1. **Data acquisition** — downloads ~10 years of Intel (`INTC`) history with
   `yfinance` and flattens its `(field, ticker)` column MultiIndex.
2. **Integration** — converts the pandas DataFrame into a Deephaven table via
   `deephaven.pandas.to_table`.
3. **Scenario engine** — `calculate_profit(buy, sell)` keeps rows where
   `Close <= buy_price` and adds `Simulated_Profit = sell_price - Close`.
4. **UI** — an `@ui.component` with `ui.use_state` buy/sell prices, two
   `ui.slider` controls, `ui.use_memo` for reactive recompute, and a `ui.flex`
   layout holding the sliders and `ui.table`.

> Runs entirely on free historical data — no API keys. The script needs outbound
> internet (from the container) for the one-time `yfinance` download.

## Notes

- `data/` is git-ignored: it holds the local Deephaven data root and any
  downloaded data, and is recreated at runtime.
- `pandas` is intentionally left unpinned in `requirements.txt` so it defers to
  the version bundled in the Deephaven base image.

## Roadmap

- **Now:** static historical backtests.
- **Later (optional):** stream live, ticking market data (e.g. Alpaca, Alpha
  Vantage) into the engine for real-time scenarios.
