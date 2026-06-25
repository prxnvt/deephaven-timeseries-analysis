# Deephaven Time-Series Analysis — Stock "What-If" Engine

A local, containerized [Deephaven](https://deephaven.io) stack for running
stock-market **"what-if"** scenarios over historical price data. Historical
prices are pulled for free from Yahoo Finance via
[`yfinance`](https://github.com/ranaroussi/yfinance), loaded into a live
Deephaven table, and explored through a reactive `deephaven.ui` dashboard —
no JavaScript or CSS required.

> **Status:** scaffolding only. The infrastructure (Docker, Compose, deps, git)
> is ready; the dashboard logic in `scripts/` is not yet implemented.

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
└── scripts/              # Python scripts — mounted into the IDE "Notebooks" panel
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

## Building the dashboard

The scenario + dashboard logic is intended to live in
`scripts/what_if_dashboard.py`. Because `./scripts` is mounted to
`/data/storage/notebooks`, any file you drop there shows up in the IDE's
**Notebooks** panel, ready to run against the engine.

Intended flow (per the [project spec](docs/SPEC.md)):

1. **Data acquisition** — download ~10 years of Intel (`INTC`) history with
   `yfinance`, then reset the DataFrame index.
2. **Integration** — convert the pandas DataFrame into a live Deephaven table
   via `deephaven.pandas.to_table`.
3. **Scenario engine** — keep rows where `Close <= buy_price`, then add a
   `Simulated_Profit = sell_price - Close` column.
4. **UI** — an `@ui.component` using `ui.use_state` for buy/sell prices,
   `ui.slider` controls, `ui.use_memo` to recompute the table reactively, all
   returned inside a `ui.flex` layout alongside `ui.table`.

## Notes

- `data/` is git-ignored: it holds the local Deephaven data root and any
  downloaded data, and is recreated at runtime.
- `pandas` is intentionally left unpinned in `requirements.txt` so it defers to
  the version bundled in the Deephaven base image.

## Roadmap

- **Now:** static historical backtests.
- **Later (optional):** stream live, ticking market data (e.g. Alpaca, Alpha
  Vantage) into the engine for real-time scenarios.
