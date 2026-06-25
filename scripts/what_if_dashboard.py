"""INTC stock-market "what-if" profit simulator (Deephaven what-if dashboard).

Implements docs/SPEC.md section 3 (Core Logic & Dashboard Generation):

    1. Data acquisition  - download ~10 years of Intel (INTC) history via yfinance.
    2. Data integration  - convert the pandas DataFrame into a Deephaven table.
    3. Scenario engine    - keep days where Close <= buy_price and compute
                            Simulated_Profit = sell_price - Close.
    4. UI construction    - a reactive deephaven.ui dashboard whose two sliders
                            drive buy/sell prices; the results table recomputes live.

Run it from the Deephaven IDE: ./scripts is mounted to /data/storage/notebooks,
so this file appears in the IDE "Notebooks" panel. Executing it binds `dashboard`,
which the IDE renders automatically.

Note: this script imports `deephaven.*`, so it only runs inside the Deephaven
engine (the IDE console or an application), not under a plain `python` process.
"""

import pandas as pd
import yfinance as yf

import deephaven.pandas as dhpd
from deephaven import ui

# --- Configuration -----------------------------------------------------------
TICKER = "INTC"
START = "2014-01-01"  # ~10 years of history (per the spec example window)
END = "2024-01-01"

# Default scenario assumptions, also the sliders' starting positions.
DEFAULT_BUY_PRICE = 25.0
DEFAULT_SELL_PRICE = 45.0


# --- 1. Data acquisition -----------------------------------------------------
def load_intel_prices() -> pd.DataFrame:
    """Download 10y of INTC daily bars as a flat, Deephaven-ready DataFrame."""
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {TICKER} ({START}..{END}). "
            "Check the container's network access and the ticker symbol."
        )

    # yfinance returns a (field, ticker) column MultiIndex even for one ticker,
    # but deephaven.pandas.to_table needs flat string column names. Drop the
    # ticker level, then reset the DatetimeIndex into a plain 'Date' column.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df.columns = [str(c) for c in df.columns]
    df.columns.name = None
    # reset_index() turns the DatetimeIndex into the first column; pin its name
    # to 'Date' so the engine queries resolve even if yfinance renames the index.
    df = df.rename(columns={df.columns[0]: "Date"})
    missing = {"Date", "Close"} - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Unexpected columns from yfinance: {list(df.columns)} (missing {missing})."
        )
    return df


intel_df = load_intel_prices()

# --- 2. Data integration -----------------------------------------------------
# Columns: Date (datetime), Close, High, Low, Open, Volume.
intel_table = dhpd.to_table(intel_df)


# --- 3. Scenario engine ------------------------------------------------------
def calculate_profit(buy_price: float, sell_price: float):
    """Days where Close <= buy_price, with Simulated_Profit = sell_price - Close.

    Profit is modelled (per the spec) as sell_price - Close: the best-case fill
    at each qualifying day's closing price, not sell_price - buy_price. Prices
    are formatted to fixed decimals so the interpolated query is always a plain
    decimal literal (never scientific notation), regardless of slider range.
    """
    return intel_table.where(f"Close <= {buy_price:.4f}").update(
        f"Simulated_Profit = {sell_price:.4f} - Close"
    )


# --- 4. UI construction ------------------------------------------------------
@ui.component
def stock_dashboard():
    # Interactive state for the buy/sell prices.
    buy_price, set_buy_price = ui.use_state(DEFAULT_BUY_PRICE)
    sell_price, set_sell_price = ui.use_state(DEFAULT_SELL_PRICE)

    # Recompute the scenario table only when a slider value actually changes.
    result_table = ui.use_memo(
        lambda: calculate_profit(buy_price, sell_price),
        [buy_price, sell_price],
    )
    # Memoize the (static) row count alongside the table so the blocking .size
    # read happens on dependency change, keeping the render path itself light.
    qualifying_days = ui.use_memo(lambda: result_table.size, [result_table])

    return ui.flex(
        ui.heading(f"{TICKER} What-If Profit Simulator", level=2),
        ui.text(
            f"Buy when Close ≤ ${buy_price:,.2f}  •  "
            f"Sell at ${sell_price:,.2f}  →  "
            f"{qualifying_days:,} qualifying day(s)"
        ),
        ui.slider(
            label="Buy price (fill when Close ≤)",
            value=buy_price,
            on_change=set_buy_price,
            min_value=10.0,
            max_value=50.0,
            step=0.5,
        ),
        ui.slider(
            label="Sell price",
            value=sell_price,
            on_change=set_sell_price,
            min_value=20.0,
            max_value=100.0,
            step=0.5,
        ),
        ui.table(result_table),
        direction="column",
        gap="size-150",
    )


# Binding the component renders the dashboard in the IDE.
dashboard = stock_dashboard()
