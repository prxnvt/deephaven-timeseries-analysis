"""Market-data layer: download, cache, and slice the ticker universe.

Pure pandas/yfinance (no Deephaven imports — see the package docstring). Shared
by the Deephaven dashboard scripts in the container and by host-side tooling.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "UNH",
    "XOM", "JNJ", "WMT", "PG", "MA", "HD", "KO", "PEP", "COST", "CVX",
]
HISTORY_START = "2014-01-01"
HISTORY_END = "2024-01-01"
MIN_ROWS = 200  # drop a ticker with less history than this


def download_universe(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = HISTORY_START,
    end: str = HISTORY_END,
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """Download the ticker basket and reshape to tidy long form.

    Columns: Date, Ticker, Open, High, Low, Close, Volume.
    """
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, group_by="ticker", progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check network access.")
    # group_by='ticker' yields a (Ticker, Field) column MultiIndex; stack level 0.
    long_df = (
        raw.stack(level=0, future_stack=True)
        .rename_axis(["Date", "Ticker"])
        .reset_index()
    )
    long_df = long_df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    long_df = long_df.dropna(subset=["Close"])
    long_df["Ticker"] = long_df["Ticker"].astype(str)
    # Drop tickers with too little history (keeps the basket clean / picker honest).
    counts = long_df.groupby("Ticker")["Close"].transform("size")
    long_df = long_df[counts >= min_rows].reset_index(drop=True)
    return long_df


def load_universe(
    cache_path: str,
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = HISTORY_START,
    end: str = HISTORY_END,
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """Load the universe from a local parquet cache, else download and cache it."""
    try:
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            return cached
    except Exception:
        pass
    long_df = download_universe(tickers, start, end, min_rows)
    try:
        long_df.to_parquet(cache_path, index=False)
    except Exception:
        pass  # caching is best-effort; never block on it
    return long_df


def slice_universe(
    prices_long: pd.DataFrame,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """One ticker's rows within [start, end], date-sorted with a fresh index."""
    return (
        prices_long[
            (prices_long["Ticker"] == ticker)
            & (prices_long["Date"] >= start)
            & (prices_long["Date"] <= end)
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )
