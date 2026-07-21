"""Shared fixtures: deterministic in-memory price data (no network, no binaries)."""

import numpy as np
import pandas as pd
import pytest


def make_price_frame(closes, ticker: str = "TEST", start: str = "2020-01-01") -> pd.DataFrame:
    """Tidy per-date frame (schema of data.slice_universe output) from a close series."""
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Open": closes,
        "High": closes,
        "Low": closes,
        "Close": closes,
        "Volume": 1_000_000.0,
    })


@pytest.fixture
def price_frame():
    return make_price_frame


@pytest.fixture
def mini_universe() -> pd.DataFrame:
    """Two tickers x 300 business days of fixed-seed random-walk prices, long form."""
    rng = np.random.default_rng(7)
    frames = []
    for ticker in ("AAA", "BBB"):
        rets = rng.normal(0.0005, 0.02, 300)
        closes = 100.0 * np.exp(np.cumsum(rets))
        frames.append(make_price_frame(closes, ticker))
    return pd.concat(frames, ignore_index=True)
