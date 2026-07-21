"""Unit tests for marketlab.data — yfinance reshaping (mocked), caching, slicing."""

import numpy as np
import pandas as pd
import pytest

import marketlab.data as data


def _fake_yf_frame(n_full: int = 10, n_short: int = 3) -> pd.DataFrame:
    """Mimic yf.download(group_by='ticker'): (Ticker, Field) MultiIndex columns,
    DatetimeIndex, and NaN padding where the short ticker has no data."""
    dates = pd.bdate_range("2020-01-01", periods=n_full, name="Date")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], fields])
    frame = pd.DataFrame(np.nan, index=dates, columns=cols)
    for field in fields:
        frame[("AAA", field)] = np.arange(1.0, n_full + 1.0)
        frame.loc[dates[:n_short], ("BBB", field)] = 50.0
    return frame


def test_download_universe_reshapes_and_filters(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: _fake_yf_frame())
    out = data.download_universe(["AAA", "BBB"], min_rows=5)
    assert list(out.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    # BBB has only 3 rows < min_rows=5 and is dropped; AAA's 10 rows survive.
    assert out["Ticker"].unique().tolist() == ["AAA"]
    assert len(out) == 10
    assert out["Close"].tolist() == list(np.arange(1.0, 11.0))


def test_download_universe_empty_raises(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        data.download_universe(["AAA"])


def test_load_universe_uses_cache_on_second_call(monkeypatch, tmp_path, mini_universe):
    cache = str(tmp_path / "universe.parquet")
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return mini_universe

    monkeypatch.setattr(data, "download_universe", fake_download)
    first = data.load_universe(cache)
    second = data.load_universe(cache)
    assert calls["n"] == 1  # second call served from parquet, not the network
    pd.testing.assert_frame_equal(first, second)


def test_slice_universe(mini_universe):
    dates = mini_universe[mini_universe["Ticker"] == "AAA"]["Date"]
    start, end = dates.iloc[10], dates.iloc[20]
    out = data.slice_universe(mini_universe, "AAA", start, end)
    assert (out["Ticker"] == "AAA").all()
    assert len(out) == 11  # inclusive bounds
    assert out.index.tolist() == list(range(11))  # fresh index
    assert out["Date"].is_monotonic_increasing
