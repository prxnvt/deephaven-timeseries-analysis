"""Monte-Carlo robustness harness: a strategy's OUTCOME DISTRIBUTION, not one verdict.

Backtesting against the single real history answers "what happened"; running the
same strategy over N generated histories answers "how typical was that" — the
percentile band of final values / drawdowns, and where the real result lands in
it. No Deephaven imports; the fan-chart dashboard reuses these helpers via the
container's torch install.

CLI:
    .venv/bin/python -m marketlab.mc --ticker AAPL --strategy sma --n 500
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from marketlab.backtest import run_backtest
from marketlab.data import load_universe
from marketlab.sample import generate, load_artifacts, to_price_paths

PERCENTILES = (5, 25, 50, 75, 95)

DEFAULT_PARAMS = {
    "sma": {"fast": 20, "slow": 100},
    "dip": {"drop_pct": 10.0, "recover_pct": 15.0, "lookback": 60},
    "dca": {"amount": 500.0, "every_n": 21},
}


def synthetic_price_frame(returns: np.ndarray, start_price: float,
                          start_date: str = "2024-01-01") -> pd.DataFrame:
    """One generated return path -> the per-date frame run_backtest expects."""
    closes = to_price_paths(returns[None, :], start_price)[0]
    return pd.DataFrame({
        "Date": pd.bdate_range(start_date, periods=len(closes)),
        "Ticker": "SYNTH",
        "Close": closes,
    })


def mc_backtest(paths: np.ndarray, start_price: float, strategy: str,
                params: dict, capital: float) -> pd.DataFrame:
    """Run one strategy over every (n_paths, T) generated path; per-path stats."""
    rows = []
    for i, path_returns in enumerate(paths):
        frame = synthetic_price_frame(path_returns, start_price)
        _, _, stats = run_backtest(frame, strategy, params, capital)
        rows.append({"path": i, **{k: v for k, v in stats.items() if k != "win_rate"}})
    return pd.DataFrame(rows)


def summarize(per_path: pd.DataFrame, real_stats: dict | None = None) -> pd.DataFrame:
    """Percentile table over paths, with the real-history result as a column."""
    metrics = ["final_value", "total_return", "vs_bh", "max_drawdown"]
    out = {"metric": metrics}
    for p in PERCENTILES:
        out[f"p{p}"] = [float(np.percentile(per_path[m], p)) for m in metrics]
    if real_stats is not None:
        out["real"] = [float(real_stats[m]) for m in metrics]
        out["real_pctile"] = [
            float((per_path[m] < real_stats[m]).mean() * 100.0) for m in metrics
        ]
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--cache", default="data/universe_cache.parquet")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--strategy", default="sma", choices=sorted(DEFAULT_PARAMS))
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=750)  # ~3 trading years
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    model, tokenizer, meta = load_artifacts(args.artifacts)
    ticker_id = meta["tickers"].index(args.ticker)
    prices = load_universe(args.cache)
    real = prices[prices["Ticker"] == args.ticker].sort_values("Date").reset_index(drop=True)
    real_window = real.tail(args.horizon).reset_index(drop=True)
    start_price = float(real_window["Close"].iloc[0])
    params = DEFAULT_PARAMS[args.strategy]

    _, _, real_stats = run_backtest(real_window, args.strategy, params, args.capital)

    print(f"generating {args.n} x {args.horizon}d paths for {args.ticker} ...", flush=True)
    paths = generate(model, tokenizer, ticker_id, args.n, args.horizon,
                     temperature=args.temperature, seed=args.seed)
    per_path = mc_backtest(paths, start_price, args.strategy, params, args.capital)
    table = summarize(per_path, real_stats)

    print(f"\n{args.strategy} on {args.ticker}: real last-{args.horizon}d result vs "
          f"{args.n} synthetic histories (capital ${args.capital:,.0f})")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(table.to_string(index=False))
    print("\nreal_pctile = share of synthetic outcomes below the real result.")


if __name__ == "__main__":
    main()
