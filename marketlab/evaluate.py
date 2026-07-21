"""Stylized-facts evaluation: does the generator reproduce how markets BEHAVE?

The checklist (Cont 2001) that matters for this model class:
  * near-zero autocorrelation of raw returns (markets are hard to predict);
  * positive, slowly-decaying autocorrelation of |returns| (vol clustering);
  * fat-tailed return distribution (excess kurtosis > 0).

Honesty note baked into the report: quantile-bin tokenization hands the model
the BULK of the marginal distribution for free — but decoding to per-bin means
CAPS every move at the outer bins' average, so synthetic tails come out thinner
than real ones (kurtosis is a known weakness of this tokenizer, not a win
either way). The earned result is the temporal one: |return| autocorrelation.

CLI:
    .venv/bin/python -m marketlab.evaluate --ticker AAPL --n 200 --horizon 1000
"""

from __future__ import annotations

import argparse

import numpy as np

from marketlab.data import load_universe
from marketlab.sample import generate, load_artifacts


def acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def mean_path_acf(paths: np.ndarray, max_lag: int = 20, absolute: bool = False) -> np.ndarray:
    """ACF averaged across sampled paths (never across path boundaries)."""
    rows = np.abs(paths) if absolute else paths
    return np.mean([acf(row, max_lag) for row in rows], axis=0)


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.mean(x**2)
    if var == 0.0:
        return 0.0
    return float(np.mean(x**4) / var**2 - 3.0)


def stylized_facts(real_returns: np.ndarray, synth_paths: np.ndarray,
                   max_lag: int = 20) -> dict:
    """Headline metrics comparing one real return series to (n_paths, T) synthetic."""
    return {
        "real_ret_acf_mean_abs": float(np.mean(np.abs(acf(real_returns, max_lag)))),
        "synth_ret_acf_mean_abs": float(np.mean(np.abs(mean_path_acf(synth_paths, max_lag)))),
        "real_vol_clustering": float(np.mean(acf(np.abs(real_returns), max_lag))),
        "synth_vol_clustering": float(np.mean(mean_path_acf(synth_paths, max_lag, absolute=True))),
        "real_excess_kurtosis": excess_kurtosis(real_returns),
        "synth_excess_kurtosis": excess_kurtosis(synth_paths.ravel()),
        "real_daily_vol": float(np.std(real_returns)),
        "synth_daily_vol": float(np.std(synth_paths)),
    }


def make_figure(real_returns: np.ndarray, synth_paths: np.ndarray, out_path: str,
                ticker: str, max_lag: int = 20) -> None:
    """2x2 comparison panel; matplotlib imported lazily (host-side only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(f"Stylized facts: real {ticker} vs MarketGPT synthetic", fontsize=13)

    ax = axes[0][0]
    real_px = 100.0 * np.exp(np.cumsum(real_returns))
    ax.plot(real_px, label=f"real {ticker}", lw=1.0)
    for i in range(min(3, len(synth_paths))):
        ax.plot(100.0 * np.exp(np.cumsum(synth_paths[i])), lw=0.8, alpha=0.7,
                label="synthetic" if i == 0 else None)
    ax.set_title("Price paths (indexed to 100)")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    bins = np.linspace(-0.1, 0.1, 80)
    ax.hist(real_returns, bins=bins, density=True, alpha=0.6, label="real")
    ax.hist(synth_paths.ravel(), bins=bins, density=True, alpha=0.6, label="synthetic")
    ax.set_yscale("log")
    ax.set_title("Daily return distribution (log density)")
    ax.legend(fontsize=8)

    lags = np.arange(1, max_lag + 1)
    ax = axes[1][0]
    ax.bar(lags - 0.2, acf(real_returns, max_lag), width=0.4, label="real")
    ax.bar(lags + 0.2, mean_path_acf(synth_paths, max_lag), width=0.4, label="synthetic")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("ACF of returns (should be ~0)")
    ax.set_xlabel("lag (days)")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.bar(lags - 0.2, acf(np.abs(real_returns), max_lag), width=0.4, label="real")
    ax.bar(lags + 0.2, mean_path_acf(synth_paths, max_lag, absolute=True), width=0.4,
           label="synthetic")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("ACF of |returns| — vol clustering (should be > 0, decaying)")
    ax.set_xlabel("lag (days)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--cache", default="data/universe_cache.parquet")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--figure", default="artifacts/stylized_facts.png")
    args = ap.parse_args(argv)

    model, tokenizer, meta = load_artifacts(args.artifacts)
    ticker_id = meta["tickers"].index(args.ticker)
    prices = load_universe(args.cache)
    close = (
        prices[prices["Ticker"] == args.ticker].sort_values("Date")["Close"].to_numpy()
    )
    real_returns = np.diff(np.log(close))

    print(f"generating {args.n} x {args.horizon}d synthetic paths for {args.ticker} ...",
          flush=True)
    synth = generate(model, tokenizer, ticker_id, args.n, args.horizon,
                     temperature=args.temperature, seed=args.seed)

    metrics = stylized_facts(real_returns, synth)
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"  {key:<{width}}  {value:+.4f}")
    print("  (bin-mean decoding truncates extreme tails, so synthetic kurtosis runs",
          "\n   low; the earned result is synth_vol_clustering > 0, tracking real)")

    make_figure(real_returns, synth, args.figure, args.ticker)
    print(f"figure written to {args.figure}")


if __name__ == "__main__":
    main()
