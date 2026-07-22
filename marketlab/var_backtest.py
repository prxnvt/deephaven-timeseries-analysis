"""VaR backtesting: which scenario generator should a risk desk trust?

Every generator forecasts a 1-day return quantile (VaR) for each held-out day,
conditioned only on information through the prior day; we then count breaches
(realized return below the forecast quantile) and apply the standard coverage
tests:

  * Kupiec POF (unconditional coverage): is the breach RATE consistent with
    the target alpha? chi-square(1).
  * Christoffersen independence: do breaches CLUSTER? A model can pass Kupiec
    while piling all its breaches into one volatile month — the signature of
    missing vol dynamics. chi-square(1) on first-order Markov transitions.
  * Conditional coverage: both at once (LR_pof + LR_ind, chi-square(2)).
  * Basel traffic light: the regulatory bucketing of 99% VaR models by breach
    count per 250 trading days (green < 5, yellow 5-9, red >= 10).

Protocol (leak-free): MarketGPT trains on <= 2021 and early-stops on 2022 only;
the classical models fit on everything <= --fit-end (2022-12-31 by default —
strictly MORE data than MarketGPT's gradients ever saw); all models are then
tested on 2023, which nothing was trained, fit, or selected on.

CLI:
    .venv/bin/python -m marketlab.var_backtest --cache data/universe_cache.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from marketlab.baselines import BlockBootstrap, Garch11, IIDGaussian, MarketGPTGenerator
from marketlab.data import load_universe
from marketlab.train import log_returns_by_ticker

DEFAULT_TRAIN_END = "2021-12-31"   # MarketGPT's gradient-training boundary
DEFAULT_FIT_END = "2022-12-31"     # classical models fit through here; test = 2023+
ALPHAS = (0.05, 0.01)
MIN_TEST_OBS = 100                 # skip tickers with less held-out history


# --- Breaches and coverage statistics -----------------------------------------
def breaches(realized: np.ndarray, var_series: np.ndarray) -> np.ndarray:
    return np.asarray(realized) < np.asarray(var_series)


def kupiec_pof(n_obs: int, n_breaches: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test. Returns (LR, p-value)."""
    x, n = n_breaches, n_obs
    pi = x / n
    ll_null = xlogy(n - x, 1.0 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(n - x, 1.0 - pi) + xlogy(x, pi)
    lr = float(-2.0 * (ll_null - ll_alt))
    return lr, float(stats.chi2.sf(lr, df=1))


def markov_counts(breach_series: np.ndarray) -> tuple[int, int, int, int]:
    """(n00, n01, n10, n11) transition counts over consecutive days."""
    b = np.asarray(breach_series, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen_independence(
    counts: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Christoffersen LR test of breach independence from Markov transition
    counts (pool tickers by SUMMING counts, not concatenating series)."""
    n00, n01, n10, n11 = counts
    if (n01 + n11) == 0:  # no breaches at all: independence is vacuous
        return float("nan"), float("nan")
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll_null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01)
              + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11))
    lr = float(-2.0 * (ll_null - ll_alt))
    return lr, float(stats.chi2.sf(lr, df=1))


def conditional_coverage(n_obs: int, counts: tuple[int, int, int, int],
                         alpha: float) -> tuple[float, float]:
    """Joint test: right rate AND independent breaches. chi-square(2)."""
    n_breaches = counts[1] + counts[3]
    lr_pof, _ = kupiec_pof(n_obs, n_breaches, alpha)
    lr_ind, p_ind = christoffersen_independence(counts)
    if np.isnan(lr_ind):
        return float("nan"), float("nan")
    lr = lr_pof + lr_ind
    return lr, float(stats.chi2.sf(lr, df=2))


def basel_traffic_light(n_breaches: int, n_obs: int) -> str:
    """Regulatory zones for 99% VaR, scaled to breaches per 250 trading days."""
    per_250 = n_breaches * 250.0 / n_obs
    if per_250 < 5.0:
        return "green"
    if per_250 < 10.0:
        return "yellow"
    return "red"


# --- The bake-off --------------------------------------------------------------
def build_generators(train_returns: np.ndarray, gpt: MarketGPTGenerator | None) -> dict:
    gens = {
        "iid_gaussian": IIDGaussian().fit(train_returns),
        "block_bootstrap": BlockBootstrap().fit(train_returns),
        "garch11": Garch11().fit(train_returns),
        "garch11_t": Garch11(dist="t").fit(train_returns),
    }
    if gpt is not None:
        gens["marketgpt"] = gpt
    return gens


def leaderboard(
    prices_long: pd.DataFrame,
    artifacts_dir: str | None,
    fit_end: str = DEFAULT_FIT_END,
    alphas: tuple[float, ...] = ALPHAS,
) -> pd.DataFrame:
    """Pooled coverage results per (model, alpha) across all usable tickers.

    Classical models fit on returns <= fit_end; every model is tested strictly
    after it. MarketGPT stays frozen (trained <= 2021, selected on 2022), so
    with the default fit_end the baselines see strictly more fitting data.
    """
    per_ticker = log_returns_by_ticker(prices_long)
    cutoff = np.datetime64(pd.Timestamp(fit_end))

    if artifacts_dir is not None:
        from marketlab.sample import load_artifacts
        model, tokenizer, meta = load_artifacts(artifacts_dir)
    else:
        model = tokenizer = meta = None

    model_names: list[str] = []
    # accumulators: (model, alpha) -> dict(n, x, counts)
    acc: dict[tuple[str, float], dict] = {}

    for ticker, (dates, rets) in sorted(per_ticker.items()):
        test_start = int(np.searchsorted(dates, cutoff, side="right"))
        n_test = len(rets) - test_start
        if test_start < MIN_TEST_OBS or n_test < MIN_TEST_OBS:
            continue
        if meta is not None and ticker in meta["tickers"]:
            gpt = MarketGPTGenerator(model, tokenizer, meta["tickers"].index(ticker))
        else:
            gpt = None
        gens = build_generators(rets[:test_start], gpt)
        model_names = list(gens)
        realized = rets[test_start:]
        for name, gen in gens.items():
            for alpha in alphas:
                var_series = gen.var_series(rets, test_start, alpha)
                b = breaches(realized, var_series)
                slot = acc.setdefault((name, alpha), {"n": 0, "x": 0, "c": np.zeros(4, int)})
                slot["n"] += len(b)
                slot["x"] += int(b.sum())
                slot["c"] += np.array(markov_counts(b))

    rows = []
    for name in model_names:
        for alpha in alphas:
            slot = acc[(name, alpha)]
            counts = tuple(int(v) for v in slot["c"])
            _, p_pof = kupiec_pof(slot["n"], slot["x"], alpha)
            _, p_ind = christoffersen_independence(counts)
            _, p_cc = conditional_coverage(slot["n"], counts, alpha)
            rows.append({
                "model": name,
                "alpha": alpha,
                "n_obs": slot["n"],
                "n_breaches": slot["x"],
                "breach_rate": slot["x"] / slot["n"],
                "expected": alpha,
                "kupiec_p": p_pof,
                "christoffersen_p": p_ind,
                "cc_p": p_cc,
                "traffic_light": basel_traffic_light(slot["x"], slot["n"])
                if alpha == 0.01 else "",
            })
    return pd.DataFrame(rows)


def monitor_frames(
    prices_long: pd.DataFrame,
    ticker: str,
    alpha: float,
    artifacts_dir: str | None = None,
    fit_end: str = DEFAULT_FIT_END,
    model_bundle: tuple | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date frames for the live VaR monitor dashboard (pure pandas).

    Returns (wide_df, long_df):
      wide: Date, Return, VaR_<model>..., Breach_<model>... (ints) — chart feed.
      long: Date, Model, Return, VaR, Breach — 1 row per model per day, feeds
            the ticking scorecard aggregation and the breach blotter.

    `model_bundle` is an already-loaded (model, tokenizer, meta) so dashboards
    don't re-read the checkpoint on every rebuild; `artifacts_dir` is the
    load-it-yourself alternative. Both None -> classical models only.
    """
    per_ticker = log_returns_by_ticker(prices_long)
    if ticker not in per_ticker:
        raise ValueError(f"No return history for {ticker!r}.")
    dates, rets = per_ticker[ticker]
    cutoff = np.datetime64(pd.Timestamp(fit_end))
    test_start = int(np.searchsorted(dates, cutoff, side="right"))
    n_test = len(rets) - test_start
    if test_start < MIN_TEST_OBS or n_test < MIN_TEST_OBS:
        raise ValueError(f"Not enough history around {fit_end} for {ticker!r}.")

    if model_bundle is None and artifacts_dir is not None:
        from marketlab.sample import load_artifacts
        model_bundle = load_artifacts(artifacts_dir)
    if model_bundle is not None:
        model, tokenizer, meta = model_bundle
        gpt = (MarketGPTGenerator(model, tokenizer, meta["tickers"].index(ticker))
               if ticker in meta["tickers"] else None)
    else:
        gpt = None

    gens = build_generators(rets[:test_start], gpt)
    realized = rets[test_start:]
    wide = pd.DataFrame({"Date": pd.DatetimeIndex(dates[test_start:]), "Return": realized})
    long_parts = []
    for name, gen in gens.items():
        var_series = gen.var_series(rets, test_start, alpha)
        b = breaches(realized, var_series).astype(int)
        wide[f"VaR_{name}"] = var_series
        wide[f"Breach_{name}"] = b
        long_parts.append(pd.DataFrame({
            "Date": wide["Date"], "Model": name, "Return": realized,
            "VaR": var_series, "Breach": b,
        }))
    long_df = (
        pd.concat(long_parts, ignore_index=True)
        .sort_values(["Date", "Model"], kind="stable")
        .reset_index(drop=True)
    )
    return wide, long_df


def make_figure(prices_long: pd.DataFrame, artifacts_dir: str | None, ticker: str,
                out_path: str, fit_end: str = DEFAULT_FIT_END,
                alpha: float = 0.05) -> None:
    """Held-out realized returns vs each model's rolling 95% VaR, breaches marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_ticker = log_returns_by_ticker(prices_long)
    dates, rets = per_ticker[ticker]
    cutoff = np.datetime64(pd.Timestamp(fit_end))
    test_start = int(np.searchsorted(dates, cutoff, side="right"))
    realized = rets[test_start:]
    test_dates = pd.DatetimeIndex(dates[test_start:])

    gpt = (MarketGPTGenerator.from_artifacts(artifacts_dir, ticker)
           if artifacts_dir is not None else None)
    gens = build_generators(rets[:test_start], gpt)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.plot(test_dates, realized, color="0.65", lw=0.7, label="realized return")
    styles = {
        "iid_gaussian": dict(color="#6b7280", ls="--", lw=1.0),
        "block_bootstrap": dict(color="#0e7c86", ls="--", lw=1.0),
        "garch11": dict(color="#1d4ed8", ls="-", lw=1.2),
        "garch11_t": dict(color="#15803d", ls="-", lw=1.2),
        "marketgpt": dict(color="#c2410c", ls="-", lw=1.2),
    }
    markers = {"garch11": ("x", "#1d4ed8"), "marketgpt": ("o", "#c2410c")}
    for name, gen in gens.items():
        vs = gen.var_series(rets, test_start, alpha)
        ax.plot(test_dates, vs, label=f"{name} VaR{int((1 - alpha) * 100)}",
                **styles[name])
        if name in markers:  # mark breaches for the conditional models only
            m, color = markers[name]
            hit = breaches(realized, vs)
            ax.scatter(test_dates[hit], realized[hit], marker=m, s=28,
                       color=color, zorder=5, label=f"{name} breaches")
    ax.set_title(f"{ticker} held-out period: 1-day VaR{int((1 - alpha) * 100)} "
                 "forecasts vs realized returns")
    ax.set_ylabel("daily log-return")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--cache", default="data/universe_cache.parquet")
    ap.add_argument("--fit-end", default=DEFAULT_FIT_END,
                    help="classical models fit through here; all models test after it")
    ap.add_argument("--ticker", default="AAPL", help="detail figure ticker")
    ap.add_argument("--figure", default="artifacts/var_backtest.png")
    args = ap.parse_args(argv)

    prices = load_universe(args.cache)
    table = leaderboard(prices, args.artifacts, args.fit_end)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 140):
        print(table.to_string(index=False))
    print("\nkupiec_p: breach RATE consistent with alpha?  "
          "christoffersen_p: breaches independent (not clustered)?\n"
          "Low p = reject the model. traffic_light: Basel zones for 99% VaR "
          "(breaches per 250 days: green<5, yellow 5-9, red>=10).")

    make_figure(prices, args.artifacts, args.ticker, args.figure, args.fit_end)
    print(f"figure written to {args.figure}")


if __name__ == "__main__":
    main()
