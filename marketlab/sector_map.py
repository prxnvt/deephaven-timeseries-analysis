"""S&P 500 scale-up: does sector structure emerge from returns alone?

The single-name model conditions on a learned per-ticker embedding. Scale the
universe to the S&P 500 and those embeddings become a map of the market the
model drew for itself — no sector labels were ever shown to it. Projecting
the ~500 learned vectors to 2-D (plain PCA) and coloring by GICS sector
answers a genuinely interesting question: does "energy", "utilities", "tech"
emerge purely from co-movement statistics?

Pipeline (host-side, cached):
  1. Constituents + GICS sectors scraped from the Wikipedia S&P 500 list.
  2. One batched yfinance download -> data/sp500_cache.parquet (gitignored).
  3. Train the standard tiny MarketGPT config on it (windows strided to keep
     memory sane; ~500 ticker embeddings).
  4. PCA the embedding table, scatter colored by sector ->
     artifacts/sector_map.png.

CLI:
    .venv/bin/python -m marketlab.sector_map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from marketlab.data import HISTORY_END, HISTORY_START, download_universe
from marketlab.model import ModelConfig
from marketlab.train import build_windows, save_artifacts, train

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_constituents() -> pd.DataFrame:
    """(Symbol, Sector) from Wikipedia; symbols normalized to yfinance form.
    Wikipedia 403s urllib's default user agent, so fetch with a real UA."""
    import io
    import urllib.request

    req = urllib.request.Request(
        WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (marketlab research script)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    df = tables[0][["Symbol", "GICS Sector"]].rename(columns={"GICS Sector": "Sector"})
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df.dropna().reset_index(drop=True)


def ensure_cache(cache_path: str, sectors_path: str,
                 min_rows: int = 1500) -> pd.DataFrame:
    """Download the S&P 500 panel once; thereafter serve from parquet."""
    cache, sectors = Path(cache_path), Path(sectors_path)
    if cache.exists() and sectors.exists():
        return pd.read_parquet(cache)
    cons = fetch_constituents()
    sectors.parent.mkdir(parents=True, exist_ok=True)
    cons.to_csv(sectors, index=False)
    print(f"downloading {len(cons)} tickers {HISTORY_START}..{HISTORY_END} "
          "(one batched call, a few minutes) ...", flush=True)
    long_df = download_universe(cons["Symbol"].tolist(), HISTORY_START, HISTORY_END,
                                min_rows=min_rows)
    long_df.to_parquet(cache, index=False)
    print(f"cached {long_df['Ticker'].nunique()} tickers with >= {min_rows} rows",
          flush=True)
    return long_df


def sector_similarity_test(embeddings: np.ndarray, labels: np.ndarray,
                           n_perm: int = 500, seed: int = 0) -> tuple[float, float]:
    """(diff, p): mean cosine similarity of same-sector pairs minus cross-sector
    pairs, with a permutation p-value. The pre-registered claim was that
    sectors emerge from returns alone; this is the test that judges it."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)
    pair_sims = sim[iu]

    def _diff(labs: np.ndarray) -> float:
        same = labs[iu[0]] == labs[iu[1]]
        return float(pair_sims[same].mean() - pair_sims[~same].mean())

    observed = _diff(labels)
    rng = np.random.default_rng(seed)
    null = np.array([_diff(rng.permutation(labels)) for _ in range(n_perm)])
    return observed, float((null >= observed).mean())


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(projected Nx2, explained-variance ratios) via plain SVD."""
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    return centered @ vt[:2].T, explained[:2]


def make_map(embeddings: np.ndarray, tickers: list[str], sectors: pd.DataFrame,
             out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj, explained = pca_2d(embeddings)
    sector_of = dict(zip(sectors["Symbol"], sectors["Sector"]))
    labels = [sector_of.get(t, "Unknown") for t in tickers]
    palette = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, sector in enumerate(sorted(set(labels))):
        mask = np.array([lab == sector for lab in labels])
        ax.scatter(proj[mask, 0], proj[mask, 1], s=18, alpha=0.8,
                   color=palette[i % len(palette)], label=sector)
    ax.set_title("S&P 500 ticker embeddings, learned from returns alone "
                 f"(PCA, {100 * explained.sum():.0f}% var), colored by GICS sector")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, ncol=2, markerscale=1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/sp500_cache.parquet")
    ap.add_argument("--sectors", default="data/sp500_sectors.csv")
    ap.add_argument("--out", default="artifacts/sp500")
    ap.add_argument("--figure", default="artifacts/sector_map.png")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n-bins", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    prices = ensure_cache(args.cache, args.sectors)
    sectors = pd.read_csv(args.sectors)

    data = build_windows(prices, n_bins=args.n_bins, block_size=128,
                         train_end="2021-12-31", val_end="2022-12-31",
                         stride=args.stride)
    cfg = ModelConfig(vocab_size=args.n_bins, block_size=128, n_layer=1, n_head=4,
                      n_embd=32, n_tickers=len(data.tickers), dropout=args.dropout)
    print(f"{len(data.tickers)} tickers | train windows {len(data.x_train):,} "
          f"(stride {args.stride}) | baseline val CE {data.baseline_val_ce:.4f}")
    model, metrics = train(data, cfg, steps=args.steps, batch_size=128,
                           weight_decay=0.1, device=args.device, seed=args.seed)
    print(f"val CE {metrics['val_ce']:.4f} vs baseline {metrics['baseline_val_ce']:.4f}")
    save_artifacts(args.out, model, data, "2021-12-31", metrics, val_end="2022-12-31")

    embeddings = model.ticker_emb.weight.detach().cpu().numpy()
    sector_of = dict(zip(sectors["Symbol"], sectors["Sector"]))
    labels = np.array([sector_of.get(t, "Unknown") for t in data.tickers])
    diff, p = sector_similarity_test(embeddings, labels)
    print(f"same-sector vs cross-sector embedding similarity: diff {diff:+.4f}, "
          f"permutation p={p:.3f} (p >= 0.05 -> no detectable sector structure)")
    make_map(embeddings, data.tickers, sectors, args.figure)
    print(f"sector map written to {args.figure}")


if __name__ == "__main__":
    main()
