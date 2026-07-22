"""Adversarial fidelity check: can a trained classifier tell real windows from
MarketGPT windows?

A GAN-flavored evaluation without GAN training pain: sample thousands of
60-day return windows from real history and from the generator, train a small
MLP to separate them, and report held-out accuracy/AUC. 50% accuracy means
indistinguishable; 100% means the generator has an obvious tell. The
discriminator sees RAW returns on purpose — scale differences (e.g. the
generator running hot on vol) are legitimate tells, not artifacts to normalize
away.

CLI:
    .venv/bin/python -m marketlab.discriminator --n-per-class 4000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from marketlab.data import load_universe
from marketlab.sample import generate, load_artifacts

WINDOW = 60


def real_windows(prices_long, tickers: list[str], n_per_ticker: int,
                 window: int = WINDOW, seed: int = 0) -> np.ndarray:
    """Random real return windows, pooled across tickers."""
    rng = np.random.default_rng(seed)
    out = []
    for ticker in tickers:
        close = (
            prices_long[prices_long["Ticker"] == ticker]
            .sort_values("Date")["Close"].to_numpy(dtype=np.float64)
        )
        rets = np.diff(np.log(close))
        starts = rng.integers(0, len(rets) - window, size=n_per_ticker)
        out.extend(rets[s:s + window] for s in starts)
    return np.asarray(out)


def synthetic_windows(model, tokenizer, meta, tickers: list[str],
                      n_per_ticker: int, window: int = WINDOW,
                      seed: int = 0) -> np.ndarray:
    out = []
    for i, ticker in enumerate(tickers):
        paths = generate(model, tokenizer, meta["tickers"].index(ticker),
                         n_paths=n_per_ticker, horizon=window, seed=seed + i)
        out.append(paths)
    return np.concatenate(out, axis=0)


class _Mlp(nn.Module):
    """Input is [returns, |returns|]: zero-mean symmetric classes that differ
    in VARIANCE are linearly inseparable on raw returns alone, so the |r| half
    hands the discriminator explicit volatility features."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc_score(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(score_pos > score_neg) via the rank statistic (ties get half credit)."""
    all_scores = np.concatenate([scores_pos, scores_neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1.0
    r_pos = ranks[: len(scores_pos)].sum()
    n_p, n_n = len(scores_pos), len(scores_neg)
    return float((r_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def train_discriminator(
    real: np.ndarray,
    synth: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict:
    """Train the MLP on scaled windows; report held-out accuracy and AUC."""
    rng = np.random.default_rng(seed)
    raw = np.concatenate([real, synth]).astype(np.float32) * 100.0  # percent scale
    x = np.concatenate([raw, np.abs(raw)], axis=1)  # returns + vol features
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))]).astype(np.float32)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_test = int(len(x) * test_frac)
    # Standardize on the TRAINING rows only (the |r| block has a large positive
    # mean, which otherwise stalls the decision threshold for many epochs).
    mu = x[n_test:].mean(axis=0, keepdims=True)
    sd = x[n_test:].std(axis=0, keepdims=True) + 1e-8
    x = (x - mu) / sd
    x_test, y_test = torch.from_numpy(x[:n_test]), torch.from_numpy(y[:n_test])
    x_tr, y_tr = torch.from_numpy(x[n_test:]), torch.from_numpy(y[n_test:])

    torch.manual_seed(seed)
    model = _Mlp(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for lo in range(0, len(x_tr), 512):
            xb, yb = x_tr[lo:lo + 512], y_tr[lo:lo + 512]
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        scores = model(x_test).numpy()
    preds = scores > 0.0
    accuracy = float((preds == (y_test.numpy() > 0.5)).mean())
    auc = auc_score(scores[y_test.numpy() > 0.5], scores[y_test.numpy() <= 0.5])
    return {"accuracy": accuracy, "auc": auc, "n_test": int(n_test)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/universe_cache.parquet")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n-per-ticker", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    prices = load_universe(args.cache)
    model, tokenizer, meta = load_artifacts(args.artifacts)
    tickers = sorted(set(meta["tickers"]) & set(prices["Ticker"].unique()))

    print(f"building {args.n_per_ticker * len(tickers):,} windows per class ...",
          flush=True)
    real = real_windows(prices, tickers, args.n_per_ticker, seed=args.seed)
    synth = synthetic_windows(model, tokenizer, meta, tickers,
                              args.n_per_ticker, seed=args.seed)
    result = train_discriminator(real, synth, epochs=args.epochs, seed=args.seed)
    print(f"held-out accuracy {result['accuracy']:.1%} | AUC {result['auc']:.3f} "
          f"(n_test={result['n_test']:,}; 50% = indistinguishable)")


if __name__ == "__main__":
    main()
