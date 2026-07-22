"""Joint multi-ticker generation: a trading day as a 20-token sentence.

Single-name MarketGPT samples each ticker independently — which misses the
thing that actually kills portfolios: correlation, and the way it tightens
when volatility spikes. The joint model flattens each trading day into one
token per ticker (fixed order), concatenates days into one long stream, and
trains the SAME causal-transformer machinery on it (per-position ticker-slot
embeddings via MarketGPT's 2-D ticker_ids path). Within a day, later slots
condition on earlier slots' same-day moves — cross-sectional dependence is
learned autoregressively, the way raster-scan image models learn 2-D
structure.

Evaluation targets the joint facts a risk desk cares about:
  * the pairwise correlation matrix, real vs synthetic;
  * correlation-in-stress: markets correlate MORE on high-vol days — does the
    generator reproduce that?
  * the money question: portfolio VaR from joint scenarios vs the same
    scenarios with cross-correlation destroyed (column-wise shuffling) — how
    badly does an independence assumption understate portfolio tail risk?

CLI:
    .venv/bin/python -m marketlab.joint train
    .venv/bin/python -m marketlab.joint evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from marketlab.data import load_universe
from marketlab.model import MarketGPT, ModelConfig
from marketlab.tokenizer import ReturnTokenizer
from marketlab.train import WindowData, save_artifacts, train

DEFAULT_TRAIN_END = "2021-12-31"
DEFAULT_VAL_END = "2022-12-31"
DEFAULT_BLOCK_DAYS = 12


# --- Panel prep ----------------------------------------------------------------
def align_panel(prices_long: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(dates, returns_matrix DxK, tickers): log-returns on complete days only,
    tickers in fixed sorted order (the slot order everywhere downstream)."""
    closes = prices_long.pivot_table(index="Date", columns="Ticker", values="Close")
    closes = closes.dropna(axis=0)  # keep only days where every name traded
    rets = np.diff(np.log(closes.to_numpy(dtype=np.float64)), axis=0)
    dates = closes.index.to_numpy()[1:]  # each return dated by its later bar
    return dates, rets, [str(t) for t in closes.columns]


def flatten_days(returns_matrix: np.ndarray, tokenizer: ReturnTokenizer) -> np.ndarray:
    """(D, K) returns -> (D*K,) token stream, day-major, slot order preserved."""
    return tokenizer.encode(returns_matrix.reshape(-1))


def _day_aligned_windows(tokens: np.ndarray, n_slots: int,
                         block_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Day-aligned stride-K windows over a flat stream; y is the flat stream
    shifted by ONE TOKEN (the autoregressive factorization is per token, so a
    window's last target is the next day's first slot)."""
    T = block_days * n_slots
    starts = range(0, len(tokens) - T, n_slots)
    xs = np.stack([tokens[s:s + T] for s in starts]) if len(tokens) > T else \
        np.empty((0, T), dtype=np.int64)
    ys = np.stack([tokens[s + 1:s + T + 1] for s in starts]) if len(tokens) > T else \
        np.empty((0, T), dtype=np.int64)
    return xs, ys


def build_joint_windows(
    prices_long: pd.DataFrame,
    tokenizer: ReturnTokenizer,
    block_days: int = DEFAULT_BLOCK_DAYS,
    train_end: str = DEFAULT_TRAIN_END,
    val_end: str | None = DEFAULT_VAL_END,
) -> WindowData:
    """WindowData over the flattened joint stream; tid is (N, T) slot ids.
    The tokenizer is passed in (fit on pooled single-name TRAIN returns —
    the committed artifacts/tokenizer.json is exactly that)."""
    dates, rets, tickers = align_panel(prices_long)
    n_slots = len(tickers)
    cutoff = np.datetime64(pd.Timestamp(train_end))
    val_cutoff = np.datetime64(pd.Timestamp(val_end)) if val_end else None

    train_mask = dates <= cutoff
    val_mask = ~train_mask if val_cutoff is None else (
        (dates > cutoff) & (dates <= val_cutoff)
    )

    slot_cycle = np.tile(np.arange(n_slots, dtype=np.int64), block_days)
    parts = {}
    val_tokens = np.empty(0, dtype=np.int64)
    for key, mask in (("t", train_mask), ("v", val_mask)):
        tokens = flatten_days(rets[mask], tokenizer)
        x, y = _day_aligned_windows(tokens, n_slots, block_days)
        parts[f"x{key}"], parts[f"y{key}"] = x, y
        parts[f"t{key}"] = np.broadcast_to(slot_cycle, x.shape).copy()
        if key == "v":
            val_tokens = tokens

    if len(val_tokens):
        p = np.clip(tokenizer.bin_probs, 1e-12, None)
        baseline = float(-np.mean(np.log(p[val_tokens])))
    else:
        baseline = float("nan")

    return WindowData(
        tokenizer=tokenizer,
        tickers=tickers,
        x_train=torch.from_numpy(parts["xt"]), y_train=torch.from_numpy(parts["yt"]),
        tid_train=torch.from_numpy(parts["tt"]),
        x_val=torch.from_numpy(parts["xv"]), y_val=torch.from_numpy(parts["yv"]),
        tid_val=torch.from_numpy(parts["tv"]),
        baseline_val_ce=baseline,
    )


# --- Sampling --------------------------------------------------------------------
def load_joint_artifacts(artifacts_dir: str, device: str = "cpu"):
    art = Path(artifacts_dir)
    meta = json.loads((art / "config.json").read_text())
    cfg = ModelConfig.from_dict(meta["model"])
    model = MarketGPT(cfg)
    model.load_state_dict(torch.load(art / "model.pt", map_location=device))
    model.to(device).eval()
    tokenizer = ReturnTokenizer.from_json((art / "tokenizer.json").read_text())
    return model, tokenizer, meta


@torch.no_grad()
def sample_joint(
    model: MarketGPT,
    tokenizer: ReturnTokenizer,
    n_slots: int,
    n_paths: int,
    horizon_days: int,
    prefix_matrix: np.ndarray | None = None,
    temperature: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Sample (n_paths, horizon_days, n_slots) joint daily log-returns.

    Context cropping stays DAY-ALIGNED (window start at slot 0) so position
    and slot patterns match what training saw."""
    model.eval()
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    block = model.cfg.block_size

    if prefix_matrix is not None and len(prefix_matrix):
        prefix = torch.from_numpy(flatten_days(np.asarray(prefix_matrix), tokenizer))
        ctx = prefix[None, :].repeat(n_paths, 1)
    else:
        probs = torch.from_numpy(tokenizer.bin_probs)
        ctx = torch.multinomial(probs.expand(n_paths, -1), 1, generator=gen)

    n_prefix = ctx.size(1)
    for _ in range(horizon_days * n_slots):
        start = max(0, ctx.size(1) - block)
        start += (-start) % n_slots  # snap the window start to a day boundary
        window = ctx[:, start:]
        slots = (torch.arange(window.size(1)) % n_slots).expand(window.size(0), -1)
        logits = model(window, slots)[:, -1, :] / temperature
        nxt = torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1, generator=gen)
        ctx = torch.cat([ctx, nxt], dim=1)

    flat = ctx[:, n_prefix:].numpy()
    rets = tokenizer.decode(flat)
    return rets.reshape(n_paths, horizon_days, n_slots)


# --- Joint stylized facts ----------------------------------------------------------
def mean_pairwise_corr(returns_matrix: np.ndarray) -> float:
    """Mean off-diagonal pairwise correlation of a (days, K) panel."""
    corr = np.corrcoef(returns_matrix.T)
    K = corr.shape[0]
    return float((corr.sum() - K) / (K * (K - 1)))


def stress_split_corr(returns_matrix: np.ndarray, q: float = 0.2) -> tuple[float, float]:
    """(calm_corr, stress_corr): mean pairwise correlation on the bottom-q and
    top-q days ranked by cross-sectional mean |return| (a day-vol proxy)."""
    day_vol = np.mean(np.abs(returns_matrix), axis=1)
    lo, hi = np.quantile(day_vol, [q, 1.0 - q])
    calm = returns_matrix[day_vol <= lo]
    stress = returns_matrix[day_vol >= hi]
    return mean_pairwise_corr(calm), mean_pairwise_corr(stress)


def portfolio_var_joint_vs_independent(
    paths: np.ndarray, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Equal-weight portfolio VaR from joint scenarios vs the SAME scenarios
    with cross-correlation destroyed (each ticker's column shuffled across
    days within each path — marginals and per-name dynamics kept)."""
    rng = np.random.default_rng(seed)
    port_joint = paths.mean(axis=2).reshape(-1)  # (paths*days,)
    shuffled = paths.copy()
    n_paths, n_days, n_slots = shuffled.shape
    for p in range(n_paths):
        for k in range(n_slots):
            rng.shuffle(shuffled[p, :, k])
    port_indep = shuffled.mean(axis=2).reshape(-1)
    return float(np.quantile(port_joint, alpha)), float(np.quantile(port_indep, alpha))


def make_joint_figure(real_panel: np.ndarray, synth_panel: np.ndarray,
                      tickers: list[str], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, panel, label in ((axes[0], real_panel, "real (train years)"),
                             (axes[1], synth_panel, "synthetic")):
        im = ax.imshow(np.corrcoef(panel.T), vmin=-0.1, vmax=1.0, cmap="magma")
        ax.set_title(f"Pairwise correlation — {label}")
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=90, fontsize=6)
        ax.set_yticklabels(tickers, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[2]
    real_cs = stress_split_corr(real_panel)
    synth_cs = stress_split_corr(synth_panel)
    x = np.arange(2)
    ax.bar(x - 0.18, real_cs, width=0.36, label="real", color="#0e7c86")
    ax.bar(x + 0.18, synth_cs, width=0.36, label="synthetic", color="#c2410c")
    ax.set_xticks(x)
    ax.set_xticklabels(["calm days\n(bottom vol quintile)", "stressed days\n(top vol quintile)"])
    ax.set_ylabel("mean pairwise correlation")
    ax.set_title("Correlation tightens under stress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--cache", default="data/universe_cache.parquet")
    tr.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    tr.add_argument("--out", default="artifacts/joint")
    tr.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    tr.add_argument("--n-layer", type=int, default=2)
    tr.add_argument("--n-head", type=int, default=4)
    tr.add_argument("--n-embd", type=int, default=48)
    tr.add_argument("--dropout", type=float, default=0.3)
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    tr.add_argument("--val-end", default=DEFAULT_VAL_END)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--seed", type=int, default=1337)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--cache", default="data/universe_cache.parquet")
    ev.add_argument("--artifacts", default="artifacts/joint")
    ev.add_argument("--n", type=int, default=40)
    ev.add_argument("--horizon-days", type=int, default=250)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--figure", default="artifacts/joint/corr_facts.png")
    args = ap.parse_args(argv)

    prices = load_universe(args.cache)

    if args.cmd == "train":
        tokenizer = ReturnTokenizer.from_json(Path(args.tokenizer).read_text())
        data = build_joint_windows(prices, tokenizer, args.block_days,
                                   args.train_end, args.val_end)
        n_slots = len(data.tickers)
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size, block_size=args.block_days * n_slots,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            n_tickers=n_slots, dropout=args.dropout,
        )
        print(f"{n_slots} slots | block {cfg.block_size} tokens ({args.block_days} days) | "
              f"train windows {len(data.x_train):,} | val windows {len(data.x_val):,} | "
              f"baseline val CE {data.baseline_val_ce:.4f}")
        model, metrics = train(data, cfg, steps=args.steps, batch_size=args.batch_size,
                               weight_decay=args.weight_decay, device=args.device,
                               seed=args.seed)
        print(f"val CE {metrics['val_ce']:.4f} vs marginal baseline "
              f"{metrics['baseline_val_ce']:.4f} ({metrics['n_params']:,} params)")
        metrics["block_days"] = args.block_days
        save_artifacts(args.out, model, data, args.train_end, metrics,
                       val_end=args.val_end)
        print(f"artifacts written to {args.out}/")
        return

    # evaluate
    model, tokenizer, meta = load_joint_artifacts(args.artifacts)
    dates, rets, tickers = align_panel(prices)
    assert tickers == meta["tickers"], "ticker slot order drifted from training"
    cutoff = np.datetime64(pd.Timestamp(meta["train_end"]))
    real_train_panel = rets[dates <= cutoff]

    print(f"sampling {args.n} joint paths x {args.horizon_days} days ...", flush=True)
    paths = sample_joint(model, tokenizer, len(tickers), args.n, args.horizon_days,
                         seed=args.seed)
    synth_panel = paths.reshape(-1, len(tickers))

    real_c = mean_pairwise_corr(real_train_panel)
    synth_c = mean_pairwise_corr(synth_panel)
    real_calm, real_stress = stress_split_corr(real_train_panel)
    synth_calm, synth_stress = stress_split_corr(synth_panel)
    var_joint, var_indep = portfolio_var_joint_vs_independent(paths)

    print(f"  mean pairwise corr      real {real_c:+.3f} | synth {synth_c:+.3f}")
    print(f"  calm-day corr           real {real_calm:+.3f} | synth {synth_calm:+.3f}")
    print(f"  stressed-day corr       real {real_stress:+.3f} | synth {synth_stress:+.3f}")
    print(f"  eq-weight portfolio VaR95: joint {var_joint:+.4f} vs "
          f"independence-shuffled {var_indep:+.4f} "
          f"(independence understates by {abs(var_joint / var_indep):.2f}x)")
    make_joint_figure(real_train_panel, synth_panel, tickers, args.figure)
    print(f"figure written to {args.figure}")


if __name__ == "__main__":
    main()
