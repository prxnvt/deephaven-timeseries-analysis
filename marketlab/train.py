"""Train MarketGPT on the ticker universe (host-side; the container never trains).

Usage:
    .venv/bin/python -m marketlab.train --cache data/universe_cache.parquet --out artifacts

Temporal split (no leakage): the tokenizer's bins and all training windows come
from returns dated <= --train-end; validation (used for early-stopping
selection) is the window (--train-end, --val-end]; anything after --val-end is
NEVER seen — not by gradients, not by checkpoint selection — so downstream
evaluations (the VaR bake-off) run on genuinely untouched data. The val loss is
reported next to the *marginal baseline* — the cross-entropy of val tokens
under the training marginal distribution. Beating it is the evidence that the
model learned temporal structure, not just the (baked-in) return distribution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from marketlab.data import load_universe
from marketlab.model import MarketGPT, ModelConfig
from marketlab.tokenizer import ReturnTokenizer


@dataclass
class WindowData:
    """Tokenized sliding windows plus everything needed to interpret them."""
    tokenizer: ReturnTokenizer
    tickers: list[str]                 # index in this list == ticker_id
    x_train: torch.Tensor              # (N, T) int64
    y_train: torch.Tensor              # (N, T) int64, x shifted by one
    tid_train: torch.Tensor            # (N,) int64
    x_val: torch.Tensor
    y_val: torch.Tensor
    tid_val: torch.Tensor
    baseline_val_ce: float             # CE of val tokens under the train marginal


def log_returns_by_ticker(prices_long: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per ticker: (dates, log-returns), each return dated by its later bar."""
    out = {}
    for ticker, group in prices_long.groupby("Ticker"):
        group = group.sort_values("Date")
        close = group["Close"].to_numpy(dtype=np.float64)
        if len(close) < 2:
            continue
        rets = np.diff(np.log(close))
        out[str(ticker)] = (group["Date"].to_numpy()[1:], rets)
    return out


def _windows_from_tokens(tokens: np.ndarray, block_size: int,
                         stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) windows of length block_size at the given stride; empty if short."""
    n = len(tokens) - block_size
    if n <= 0:
        empty = np.empty((0, block_size), dtype=np.int64)
        return empty, empty.copy()
    idx = np.arange(0, n, stride)[:, None] + np.arange(block_size)[None, :]
    return tokens[idx], tokens[idx + 1]


def build_windows(
    prices_long: pd.DataFrame,
    n_bins: int = 64,
    block_size: int = 128,
    train_end: str = "2021-12-31",
    val_end: str | None = None,
    stride: int = 1,
) -> WindowData:
    """Three-way temporal split: train <= train_end < val <= val_end < (unseen).

    Anything after `val_end` is excluded from BOTH gradient updates and
    early-stopping selection — it stays genuinely unseen for downstream
    evaluation (e.g. the VaR bake-off). `val_end=None` keeps the legacy
    two-way behavior (everything after train_end is validation).
    """
    per_ticker = log_returns_by_ticker(prices_long)
    tickers = sorted(per_ticker)
    if not tickers:
        raise ValueError("No tickers with enough history to compute returns.")
    cutoff = np.datetime64(pd.Timestamp(train_end))
    val_cutoff = np.datetime64(pd.Timestamp(val_end)) if val_end else None
    if val_cutoff is not None and val_cutoff <= cutoff:
        raise ValueError("val_end must be after train_end.")

    train_returns = np.concatenate(
        [rets[dates <= cutoff] for dates, rets in per_ticker.values()]
    )
    tokenizer = ReturnTokenizer(n_bins).fit(train_returns)

    parts: dict[str, list] = {k: [] for k in ("xt", "yt", "tt", "xv", "yv", "tv")}
    val_tokens_all = []
    for tid, ticker in enumerate(tickers):
        dates, rets = per_ticker[ticker]
        for is_train in (True, False):
            if is_train:
                mask = dates <= cutoff
            else:
                mask = dates > cutoff
                if val_cutoff is not None:
                    mask &= dates <= val_cutoff
            tokens = tokenizer.encode(rets[mask])
            x, y = _windows_from_tokens(tokens, block_size, stride)
            key = "t" if is_train else "v"
            parts[f"x{key}"].append(x)
            parts[f"y{key}"].append(y)
            parts[f"t{key}"].append(np.full(len(x), tid, dtype=np.int64))
            if not is_train:
                val_tokens_all.append(tokens)

    def _cat(name: str) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(parts[name]))

    val_tokens = np.concatenate(val_tokens_all) if val_tokens_all else np.empty(0, np.int64)
    if len(val_tokens):
        p = np.clip(tokenizer.bin_probs, 1e-12, None)
        baseline = float(-np.mean(np.log(p[val_tokens])))
    else:
        baseline = float("nan")

    return WindowData(
        tokenizer=tokenizer,
        tickers=tickers,
        x_train=_cat("xt"), y_train=_cat("yt"), tid_train=_cat("tt"),
        x_val=_cat("xv"), y_val=_cat("yv"), tid_val=_cat("tv"),
        baseline_val_ce=baseline,
    )


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate_ce(model: MarketGPT, x: torch.Tensor, y: torch.Tensor, tid: torch.Tensor,
                device: str, batch_size: int = 512, max_batches: int = 20) -> float:
    """Mean next-token cross-entropy over (a capped sample of) the given windows."""
    if len(x) == 0:
        return float("nan")
    model.eval()
    losses = []
    for start in range(0, min(len(x), batch_size * max_batches), batch_size):
        xb = x[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        tb = tid[start:start + batch_size].to(device)
        logits = model(xb, tb)
        losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def train(
    data: WindowData,
    cfg: ModelConfig,
    steps: int = 3000,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    device: str = "auto",
    seed: int = 1337,
    log_every: int = 200,
    eval_every: int = 200,
) -> tuple[MarketGPT, dict]:
    """Train with periodic validation; the returned model carries the weights of
    its best-val-CE step (early-stopping restore), not the last step — daily
    returns are mostly noise, so unchecked training memorizes the train years."""
    device = pick_device(device)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)  # CPU generator: reproducible batches

    model = MarketGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    first_loss = last_loss = float("nan")
    best_val, best_step = float("inf"), 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    have_val = len(data.x_val) > 0

    model.train()
    for step in range(1, steps + 1):
        ix = torch.randint(len(data.x_train), (batch_size,), generator=gen)
        xb = data.x_train[ix].to(device)
        yb = data.y_train[ix].to(device)
        tb = data.tid_train[ix].to(device)
        logits = model(xb, tb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_loss = loss.item()
        if step == 1:
            first_loss = last_loss

        if have_val and eval_every and (step % eval_every == 0 or step == steps):
            val_ce = evaluate_ce(model, data.x_val, data.y_val, data.tid_val, device)
            if val_ce < best_val:
                best_val, best_step = val_ce, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log_every:
                print(f"step {step:>5}/{steps}  train_loss {last_loss:.4f}  "
                      f"val_ce {val_ce:.4f}  (best {best_val:.4f} @ {best_step})", flush=True)
        elif log_every and (step % log_every == 0 or step == 1):
            print(f"step {step:>5}/{steps}  train_loss {last_loss:.4f}", flush=True)

    if have_val:
        model.load_state_dict(best_state)  # restore the best-validation weights
        model.to(device)
        val_ce = best_val
    else:
        val_ce = float("nan")

    metrics = {
        "first_train_loss": first_loss,
        "final_train_loss": last_loss,
        "val_ce": val_ce,
        "best_step": best_step,
        "baseline_val_ce": data.baseline_val_ce,
        "n_params": model.n_params(),
        "steps": steps,
        "device": device,
    }
    return model, metrics


def save_artifacts(out_dir: str, model: MarketGPT, data: WindowData,
                   train_end: str, metrics: dict, val_end: str | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "tokenizer.json").write_text(data.tokenizer.to_json())
    (out / "config.json").write_text(json.dumps({
        "model": model.cfg.to_dict(),
        "tickers": data.tickers,
        "train_end": train_end,
        "val_end": val_end,
        "metrics": metrics,
    }, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/universe_cache.parquet")
    ap.add_argument("--out", default="artifacts")
    # Defaults come from a small val-CE sweep (2026-07): daily returns are mostly
    # noise, so the winning config is TINY and heavily regularized — bigger /
    # less-regularized variants memorize 2014-2021 and lose to the marginal
    # baseline out-of-sample.
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--n-bins", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--train-end", default="2021-12-31")
    ap.add_argument("--val-end", default="2022-12-31",
                    help="early-stop selection window ends here; later data stays unseen")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    prices = load_universe(args.cache)
    data = build_windows(prices, args.n_bins, args.block_size, args.train_end, args.val_end)
    cfg = ModelConfig(
        vocab_size=args.n_bins, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, n_tickers=len(data.tickers),
        dropout=args.dropout,
    )
    print(f"{len(data.tickers)} tickers | train windows {len(data.x_train):,} | "
          f"val windows {len(data.x_val):,} | baseline val CE {data.baseline_val_ce:.4f}")

    model, metrics = train(
        data, cfg, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=args.device, seed=args.seed,
    )
    print(f"val CE {metrics['val_ce']:.4f} vs marginal baseline {metrics['baseline_val_ce']:.4f} "
          f"({metrics['n_params']:,} params)")
    save_artifacts(args.out, model, data, args.train_end, metrics, val_end=args.val_end)
    print(f"artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
