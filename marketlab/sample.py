"""Generate synthetic return/price paths from a trained MarketGPT checkpoint.

Two modes:
  * unconditional — the first token is drawn from the training marginal, then
    the model continues autoregressively ("a market that never happened");
  * prefix-conditioned — encode a run of REAL recent returns as the prompt and
    sample continuations ("1,000 plausible next quarters from today"), which is
    what the fan-chart dashboard uses.

Temperature scales the pre-softmax logits, exactly as in makemore: <1 damps the
tails ("calm market"), >1 fattens them ("stressed market").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from marketlab.model import MarketGPT, ModelConfig
from marketlab.tokenizer import ReturnTokenizer


def load_artifacts(artifacts_dir: str, device: str = "cpu"):
    """Load (model, tokenizer, meta) written by marketlab.train.save_artifacts."""
    art = Path(artifacts_dir)
    meta = json.loads((art / "config.json").read_text())
    cfg = ModelConfig.from_dict(meta["model"])
    model = MarketGPT(cfg)
    model.load_state_dict(torch.load(art / "model.pt", map_location=device))
    model.to(device).eval()
    tokenizer = ReturnTokenizer.from_json((art / "tokenizer.json").read_text())
    return model, tokenizer, meta


@torch.no_grad()
def generate(
    model: MarketGPT,
    tokenizer: ReturnTokenizer,
    ticker_id: int,
    n_paths: int,
    horizon: int,
    temperature: float = 1.0,
    prefix_returns: np.ndarray | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Sample (n_paths, horizon) daily log-returns for one ticker id."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    model.eval()
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    if prefix_returns is not None and len(prefix_returns):
        prefix = torch.from_numpy(tokenizer.encode(np.asarray(prefix_returns)))
        ctx = prefix[None, :].repeat(n_paths, 1)
    else:
        probs = torch.from_numpy(tokenizer.bin_probs)
        first = torch.multinomial(probs.expand(n_paths, -1), 1, generator=gen)
        ctx = first

    tid = torch.full((n_paths,), int(ticker_id), dtype=torch.int64, device=device)
    n_prefix = ctx.size(1)
    for _ in range(horizon):
        window = ctx[:, -model.cfg.block_size:].to(device)
        logits = model(window, tid)[:, -1, :] / temperature
        # Sample on CPU so a fixed seed is reproducible across devices.
        probs = F.softmax(logits, dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=gen)
        ctx = torch.cat([ctx, nxt], dim=1)

    sampled = ctx[:, n_prefix:].numpy()
    return tokenizer.decode(sampled)


def to_price_paths(log_returns: np.ndarray, start_price: float) -> np.ndarray:
    """Cumulate (n_paths, horizon) log-returns into price paths from start_price."""
    return float(start_price) * np.exp(np.cumsum(log_returns, axis=1))
