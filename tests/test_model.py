"""Unit tests for marketlab.model (MarketGPT)."""

import pytest
import torch
import torch.nn.functional as F

from marketlab.model import MarketGPT, ModelConfig

TINY = ModelConfig(
    vocab_size=16, block_size=32, n_layer=1, n_head=2, n_embd=32, n_tickers=3, dropout=0.0
)


def test_forward_shape_and_param_count():
    torch.manual_seed(0)
    model = MarketGPT(TINY)
    idx = torch.randint(0, 16, (4, 20))
    tids = torch.tensor([0, 1, 2, 0])
    logits = model(idx, tids)
    assert logits.shape == (4, 20, 16)
    assert 0 < model.n_params() < 1_000_000


def test_sequence_longer_than_block_raises():
    model = MarketGPT(TINY)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 33, dtype=torch.long), torch.zeros(1, dtype=torch.long))


def test_causal_mask_blocks_future_influence():
    """Perturbing tokens after position t must leave logits at <= t unchanged."""
    torch.manual_seed(0)
    model = MarketGPT(TINY).eval()
    t = 10
    a = torch.randint(0, 16, (1, 24))
    b = a.clone()
    b[0, t + 1:] = (b[0, t + 1:] + 7) % 16
    tid = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        la, lb = model(a, tid), model(b, tid)
    assert torch.allclose(la[:, : t + 1], lb[:, : t + 1], atol=1e-5)
    assert not torch.allclose(la[:, t + 1:], lb[:, t + 1:], atol=1e-5)


def test_ticker_embedding_conditions_output():
    torch.manual_seed(0)
    model = MarketGPT(TINY).eval()
    idx = torch.randint(0, 16, (1, 12))
    with torch.no_grad():
        l0 = model(idx, torch.tensor([0]))
        l1 = model(idx, torch.tensor([1]))
    assert not torch.allclose(l0, l1)


def test_can_overfit_a_repeating_pattern():
    torch.manual_seed(0)
    model = MarketGPT(TINY)
    seq = torch.tensor(([0, 1, 2, 3] * 9)[:33])  # deterministic cycle
    x, y = seq[None, :-1], seq[None, 1:]
    tid = torch.zeros(1, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        logits = model(x, tid)
        loss = F.cross_entropy(logits.view(-1, 16), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5  # memorizing a 4-cycle should be easy
