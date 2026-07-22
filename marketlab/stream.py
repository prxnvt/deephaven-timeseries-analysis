"""Streaming (per-tick) VaR forecasters + the ForecastEngine.

The live monitor's model-in-the-loop core, kept pure and deephaven-free: the
dashboard's table listener is a thin pipe into `ForecastEngine.on_bar`, so all
of the actual stream-processing logic is unit-testable on the host.

Correctness contract (pinned by tests/test_stream.py): feeding returns
one-at-a-time through a Streaming* model reproduces the corresponding batch
`var_series` from marketlab.baselines exactly — the incremental math IS the
backtested math, just folded.

Breach semantics match a real desk: a breach becomes knowable when the NEW bar
arrives and is compared against the forecast made after the PREVIOUS bar, so
`on_bar` checks breaches first, then updates the models and stores the new
forecasts (with per-model inference latency).
"""

from __future__ import annotations

from collections import deque
from time import perf_counter

import numpy as np

from marketlab.baselines import BlockBootstrap, Garch11, IIDGaussian


class StreamingIID:
    """Constant forecaster wrapping a fitted IIDGaussian."""

    def __init__(self, batch: IIDGaussian):
        self._batch = batch

    def warmup(self, returns: np.ndarray) -> None:
        pass

    def update(self, r: float) -> None:
        pass

    def var(self, alpha: float) -> float:
        return self._batch.var(None, alpha)


class StreamingBootstrap:
    """Constant forecaster wrapping a fitted BlockBootstrap."""

    def __init__(self, batch: BlockBootstrap):
        self._batch = batch

    def warmup(self, returns: np.ndarray) -> None:
        pass

    def update(self, r: float) -> None:
        pass

    def var(self, alpha: float) -> float:
        return self._batch.var(None, alpha)


class StreamingGarch11:
    """GARCH(1,1) as an O(1) stream fold over a fitted Garch11's parameters.

    State is sigma^2 for the NEXT bar; update(r) mirrors the exact expression
    in Garch11._filter_sigma2 so stream/batch parity holds to float equality.
    """

    def __init__(self, batch: Garch11):
        self.mu = batch.mu
        self.omega = batch.omega
        self.alpha = batch.alpha
        self.beta = batch.beta
        self._sigma2_next = self.omega / max(1e-12, 1.0 - self.alpha - self.beta)

    def warmup(self, returns: np.ndarray) -> None:
        for r in np.asarray(returns, dtype=np.float64):
            self.update(float(r))

    def update(self, r: float) -> None:
        e2 = (r - self.mu) ** 2
        self._sigma2_next = self.omega + self.alpha * e2 + self.beta * self._sigma2_next

    def var(self, alpha: float) -> float:
        from scipy import stats
        return self.mu + float(np.sqrt(self._sigma2_next)) * float(stats.norm.ppf(alpha))


class StreamingMarketGPT:
    """MarketGPT as a per-tick forecaster: a rolling token context and one
    (1, <=block_size) forward pass per bar, caching the next-day bucket
    cumulative distribution so var() is a lookup."""

    def __init__(self, model, tokenizer, ticker_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.ticker_id = int(ticker_id)
        self._ctx: deque[int] = deque(maxlen=model.cfg.block_size)
        self._cum = None  # cached cumulative next-day bucket probabilities

    def warmup(self, returns: np.ndarray) -> None:
        for tok in self.tokenizer.encode(np.asarray(returns, dtype=np.float64)):
            self._ctx.append(int(tok))
        self._refresh()

    def update(self, r: float) -> None:
        self._ctx.append(int(self.tokenizer.encode(np.array([r]))[0]))
        self._refresh()

    def _refresh(self) -> None:
        if not self._ctx:
            return
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            idx = torch.tensor([list(self._ctx)], dtype=torch.int64)
            tid = torch.tensor([self.ticker_id], dtype=torch.int64)
            logits = self.model(idx, tid)[:, -1, :]
            probs = F.softmax(logits, dim=-1).numpy()[0]
        self._cum = np.cumsum(probs)

    def var(self, alpha: float) -> float:
        if self._cum is None:
            raise RuntimeError("StreamingMarketGPT.var called before any context.")
        idx = int(np.argmax(self._cum >= alpha))  # bins ascend in return value
        return float(self.tokenizer.bin_values[idx])


class ForecastEngine:
    """Stateful per-bar processor: breach checks against the prior forecasts,
    then model updates, new forecasts, and per-model latency."""

    def __init__(self, models: dict, alphas: tuple[float, ...] = (0.05, 0.01)):
        self.models = models
        self.alphas = tuple(alphas)
        self._prev: dict[str, dict[float, float]] = {}

    @staticmethod
    def _level(alpha: float) -> int:
        return int(round((1.0 - alpha) * 100))

    def warmup(self, returns: np.ndarray) -> None:
        """Prime every model on history, then store the initial forecasts so
        the very first replayed bar can already be breach-checked."""
        for model in self.models.values():
            model.warmup(returns)
        self._prev = {
            name: {a: m.var(a) for a in self.alphas} for name, m in self.models.items()
        }

    def on_bar(self, date, ret: float) -> tuple[list[dict], list[dict]]:
        breaches = []
        for name, prev in self._prev.items():
            for alpha, var_q in prev.items():
                if ret < var_q:
                    breaches.append({
                        "Date": date, "Model": name, "Return": ret,
                        "VaR": var_q, "Level": self._level(alpha),
                    })

        forecasts = []
        for name, model in self.models.items():
            t0 = perf_counter()
            model.update(ret)
            new_vars = {a: model.var(a) for a in self.alphas}
            latency_ms = (perf_counter() - t0) * 1e3
            row = {"Date": date, "Model": name, "LatencyMs": latency_ms}
            for alpha, value in new_vars.items():
                row[f"VaR{self._level(alpha)}"] = value
            forecasts.append(row)
            self._prev[name] = new_vars
        return forecasts, breaches
