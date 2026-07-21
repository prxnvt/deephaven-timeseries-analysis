"""Discretize daily log-returns into tokens (and back) for the generative model.

The makemore analogy: names -> characters, markets -> return buckets. Bins are
quantiles of the TRAINING returns only (no leakage), so each token is roughly
equally likely a priori; decoding maps a token to the mean training return in
its bin. Consequence worth being honest about: the marginal return distribution
(fat tails included) is baked in by construction — the model earns its keep on
TEMPORAL structure (vol clustering), which is what evaluation must probe.
"""

from __future__ import annotations

import json

import numpy as np


class ReturnTokenizer:
    """Quantile-bin tokenizer over daily log-returns."""

    def __init__(self, n_bins: int = 64):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None       # (n_bins - 1,) interior edges
        self.bin_values: np.ndarray | None = None  # (n_bins,) decode value per bin
        self.bin_probs: np.ndarray | None = None   # (n_bins,) train marginal, for seeding

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    def fit(self, log_returns: np.ndarray) -> "ReturnTokenizer":
        r = np.asarray(log_returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if len(r) < self.n_bins:
            raise ValueError(f"Need at least {self.n_bins} returns to fit, got {len(r)}.")
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges = np.quantile(r, qs)

        tokens = self._encode_fitted(r)
        counts = np.bincount(tokens, minlength=self.n_bins).astype(np.float64)
        self.bin_probs = counts / counts.sum()

        # Decode value = mean training return per bin; empty bins (possible with
        # tied edges) fall back to interpolating between neighboring edges.
        sums = np.bincount(tokens, weights=r, minlength=self.n_bins)
        with np.errstate(invalid="ignore"):
            means = sums / counts
        if np.isnan(means).any():
            lo = np.concatenate([[r.min()], self.edges])
            hi = np.concatenate([self.edges, [r.max()]])
            midpoints = (lo + hi) / 2.0
            means = np.where(np.isnan(means), midpoints, means)
        self.bin_values = means
        return self

    def _require_fitted(self):
        if self.edges is None:
            raise RuntimeError("Tokenizer is not fitted; call fit() or from_json().")

    def _encode_fitted(self, r: np.ndarray) -> np.ndarray:
        # searchsorted against interior edges: values beyond either extreme
        # clamp into the first/last bin naturally.
        return np.searchsorted(self.edges, r, side="left").astype(np.int64)

    def encode(self, log_returns: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._encode_fitted(np.asarray(log_returns, dtype=np.float64))

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.bin_values[np.asarray(tokens, dtype=np.int64)]

    def to_json(self) -> str:
        self._require_fitted()
        return json.dumps({
            "n_bins": self.n_bins,
            "edges": self.edges.tolist(),
            "bin_values": self.bin_values.tolist(),
            "bin_probs": self.bin_probs.tolist(),
        })

    @classmethod
    def from_json(cls, payload: str) -> "ReturnTokenizer":
        obj = json.loads(payload)
        tok = cls(n_bins=int(obj["n_bins"]))
        tok.edges = np.asarray(obj["edges"], dtype=np.float64)
        tok.bin_values = np.asarray(obj["bin_values"], dtype=np.float64)
        tok.bin_probs = np.asarray(obj["bin_probs"], dtype=np.float64)
        return tok
