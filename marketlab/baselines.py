"""Classical scenario generators + the MarketGPT wrapper, behind one interface.

Every generator supports:
    fit(returns) -> self                      (train-period returns only)
    sample(n_paths, horizon, seed) -> ndarray (n_paths, horizon) of log-returns
    var(history, alpha) -> float              1-day-ahead return quantile q_alpha
    var_series(returns, test_start, alpha)    q_alpha for each t in [test_start, n),
                                              conditioned only on returns[:t]

VaR convention: q_alpha is the alpha-quantile of tomorrow's return (a negative
number for small alpha); a breach is realized_return < q_alpha.

GARCH(1,1) is hand-rolled (normal innovations — a known limitation): the
log-likelihood is ~20 lines and scipy's L-BFGS-B does the rest. Returns are
scaled to percent inside the MLE for optimizer conditioning; stored parameters
are converted back to raw return units.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

_PCT = 100.0  # percent scaling inside the GARCH MLE


class IIDGaussian:
    """The floor: constant-vol normal returns."""

    def fit(self, returns: np.ndarray) -> "IIDGaussian":
        r = np.asarray(returns, dtype=np.float64)
        self.mu = float(r.mean())
        self.sigma = float(r.std())
        return self

    def sample(self, n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mu, self.sigma, (n_paths, horizon))

    def var(self, history: np.ndarray, alpha: float) -> float:
        return self.mu + self.sigma * float(stats.norm.ppf(alpha))

    def var_series(self, returns: np.ndarray, test_start: int, alpha: float) -> np.ndarray:
        return np.full(len(returns) - test_start, self.var(None, alpha))


class BlockBootstrap:
    """Resample blocks of real history — dumb, unconditional, brutally strong."""

    def __init__(self, block_size: int = 20):
        self.block_size = block_size

    def fit(self, returns: np.ndarray) -> "BlockBootstrap":
        self.source = np.asarray(returns, dtype=np.float64)
        if len(self.source) <= self.block_size:
            raise ValueError("Not enough returns to bootstrap from.")
        return self

    def sample(self, n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_blocks = int(np.ceil(horizon / self.block_size))
        starts = rng.integers(0, len(self.source) - self.block_size + 1,
                              size=(n_paths, n_blocks))
        idx = starts[:, :, None] + np.arange(self.block_size)[None, None, :]
        return self.source[idx].reshape(n_paths, -1)[:, :horizon]

    def var(self, history: np.ndarray, alpha: float) -> float:
        # Unconditional: the empirical quantile of the TRAINING returns.
        return float(np.quantile(self.source, alpha))

    def var_series(self, returns: np.ndarray, test_start: int, alpha: float) -> np.ndarray:
        return np.full(len(returns) - test_start, self.var(None, alpha))


def simulate_garch11(omega: float, alpha: float, beta: float, n: int,
                     seed: int = 0, mu: float = 0.0, burn: int = 500,
                     dist: str = "normal", nu: float = 6.0) -> np.ndarray:
    """Simulate a GARCH(1,1) return series (raw units; normal or standardized-t
    innovations — the t draw is rescaled to unit variance)."""
    rng = np.random.default_rng(seed)
    if dist == "t":
        z = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2.0) / nu)
    else:
        z = rng.normal(0.0, 1.0, n + burn)
    sigma2 = omega / (1.0 - alpha - beta)
    out = np.empty(n + burn)
    e_prev = 0.0
    for t in range(n + burn):
        sigma2 = omega + alpha * e_prev**2 + beta * sigma2
        e_prev = np.sqrt(sigma2) * z[t]
        out[t] = mu + e_prev
    return out[burn:]


def _garch_sigma2_path(omega: float, alpha: float, beta: float,
                       e_pct: np.ndarray) -> np.ndarray:
    n = len(e_pct)
    sigma2 = np.empty(n)
    sigma2[0] = e_pct.var()
    e2 = e_pct**2
    for t in range(1, n):
        sigma2[t] = omega + alpha * e2[t - 1] + beta * sigma2[t - 1]
    return sigma2


def garch11_neg_loglik(params: np.ndarray, e_pct: np.ndarray) -> float:
    """Negative log-likelihood of demeaned percent-returns under GARCH(1,1)
    with NORMAL innovations."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10  # penalty keeps L-BFGS-B inside the stationary region
    sigma2 = _garch_sigma2_path(omega, alpha, beta, e_pct)
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + e_pct**2 / sigma2))


def garch11_t_neg_loglik(params: np.ndarray, e_pct: np.ndarray) -> float:
    """Negative log-likelihood under STANDARDIZED-t innovations (unit variance,
    nu > 2), the textbook fat-tail upgrade."""
    from scipy.special import gammaln
    omega, alpha, beta, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or nu <= 2.05:
        return 1e10
    sigma2 = _garch_sigma2_path(omega, alpha, beta, e_pct)
    z2 = e_pct**2 / sigma2
    const = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
             - 0.5 * np.log(np.pi * (nu - 2.0)))
    ll = np.sum(const - 0.5 * np.log(sigma2)
                - ((nu + 1.0) / 2.0) * np.log1p(z2 / (nu - 2.0)))
    return float(-ll)


class Garch11:
    """Hand-rolled GARCH(1,1); dist='normal' (workhorse) or 't' (standardized
    Student-t innovations with fitted nu — the standard fat-tail upgrade)."""

    def __init__(self, dist: str = "normal"):
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.nu: float | None = None

    def _z_alpha(self, alpha: float) -> float:
        """Quantile of the (unit-variance) innovation distribution — shared by
        the batch var paths and StreamingGarch11 so parity holds exactly."""
        if self.dist == "t":
            return float(stats.t.ppf(alpha, self.nu) * np.sqrt((self.nu - 2.0) / self.nu))
        return float(stats.norm.ppf(alpha))

    def fit(self, returns: np.ndarray) -> "Garch11":
        r = np.asarray(returns, dtype=np.float64)
        self.mu = float(r.mean())
        e_pct = (r - self.mu) * _PCT
        var_pct = e_pct.var()
        if self.dist == "t":
            x0 = np.array([0.05 * var_pct, 0.05, 0.90, 8.0])
            res = optimize.minimize(
                garch11_t_neg_loglik, x0, args=(e_pct,), method="L-BFGS-B",
                bounds=[(1e-8, None), (0.0, 0.999), (0.0, 0.999), (2.1, 300.0)],
            )
            omega_pct, self.alpha, self.beta, self.nu = res.x
            self.nu = float(self.nu)
        else:
            x0 = np.array([0.05 * var_pct, 0.05, 0.90])
            res = optimize.minimize(
                garch11_neg_loglik, x0, args=(e_pct,), method="L-BFGS-B",
                bounds=[(1e-8, None), (0.0, 0.999), (0.0, 0.999)],
            )
            omega_pct, self.alpha, self.beta = res.x
        self.omega = float(omega_pct) / _PCT**2   # back to raw return units
        self.last_fit_converged = bool(res.success)
        return self

    def _filter_sigma2(self, returns: np.ndarray) -> np.ndarray:
        """sigma^2_t for each t (raw units), then one step ahead at the end."""
        e2 = (np.asarray(returns, dtype=np.float64) - self.mu) ** 2
        uncond = self.omega / max(1e-12, 1.0 - self.alpha - self.beta)
        sigma2 = np.empty(len(e2) + 1)
        sigma2[0] = uncond
        for t in range(len(e2)):
            sigma2[t + 1] = self.omega + self.alpha * e2[t] + self.beta * sigma2[t]
        return sigma2  # sigma2[t] is the forecast FOR bar t given bars < t

    def sample(self, n_paths: int, horizon: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.dist == "t":
            z = rng.standard_t(self.nu, (n_paths, horizon)) * np.sqrt(
                (self.nu - 2.0) / self.nu
            )
        else:
            z = rng.normal(0.0, 1.0, (n_paths, horizon))
        sigma2 = np.full(n_paths, self.omega / max(1e-12, 1.0 - self.alpha - self.beta))
        e_prev = np.zeros(n_paths)
        out = np.empty((n_paths, horizon))
        for t in range(horizon):
            sigma2 = self.omega + self.alpha * e_prev**2 + self.beta * sigma2
            e_prev = np.sqrt(sigma2) * z[:, t]
            out[:, t] = self.mu + e_prev
        return out

    def var(self, history: np.ndarray, alpha: float) -> float:
        sigma_next = float(np.sqrt(self._filter_sigma2(history)[-1]))
        return self.mu + sigma_next * self._z_alpha(alpha)

    def var_series(self, returns: np.ndarray, test_start: int, alpha: float) -> np.ndarray:
        # One filter pass over the whole series; sigma2[t] uses info through t-1.
        sigma = np.sqrt(self._filter_sigma2(returns)[test_start:len(returns)])
        return self.mu + sigma * self._z_alpha(alpha)


class MarketGPTGenerator:
    """Wraps the trained checkpoint: its per-step softmax over return buckets IS
    a conditional distribution forecast, so VaR is a cumulative-probability read.
    Quantile resolution is limited by the 64-bin vocabulary — a structural
    handicap at the 1% tail that the README pre-registers."""

    def __init__(self, model, tokenizer, ticker_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.ticker_id = int(ticker_id)

    @classmethod
    def from_artifacts(cls, artifacts_dir: str, ticker: str) -> "MarketGPTGenerator":
        from marketlab.sample import load_artifacts
        model, tokenizer, meta = load_artifacts(artifacts_dir)
        return cls(model, tokenizer, meta["tickers"].index(ticker))

    def fit(self, returns: np.ndarray) -> "MarketGPTGenerator":
        return self  # frozen checkpoint; interface parity only

    def sample(self, n_paths: int, horizon: int, seed: int | None = None,
               prefix_returns: np.ndarray | None = None) -> np.ndarray:
        from marketlab.sample import generate
        return generate(self.model, self.tokenizer, self.ticker_id, n_paths, horizon,
                        prefix_returns=prefix_returns, seed=seed)

    def _next_day_probs(self, windows: np.ndarray) -> np.ndarray:
        """(B, T) token windows -> (B, vocab) next-day bucket probabilities."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            tid = torch.full((len(windows),), self.ticker_id, dtype=torch.int64)
            logits = self.model(torch.from_numpy(windows), tid)[:, -1, :]
            return F.softmax(logits, dim=-1).numpy()

    def var(self, history: np.ndarray, alpha: float) -> float:
        return float(self.var_series(np.asarray(history), len(history), alpha,
                                     _include_last=True)[0])

    def var_series(self, returns: np.ndarray, test_start: int, alpha: float,
                   _include_last: bool = False) -> np.ndarray:
        block = self.model.cfg.block_size
        if test_start < block:
            raise ValueError(f"Need at least block_size={block} returns before the "
                             f"test window (got test_start={test_start}).")
        tokens = self.tokenizer.encode(np.asarray(returns, dtype=np.float64))
        ends = range(test_start, len(returns) + (1 if _include_last else 0))
        windows = np.stack([tokens[t - block:t] for t in ends])
        out = np.empty(len(windows))
        for lo in range(0, len(windows), 512):  # chunk to bound memory
            probs = self._next_day_probs(windows[lo:lo + 512])
            cum = np.cumsum(probs, axis=1)  # bins are ascending in return value
            idx = np.argmax(cum >= alpha, axis=1)
            out[lo:lo + 512] = self.tokenizer.bin_values[idx]
        return out
