"""Unit tests for marketlab.baselines (classical generators)."""

import numpy as np
import pytest
from scipy import stats

from marketlab.baselines import (
    BlockBootstrap,
    Garch11,
    IIDGaussian,
    simulate_garch11,
)

RNG = np.random.default_rng(11)
TRAIN = RNG.normal(0.0004, 0.015, 3000)


class TestIIDGaussian:
    def test_var_matches_analytic(self):
        gen = IIDGaussian().fit(TRAIN)
        expected = TRAIN.mean() + TRAIN.std() * stats.norm.ppf(0.05)
        assert gen.var(None, 0.05) == pytest.approx(expected)

    def test_sample_shape_and_moments(self):
        gen = IIDGaussian().fit(TRAIN)
        s = gen.sample(50, 400, seed=1)
        assert s.shape == (50, 400)
        assert s.mean() == pytest.approx(TRAIN.mean(), abs=3 * TRAIN.std() / 140)
        assert s.std() == pytest.approx(TRAIN.std(), rel=0.05)


class TestBlockBootstrap:
    def test_samples_come_from_source(self):
        gen = BlockBootstrap(block_size=10).fit(TRAIN)
        s = gen.sample(20, 250, seed=2)
        assert s.shape == (20, 250)
        assert np.isin(np.round(s.ravel(), 12), np.round(TRAIN, 12)).all()

    def test_preserves_moments(self):
        gen = BlockBootstrap().fit(TRAIN)
        s = gen.sample(200, 500, seed=3)
        assert s.mean() == pytest.approx(TRAIN.mean(), abs=5e-4)
        assert s.std() == pytest.approx(TRAIN.std(), rel=0.05)

    def test_var_is_empirical_quantile(self):
        gen = BlockBootstrap().fit(TRAIN)
        assert gen.var(None, 0.05) == pytest.approx(np.quantile(TRAIN, 0.05))

    def test_seeded_determinism(self):
        gen = BlockBootstrap().fit(TRAIN)
        np.testing.assert_array_equal(gen.sample(5, 50, seed=7), gen.sample(5, 50, seed=7))


TRUE_GARCH = dict(omega=4e-6, alpha=0.08, beta=0.90)


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_garch11(**TRUE_GARCH, n=6000, seed=5)
    return Garch11().fit(sim), sim


class TestGarch11:
    TRUE = TRUE_GARCH

    def test_mle_recovers_parameters(self, fitted):
        gen, _ = fitted
        assert gen.last_fit_converged
        assert gen.alpha == pytest.approx(self.TRUE["alpha"], abs=0.05)
        assert gen.beta == pytest.approx(self.TRUE["beta"], abs=0.07)
        true_uncond = np.sqrt(self.TRUE["omega"] / (1 - self.TRUE["alpha"] - self.TRUE["beta"]))
        fit_uncond = np.sqrt(gen.omega / (1 - gen.alpha - gen.beta))
        assert fit_uncond == pytest.approx(true_uncond, rel=0.20)

    def test_var_reacts_to_volatility(self, fitted):
        gen, sim = fitted
        calm = np.full(300, 1e-4)
        wild = np.concatenate([np.full(280, 1e-4), np.tile([0.05, -0.05], 10)])
        assert gen.var(wild, 0.05) < gen.var(calm, 0.05)  # more vol -> deeper VaR

    def test_var_series_matches_pointwise_var(self, fitted):
        gen, sim = fitted
        series = gen.var_series(sim[:500], test_start=490, alpha=0.05)
        pointwise = [gen.var(sim[:t], 0.05) for t in range(490, 500)]
        np.testing.assert_allclose(series, pointwise, rtol=1e-10)

    def test_sample_shows_vol_clustering(self, fitted):
        from marketlab.evaluate import acf
        gen, _ = fitted
        s = gen.sample(1, 20_000, seed=9)[0]
        assert np.mean(acf(np.abs(s), 20)) > 0.03
        assert abs(np.mean(acf(s, 20))) < 0.02


@pytest.mark.parametrize("gen_factory", [
    lambda: IIDGaussian().fit(TRAIN),
    lambda: BlockBootstrap().fit(TRAIN),
    lambda: Garch11().fit(simulate_garch11(4e-6, 0.08, 0.90, 4000, seed=6)),
])
def test_var_deepens_with_confidence(gen_factory):
    gen = gen_factory()
    history = TRAIN[:500]
    assert gen.var(history, 0.01) < gen.var(history, 0.05) < 0.0
