"""Unit tests for marketlab.var_backtest (coverage statistics + leaderboard)."""

import numpy as np
import pytest

from marketlab.var_backtest import (
    basel_traffic_light,
    breaches,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    leaderboard,
    markov_counts,
)


def test_breach_counting():
    realized = np.array([-0.03, 0.01, -0.01, -0.05])
    var_series = np.full(4, -0.02)
    assert breaches(realized, var_series).tolist() == [True, False, False, True]


class TestKupiec:
    def test_exact_rate_passes(self):
        lr, p = kupiec_pof(n_obs=1000, n_breaches=50, alpha=0.05)
        assert lr == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0)

    def test_way_off_rate_fails(self):
        _, p_high = kupiec_pof(1000, 100, 0.05)  # double the expected breaches
        _, p_zero = kupiec_pof(1000, 0, 0.05)    # suspiciously safe also fails
        assert p_high < 1e-6
        assert p_zero < 1e-6

    def test_known_value(self):
        # Hand-computed: n=250, x=8, alpha=0.01, pi=0.032 ->
        # LR = -2[242 ln(.99) + 8 ln(.01) - 242 ln(.968) - 8 ln(.032)] = 7.734
        lr, p = kupiec_pof(250, 8, 0.01)
        assert lr == pytest.approx(7.734, abs=0.01)
        assert 0.001 < p < 0.01


class TestChristoffersen:
    def test_random_breaches_pass(self):
        # A 5%-level test rejects 5% of iid draws by construction, so a fixed
        # seed is chosen where independence comfortably holds; the median p
        # across seeds is ~0.5, which is the property being pinned.
        ps = []
        for seed in range(5):
            b = np.random.default_rng(seed).random(2000) < 0.05
            ps.append(christoffersen_independence(markov_counts(b))[1])
        assert np.median(ps) > 0.05

    def test_clustered_breaches_fail(self):
        b = np.zeros(2000, dtype=bool)
        b[1000:1100] = True  # one solid run: maximal clustering
        _, p = christoffersen_independence(markov_counts(b))
        assert p < 1e-10

    def test_no_breaches_is_vacuous(self):
        lr, p = christoffersen_independence(markov_counts(np.zeros(500, dtype=bool)))
        assert np.isnan(lr) and np.isnan(p)


def test_conditional_coverage_combines_both():
    b = np.zeros(2000, dtype=bool)
    b[1000:1100] = True  # right-ish rate (5%) but fully clustered
    _, p_pof = kupiec_pof(2000, 100, alpha=0.05)
    _, p_cc = conditional_coverage(2000, markov_counts(b), alpha=0.05)
    assert p_pof == pytest.approx(1.0)  # the rate alone looks perfect...
    assert p_cc < 1e-10                 # ...but clustering sinks the joint test


def test_basel_traffic_light_boundaries():
    assert basel_traffic_light(4, 250) == "green"
    assert basel_traffic_light(5, 250) == "yellow"
    assert basel_traffic_light(9, 250) == "yellow"
    assert basel_traffic_light(10, 250) == "red"
    assert basel_traffic_light(8, 500) == "green"  # 4 per 250 after scaling


class TestMonitorFrames:
    ALPHA = 0.05

    @pytest.fixture
    def frames(self, mini_universe):
        from marketlab.var_backtest import monitor_frames
        return monitor_frames(mini_universe, "AAA", self.ALPHA, fit_end="2020-09-30")

    def test_shapes_and_columns(self, frames):
        wide, long_df = frames
        models = {"iid_gaussian", "block_bootstrap", "garch11", "garch11_t"}
        assert {f"VaR_{m}" for m in models} <= set(wide.columns)
        assert {f"Breach_{m}" for m in models} <= set(wide.columns)
        assert len(long_df) == 4 * len(wide)
        assert set(long_df["Model"]) == models
        assert wide["Date"].is_monotonic_increasing
        assert long_df["Date"].is_monotonic_increasing

    def test_breach_definition_exact(self, frames):
        wide, long_df = frames
        assert (long_df["Breach"] == (long_df["Return"] < long_df["VaR"]).astype(int)).all()

    def test_wide_and_long_agree(self, frames):
        wide, long_df = frames
        for m in ("iid_gaussian", "garch11"):
            sub = long_df[long_df["Model"] == m].reset_index(drop=True)
            np.testing.assert_allclose(sub["VaR"], wide[f"VaR_{m}"])
            assert (sub["Breach"].to_numpy() == wide[f"Breach_{m}"].to_numpy()).all()

    def test_99_var_deeper_than_95(self, mini_universe):
        from marketlab.var_backtest import monitor_frames
        wide95, _ = monitor_frames(mini_universe, "AAA", 0.05, fit_end="2020-09-30")
        wide99, _ = monitor_frames(mini_universe, "AAA", 0.01, fit_end="2020-09-30")
        for m in ("iid_gaussian", "block_bootstrap", "garch11", "garch11_t"):
            assert (wide99[f"VaR_{m}"] < wide95[f"VaR_{m}"]).all()

    def test_unknown_ticker_raises(self, mini_universe):
        from marketlab.var_backtest import monitor_frames
        with pytest.raises(ValueError, match="No return history"):
            monitor_frames(mini_universe, "ZZZ", 0.05, fit_end="2020-09-30")


def test_leaderboard_smoke_on_fixture(mini_universe):
    # No artifacts dir -> classical models only; fixture spans 2020-2021 so use
    # an in-range split with enough observations on both sides.
    table = leaderboard(mini_universe, artifacts_dir=None, fit_end="2020-09-30")
    assert set(table["model"]) == {
        "iid_gaussian", "block_bootstrap", "garch11", "garch11_t",
    }
    assert set(table["alpha"]) == {0.05, 0.01}
    assert (table["n_obs"] > 0).all()
    assert ((table["breach_rate"] >= 0) & (table["breach_rate"] <= 1)).all()
    lights = set(table.loc[table["alpha"] == 0.01, "traffic_light"])
    assert lights <= {"green", "yellow", "red"}
    assert (table.loc[table["alpha"] == 0.05, "traffic_light"] == "").all()
