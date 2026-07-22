"""Unit tests for marketlab.stream — headline: stream/batch parity."""

import numpy as np
import pytest
import torch

from marketlab.baselines import (
    BlockBootstrap,
    Garch11,
    IIDGaussian,
    MarketGPTGenerator,
    simulate_garch11,
)
from marketlab.model import MarketGPT, ModelConfig
from marketlab.stream import (
    ForecastEngine,
    StreamingBootstrap,
    StreamingGarch11,
    StreamingIID,
    StreamingMarketGPT,
)
from marketlab.tokenizer import ReturnTokenizer

RETURNS = simulate_garch11(4e-6, 0.08, 0.90, 1200, seed=21)
TEST_START = 1000


@pytest.fixture(scope="module")
def tiny_gpt():
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=16, block_size=32, n_layer=1, n_head=2,
                      n_embd=32, n_tickers=2, dropout=0.0)
    model = MarketGPT(cfg).eval()
    tokenizer = ReturnTokenizer(n_bins=16).fit(RETURNS[:TEST_START])
    return model, tokenizer


def test_garch_stream_batch_parity():
    batch = Garch11().fit(RETURNS[:TEST_START])
    expected = batch.var_series(RETURNS, TEST_START, 0.05)

    stream = StreamingGarch11(batch)
    stream.warmup(RETURNS[:TEST_START])
    got = []
    for t in range(TEST_START, len(RETURNS)):
        got.append(stream.var(0.05))  # forecast FOR bar t, info through t-1
        stream.update(float(RETURNS[t]))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_marketgpt_stream_batch_parity(tiny_gpt):
    model, tokenizer = tiny_gpt
    batch = MarketGPTGenerator(model, tokenizer, ticker_id=1)
    expected = batch.var_series(RETURNS, TEST_START, 0.05)

    stream = StreamingMarketGPT(model, tokenizer, ticker_id=1)
    stream.warmup(RETURNS[:TEST_START])
    got = []
    for t in range(TEST_START, len(RETURNS)):
        got.append(stream.var(0.05))
        stream.update(float(RETURNS[t]))
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_constant_models_are_constant():
    iid = StreamingIID(IIDGaussian().fit(RETURNS[:TEST_START]))
    boot = StreamingBootstrap(BlockBootstrap().fit(RETURNS[:TEST_START]))
    for stream in (iid, boot):
        stream.warmup(RETURNS[:TEST_START])
        before = stream.var(0.05)
        stream.update(-0.5)  # a crash bar must not move an unconditional model
        assert stream.var(0.05) == before
        assert stream.var(0.01) < stream.var(0.05) < 0.0


class TestForecastEngine:
    @pytest.fixture
    def engine(self):
        iid = IIDGaussian().fit(np.random.default_rng(0).normal(0, 0.01, 500))
        eng = ForecastEngine({"iid_gaussian": StreamingIID(iid)}, alphas=(0.05,))
        eng.warmup(RETURNS[:100])
        return eng, iid

    def test_breach_compares_against_prior_forecast(self, engine):
        eng, iid = engine
        var95 = iid.var(None, 0.05)
        _, breaches = eng.on_bar("d1", var95 - 0.01)  # below the standing forecast
        assert len(breaches) == 1
        assert breaches[0]["VaR"] == pytest.approx(var95)
        assert breaches[0]["Level"] == 95
        _, no_breach = eng.on_bar("d2", 0.0)
        assert no_breach == []

    def test_forecast_rows_shape_and_latency(self, engine):
        eng, _ = engine
        forecasts, _ = eng.on_bar("d1", 0.001)
        assert len(forecasts) == 1
        row = forecasts[0]
        assert set(row) == {"Date", "Model", "LatencyMs", "VaR95"}
        assert row["LatencyMs"] >= 0.0
        assert row["VaR95"] < 0.0

    def test_no_warmup_means_no_first_bar_breach(self):
        iid = IIDGaussian().fit(np.random.default_rng(1).normal(0, 0.01, 500))
        eng = ForecastEngine({"iid_gaussian": StreamingIID(iid)}, alphas=(0.05,))
        _, breaches = eng.on_bar("d1", -0.9)  # no prior forecast exists
        assert breaches == []

    def test_breach_count_parity_with_batch(self):
        """Engine breach counts over a window == the batch pipeline's counts."""
        from marketlab.var_backtest import breaches as batch_breaches
        batch = Garch11().fit(RETURNS[:TEST_START])
        expected = int(batch_breaches(
            RETURNS[TEST_START:], batch.var_series(RETURNS, TEST_START, 0.05)
        ).sum())

        eng = ForecastEngine({"garch11": StreamingGarch11(batch)}, alphas=(0.05,))
        eng.warmup(RETURNS[:TEST_START])
        got = 0
        for t in range(TEST_START, len(RETURNS)):
            _, brs = eng.on_bar(f"d{t}", float(RETURNS[t]))
            got += len(brs)
        assert got == expected


def test_engine_two_alpha_rows(tiny_gpt):
    model, tokenizer = tiny_gpt
    eng = ForecastEngine(
        {"marketgpt": StreamingMarketGPT(model, tokenizer, 0)}, alphas=(0.05, 0.01)
    )
    eng.warmup(RETURNS[:200])
    forecasts, _ = eng.on_bar("d1", 0.002)
    row = forecasts[0]
    assert {"VaR95", "VaR99"} <= set(row)
    assert row["VaR99"] <= row["VaR95"]
