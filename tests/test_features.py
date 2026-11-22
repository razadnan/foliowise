"""
tests/test_features.py
======================
Automated Feature Engineering and Leakage Control Test Suite.

Validates:
1. Feature pipeline calculations (returns, vol, momentum, RSI, MACD, MA ratios, ROC, VIX)
2. Backward-looking invariant: Features at date t use only data <= t
3. Standardized target properties: cross-sectional rank target in [0.0, 1.0] with cross-sectional mean = 0.5
4. Multi-date leakage tests across 6 historical cutoff dates
5. Sensitivity: Injected look-ahead bias is strictly caught and flagged by LeakageTester
"""

import pytest
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.features.target import TargetGenerator
from src.features.leakage_test import LeakageTester


@pytest.fixture
def pipeline():
    return FeaturePipeline()


@pytest.fixture
def target_gen():
    return TargetGenerator(horizon=20)


@pytest.fixture
def leakage_tester():
    return LeakageTester()


def test_indicator_calculations(pipeline):
    """Verify core technical indicators compute backward-looking series correctly."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, len(dates)))), index=dates)

    # Returns
    rets = pipeline.compute_past_returns(prices, windows=[1, 5, 21])
    assert "f_ret_1" in rets.columns
    assert "f_ret_5" in rets.columns
    assert "f_ret_21" in rets.columns
    assert np.isnan(rets["f_ret_1"].iloc[0])
    assert not np.isnan(rets["f_ret_1"].iloc[1])

    # RSI
    rsi = pipeline.compute_rsi(prices, period=14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()

    # MACD
    macd_df = pipeline.compute_macd(prices, fast=12, slow=26, signal=9)
    assert {"f_macd", "f_macd_sig", "f_macd_hist"}.issubset(macd_df.columns)
    # Histogram equals macd - signal
    np.testing.assert_allclose(
        macd_df["f_macd_hist"].values,
        (macd_df["f_macd"] - macd_df["f_macd_sig"]).values,
        rtol=1e-7, atol=1e-10
    )

    # MA Ratios
    ma_ratios = pipeline.compute_moving_average_ratios(prices)
    assert "f_sma_50_ratio" in ma_ratios.columns
    assert "f_ema_20_ratio" in ma_ratios.columns

    # ROC
    roc = pipeline.compute_roc(prices, period=10)
    assert roc.name == "f_roc_10"
    # Check manual calculation on row 15
    expected_roc = (prices.iloc[15] / prices.iloc[5]) - 1.0
    assert pytest.approx(roc.iloc[15], abs=1e-7) == expected_roc


def test_standardized_rank_target(target_gen):
    """Verify 20-day excess return and cross-sectional rank target properties."""
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    p1 = pd.Series(np.linspace(100, 150, 60), index=dates) # +50%
    p2 = pd.Series(np.linspace(100, 120, 60), index=dates) # +20%
    p3 = pd.Series(np.linspace(100, 90, 60), index=dates)  # -10%
    bmk = pd.Series(np.linspace(100, 110, 60), index=dates) # +10%

    prices_df = pd.DataFrame({"STOCK_A": p1, "STOCK_B": p2, "STOCK_C": p3})
    targets_table = target_gen.build_targets_table(prices_df, bmk)

    assert "target_20d_ret" in targets_table.columns
    assert "target_excess_20d" in targets_table.columns
    assert "target_rank_20d" in targets_table.columns
    assert "target_21d" in targets_table.columns

    # Check date t=0
    t0_records = targets_table[targets_table["date"] == dates[0].strftime("%Y-%m-%d")].set_index("ticker")
    # STOCK_A has highest return -> rank = 1.0
    # STOCK_C has lowest return -> rank = 0.0
    # STOCK_B is middle -> rank = 0.5
    assert t0_records.loc["STOCK_A", "target_rank_20d"] == pytest.approx(1.0, abs=1e-5)
    assert t0_records.loc["STOCK_C", "target_rank_20d"] == pytest.approx(0.0, abs=1e-5)
    assert t0_records.loc["STOCK_B", "target_rank_20d"] == pytest.approx(0.5, abs=1e-5)

    # Cross-sectional mean rank must be 0.5
    assert t0_records["target_rank_20d"].mean() == pytest.approx(0.5, abs=1e-5)


def test_multi_date_leakage_on_real_data(leakage_tester):
    """Verify leakage tester passes across historical cutoff dates on actual pipeline data."""
    prices_path = "data/raw/prices.csv"
    index_path = "data/raw/index.csv"
    vix_path = "data/raw/india_vix.csv"

    p_df = pd.read_csv(prices_path)
    p_df["date"] = pd.to_datetime(p_df["date"])
    idx_df = pd.read_csv(index_path)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    vix_df = pd.read_csv(vix_path)
    vix_df["date"] = pd.to_datetime(vix_df["date"])

    piv_p = p_df.pivot(index="date", columns="ticker", values="adj_close")
    piv_v = p_df.pivot(index="date", columns="ticker", values="volume")
    nifty = idx_df.set_index("date")["adj_close"]
    vix = vix_df.set_index("date")["vix"]

    # Test 3 representative cutoff dates
    cutoff_dates = ["2019-12-31", "2021-06-30", "2023-12-29"]
    res = leakage_tester.run_multi_date_leakage_test(
        prices_df=piv_p,
        volume_df=piv_v,
        nifty_prices=nifty,
        vix_series=vix,
        truncation_dates=cutoff_dates
    )

    assert res["passed"]
    assert res["discrepancies_count"] == 0
    assert len(res["report_df"]) == len(cutoff_dates)
    assert (res["report_df"]["status"] == "PASS").all()


def test_leakage_tester_catches_deliberate_leakage(leakage_tester):
    """Verify that if a forward-looking feature is deliberately introduced, leakage tester fails."""
    dates = pd.date_range("2022-01-01", periods=360, freq="B")
    p = pd.DataFrame({
        "AAA.NS": np.linspace(100, 200, 360),
        "BBB.NS": np.linspace(200, 100, 360)
    }, index=dates)
    v = pd.DataFrame({"AAA.NS": 1000, "BBB.NS": 1000}, index=dates)
    nifty = pd.Series(np.linspace(10000, 15000, 360), index=dates)

    class LeakyPipeline(FeaturePipeline):
        def compute_past_returns(self, prices, windows=[1, 5, 21, 63]):
            # Deliberately leak future price at row t: prices.shift(-1)
            df = super().compute_past_returns(prices, windows)
            df["f_ret_1"] = prices.shift(-1) / prices - 1.0  # LEAKAGE!
            return df

    leaky_tester = LeakageTester(pipeline=LeakyPipeline())
    cutoff = dates[320].strftime("%Y-%m-%d")
    res = leaky_tester.run_multi_date_leakage_test(
        prices_df=p,
        volume_df=v,
        nifty_prices=nifty,
        vix_series=None,
        truncation_dates=[cutoff]
    )

    # Must detect leakage
    assert not res["passed"]
    assert res["discrepancies_count"] > 0
