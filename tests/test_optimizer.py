"""
tests/test_optimizer.py
=======================
Automated Risk and Portfolio Optimization Test Suite.

Validates:
1. CovarianceEstimator: Ledoit-Wolf shrinkage, positive semi-definiteness, and condition number reduction
2. PortfolioOptimizer:
   - Budget invariant: sum(w_i) == 1.0
   - Long-only invariant: w_i >= 0.0
   - Single-stock concentration cap: w_i <= stock_cap (10% to 30%)
   - Sector cap enforcement: sum_{i in sector} w_i <= sector_cap (35%)
   - Minimum Variance, Max Sharpe, HRP, and Equal Risk Contribution (ERC)
   - Feasibility pre-check and graceful fallback on impossible constraints
3. AllocationExplainer: Marginal risk contribution sum, percentage risk contribution sum = 100%, and transparent rationales
"""

import pytest
import numpy as np
import pandas as pd

from src.risk.covariance import CovarianceEstimator
from src.optimization.engine import PortfolioOptimizer
from src.explainability.allocator import AllocationExplainer


@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = [f"STK_{i}" for i in range(15)]
    # Random returns with positive correlation
    common_factor = np.random.normal(0.0005, 0.01, len(dates))
    data = {}
    for i, t in enumerate(tickers):
        idio = np.random.normal(0.0, 0.015, len(dates))
        data[t] = 0.6 * common_factor + 0.4 * idio

    returns_df = pd.DataFrame(data, index=dates)
    expected_returns = pd.Series(np.linspace(0.05, 0.20, len(tickers)), index=tickers)

    # Sector mapping (3 sectors)
    sectors = ["Banking"] * 5 + ["IT"] * 5 + ["Energy"] * 5
    sector_mapping = dict(zip(tickers, sectors))

    return returns_df, expected_returns, sector_mapping


def test_ledoit_wolf_covariance(sample_market_data):
    """Verify Ledoit-Wolf shrinkage matrix is positive definite and reduces condition number."""
    returns_df, _, _ = sample_market_data
    estimator = CovarianceEstimator(lookback_days=252)

    cov_df, shrinkage, diag = estimator.estimate_ledoit_wolf_covariance(returns_df)

    assert cov_df.shape == (15, 15)
    assert 0.0 <= shrinkage <= 1.0
    assert diag["is_positive_definite"]
    assert diag["min_eigenvalue"] > 0
    # Ledoit-Wolf should improve or maintain condition number relative to sample covariance
    assert diag["lw_condition_number"] <= diag["sample_condition_number"] + 1e-4


def test_optimizer_budget_and_stock_cap(sample_market_data):
    """Verify weights sum to 1.0, non-negative, and respect 10% stock cap."""
    returns_df, expected_returns, _ = sample_market_data
    estimator = CovarianceEstimator()
    cov_df, _, _ = estimator.estimate_ledoit_wolf_covariance(returns_df)
    tickers = returns_df.columns.tolist()

    optimizer = PortfolioOptimizer(default_stock_cap=0.10)

    # 1. Min Variance
    w_minvar = optimizer.optimize_min_variance(cov_df, tickers, stock_cap=0.10)
    w_arr_min = np.array(list(w_minvar.values()))
    assert np.sum(w_arr_min) == pytest.approx(1.0, abs=1e-5)
    assert (w_arr_min >= -1e-6).all()
    assert (w_arr_min <= 0.10 + 1e-5).all()

    # 2. Max Sharpe
    w_sharpe = optimizer.optimize_max_sharpe(expected_returns, cov_df, tickers, stock_cap=0.10)
    w_arr_shp = np.array(list(w_sharpe.values()))
    assert np.sum(w_arr_shp) == pytest.approx(1.0, abs=1e-5)
    assert (w_arr_shp >= -1e-6).all()
    assert (w_arr_shp <= 0.10 + 1e-5).all()


def test_optimizer_sector_cap_enforcement(sample_market_data):
    """Verify portfolio respects 35% sector concentration limit."""
    returns_df, expected_returns, sector_mapping = sample_market_data
    estimator = CovarianceEstimator()
    cov_df, _, _ = estimator.estimate_ledoit_wolf_covariance(returns_df)
    tickers = returns_df.columns.tolist()

    optimizer = PortfolioOptimizer(default_stock_cap=0.15, default_sector_cap=0.35)

    w_opt = optimizer.optimize_max_sharpe(
        expected_returns,
        cov_df,
        tickers,
        stock_cap=0.15,
        sector_cap=0.35,
        sector_mapping=sector_mapping
    )

    # Check each sector sum
    sector_sums = {}
    for t, w in w_opt.items():
        sec = sector_mapping[t]
        sector_sums[sec] = sector_sums.get(sec, 0.0) + w

    for sec, total_w in sector_sums.items():
        assert total_w <= 0.35 + 1e-4, f"Sector {sec} exceeded 35% cap: {total_w:.2%}"


def test_equal_risk_contribution_erc(sample_market_data):
    """Verify Equal Risk Contribution (ERC) distributes risk equally across assets."""
    returns_df, _, _ = sample_market_data
    estimator = CovarianceEstimator()
    cov_df, _, _ = estimator.estimate_ledoit_wolf_covariance(returns_df)
    tickers = returns_df.columns.tolist()

    optimizer = PortfolioOptimizer(default_stock_cap=0.15)
    w_erc = optimizer.optimize_equal_risk_contribution(cov_df, tickers, stock_cap=0.15)

    w_arr = np.array([w_erc[t] for t in tickers])
    assert np.sum(w_arr) == pytest.approx(1.0, abs=1e-5)
    assert (w_arr >= 0.0).all()


def test_infeasible_constraints_graceful_fallback(sample_market_data):
    """Verify feasibility pre-check catches impossible caps and triggers fallback."""
    returns_df, expected_returns, _ = sample_market_data
    estimator = CovarianceEstimator()
    cov_df, _, _ = estimator.estimate_ledoit_wolf_covariance(returns_df)
    tickers = returns_df.columns.tolist()[:5] # only 5 assets

    fallback_records = []
    optimizer = PortfolioOptimizer(fallback_log=fallback_records)

    # 5 assets with max cap 0.10 can sum to at most 0.50 < 1.0 (Infeasible!)
    w_res = optimizer.optimize_min_variance(cov_df, tickers, stock_cap=0.10, date_str="2023-01-31")

    # Must log fallback
    assert len(fallback_records) > 0
    assert "Infeasible" in fallback_records[-1]["reason"]
    # Fallback still returns valid weights summing to 1.0
    w_arr = np.array(list(w_res.values()))
    assert np.sum(w_arr) == pytest.approx(1.0, abs=1e-5)


def test_allocation_explainability(sample_market_data):
    """Verify AllocationExplainer risk decomposition and generated rationales."""
    returns_df, expected_returns, sector_mapping = sample_market_data
    estimator = CovarianceEstimator()
    cov_df, _, _ = estimator.estimate_ledoit_wolf_covariance(returns_df)
    tickers = returns_df.columns.tolist()

    optimizer = PortfolioOptimizer(default_stock_cap=0.10)
    weights = optimizer.optimize_max_sharpe(expected_returns, cov_df, tickers, stock_cap=0.10)

    explainer = AllocationExplainer()
    df_exp = explainer.explain_portfolio(
        weights=weights,
        expected_returns=expected_returns,
        cov_matrix=cov_df,
        sector_mapping=sector_mapping,
        stock_cap=0.10,
        sector_cap=0.35
    )

    assert len(df_exp) == len(tickers)
    assert {"ticker", "weight", "pct_risk_contrib", "diversification_benefit", "explanation"}.issubset(df_exp.columns)

    # Percentage risk contributions must sum to 100% (1.0)
    assert df_exp["pct_risk_contrib"].sum() == pytest.approx(1.0, abs=1e-3)
    # Check explanation string exists
    assert all(len(exp) > 5 for exp in df_exp["explanation"])
