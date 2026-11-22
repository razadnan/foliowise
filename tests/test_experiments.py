"""
tests/test_experiments.py
=========================
Unit tests for Milestone M6: SmartFolio Experiment Framework & Stationary Bootstrap.
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.bootstrap import (
    compute_annualized_sharpe,
    stationary_block_bootstrap_indices,
    StationaryBootstrapTester
)


def test_compute_annualized_sharpe():
    # Constant daily returns 0.08% with small noise
    rng = np.random.default_rng(42)
    daily_rets = 0.0008 + rng.normal(0, 0.005, 252)
    sharpe = compute_annualized_sharpe(daily_rets, rf=0.06)
    assert sharpe > 1.0


def test_stationary_bootstrap_indices():
    n_obs = 100
    indices = stationary_block_bootstrap_indices(n_obs, avg_block_len=10.0)
    assert len(indices) == n_obs
    assert np.all(indices >= 0) and np.all(indices < n_obs)
    # Check that contiguous indices appear frequently (block property)
    diffs = np.diff(indices)
    contiguous_count = np.sum(diffs == 1) + np.sum(diffs == -(n_obs - 1))
    assert contiguous_count > 50  # With p=0.1, ~90% should be contiguous transitions


def test_bootstrap_identical_series():
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, 500)
    tester = StationaryBootstrapTester(n_bootstraps=200, avg_block_length=5.0, seed=42)
    res = tester.test_sharpe_difference(rets, rets, name_a="Series A", name_b="Series B")
    
    assert res["observed_diff"] == pytest.approx(0.0, abs=1e-5)
    assert res["ci_95_lower"] <= 0.0 <= res["ci_95_upper"]
    assert not res["significant_5pct"]


def test_bootstrap_superior_series():
    rng = np.random.default_rng(42)
    rets_a = rng.normal(0.0015, 0.008, 1000)  # Much higher Sharpe
    rets_b = rng.normal(0.0002, 0.012, 1000)
    tester = StationaryBootstrapTester(n_bootstraps=300, avg_block_length=5.0, seed=42)
    res = tester.test_sharpe_difference(rets_a, rets_b, name_a="Superior", name_b="Inferior")
    
    assert res["observed_diff"] > 1.0
    assert res["ci_95_lower"] > 0.0
    assert res["significant_5pct"]


def test_benchmark_hrp():
    from src.optimization.engine import PortfolioOptimizer
    optimizer = PortfolioOptimizer(default_stock_cap=0.10)
    tickers = [f"STOCK_{i}" for i in range(20)]
    rng = np.random.default_rng(42)
    cov_arr = rng.standard_normal((20, 20))
    cov_arr = cov_arr.T @ cov_arr + np.eye(20) * 0.1
    cov_df = pd.DataFrame(cov_arr, index=tickers, columns=tickers)

    w = optimizer.optimize_hrp(cov_df, tickers=tickers, stock_cap=0.10)
    assert len(w) == 20
    assert pytest.approx(sum(w.values()), abs=1e-4) == 1.0
    assert all(val >= -1e-6 for val in w.values())
    assert all(val <= 0.10 + 1e-4 for val in w.values())


def test_benchmark_min_variance():
    from src.optimization.engine import PortfolioOptimizer
    optimizer = PortfolioOptimizer(default_stock_cap=0.10)
    tickers = [f"STOCK_{i}" for i in range(20)]
    sectors = {t: f"Sector_{i % 4}" for i, t in enumerate(tickers)}
    rng = np.random.default_rng(42)
    cov_arr = rng.standard_normal((20, 20))
    cov_arr = cov_arr.T @ cov_arr + np.eye(20) * 0.1
    cov_df = pd.DataFrame(cov_arr, index=tickers, columns=tickers)

    w = optimizer.optimize_min_variance(cov_df, tickers=tickers, stock_cap=0.10, sector_cap=0.35, sector_mapping=sectors)
    assert len(w) == 20
    assert pytest.approx(sum(w.values()), abs=1e-4) == 1.0
    assert all(val >= -1e-6 for val in w.values())
    assert all(val <= 0.10 + 1e-4 for val in w.values())


def test_benchmark_erc():
    from src.optimization.engine import PortfolioOptimizer
    optimizer = PortfolioOptimizer(default_stock_cap=0.10)
    tickers = [f"STOCK_{i}" for i in range(20)]
    rng = np.random.default_rng(42)
    cov_arr = rng.standard_normal((20, 20))
    cov_arr = cov_arr.T @ cov_arr + np.eye(20) * 0.1
    cov_df = pd.DataFrame(cov_arr, index=tickers, columns=tickers)

    w = optimizer.optimize_equal_risk_contribution(cov_df, tickers=tickers, stock_cap=0.10)
    assert len(w) == 20
    assert pytest.approx(sum(w.values()), abs=1e-4) == 1.0
    assert all(val >= -1e-6 for val in w.values())
    assert all(val <= 0.10 + 1e-4 for val in w.values())


def test_benchmark_momentum_top_k():
    from src.optimization.engine import PortfolioOptimizer
    optimizer = PortfolioOptimizer(default_stock_cap=0.10)
    tickers = [f"STOCK_{i}" for i in range(20)]
    scores = pd.Series(range(20), index=tickers)  # STOCK_19 is highest

    w = optimizer.optimize_momentum_top_k(scores, tickers=tickers, k=5)
    assert len(w) == 20
    assert pytest.approx(sum(w.values()), abs=1e-4) == 1.0
    non_zero = [t for t, val in w.items() if val > 1e-4]
    assert len(non_zero) == 5
    for t in non_zero:
        assert pytest.approx(w[t], abs=1e-4) == 0.20
    assert "STOCK_19" in non_zero
