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
