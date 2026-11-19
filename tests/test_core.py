"""
tests/test_core.py
==================
Unit tests for core portfolio logic:
- Turnover computation
- Transaction cost deduction
- Weight drift under asset returns
- Concentration cap invariants
- Data schema and alignment
"""

import os
import pytest
import numpy as np
import pandas as pd


def compute_turnover(target_w: np.ndarray, current_w: np.ndarray) -> float:
    """Computes one-way turnover (0.5 * L1 norm) between target and current weights."""
    return float(0.5 * np.sum(np.abs(target_w - current_w)))


def compute_drifted_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Computes post-return drifted weights."""
    unnorm = weights * (1.0 + returns)
    total = np.sum(unnorm)
    if total > 0:
        return unnorm / total
    return weights


def compute_net_return(gross_ret: float, turnover: float, cost: float = 0.002) -> float:
    """Computes net return after deducting turnover transaction friction."""
    cost_penalty = cost * turnover if turnover > 0 else 0.0
    return gross_ret - cost_penalty


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_turnover_and_cost_rebalancing():
    """Weights [0.5, 0.5] going to [1.0, 0.0] yield one-way turnover 0.5 and cost 0.001."""
    current_w = np.array([0.5, 0.5])
    target_w = np.array([1.0, 0.0])
    
    turnover = compute_turnover(target_w, current_w)
    assert turnover == pytest.approx(0.5, abs=1e-7)
    
    cost = 0.002 # 20 bps
    cost_penalty = cost * turnover
    assert cost_penalty == pytest.approx(0.001, abs=1e-7)
    
    gross_ret = 0.010
    net_ret = compute_net_return(gross_ret, turnover, cost)
    assert net_ret == pytest.approx(0.009, abs=1e-7)


def test_zero_turnover_when_unchanged():
    """Identical weights produce 0.0 turnover and zero cost."""
    w = np.array([0.25, 0.25, 0.25, 0.25])
    turnover = compute_turnover(w, w)
    assert turnover == pytest.approx(0.0, abs=1e-7)
    
    net_ret = compute_net_return(0.015, turnover, cost=0.002)
    assert net_ret == pytest.approx(0.015, abs=1e-7)


def test_weight_drift_mechanics():
    """Weights drift proportionally to daily asset price movements."""
    w_initial = np.array([0.5, 0.5])
    # Asset 0 gains 10%, Asset 1 loses 10%
    returns = np.array([0.10, -0.10])
    
    w_drifted = compute_drifted_weights(w_initial, returns)
    # 0.5 * 1.10 = 0.55; 0.5 * 0.90 = 0.45; total = 1.0
    assert w_drifted[0] == pytest.approx(0.55, abs=1e-7)
    assert w_drifted[1] == pytest.approx(0.45, abs=1e-7)
    assert np.sum(w_drifted) == pytest.approx(1.0, abs=1e-7)


def test_weight_drift_asymmetric_gains():
    """Weights re-normalize properly when portfolio value grows."""
    w_initial = np.array([0.5, 0.5])
    # Asset 0 flat (0%), Asset 1 doubles (+100%)
    returns = np.array([0.00, 1.00])
    
    w_drifted = compute_drifted_weights(w_initial, returns)
    # Asset 0: 0.5 / 1.5 = 1/3; Asset 1: 1.0 / 1.5 = 2/3
    assert w_drifted[0] == pytest.approx(1.0 / 3.0, abs=1e-7)
    assert w_drifted[1] == pytest.approx(2.0 / 3.0, abs=1e-7)
    assert np.sum(w_drifted) == pytest.approx(1.0, abs=1e-7)


def test_backtest_weights_file_invariants():
    """Verify results/weights.csv conforms to sum-to-1 and concentration caps."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, "results", "weights.csv")
    
    assert os.path.exists(weights_path), f"weights.csv not found at {weights_path}"
    df = pd.read_csv(weights_path)
    
    # Check sum to 1.0 for each date and strategy
    sums = df.groupby(["date", "strategy"])["weight"].sum()
    assert (sums >= 0.999).all() and (sums <= 1.001).all()
    
    # Check caps: S1 <= 0.05, S2-S4,S6 <= 0.1501, S5 <= 0.2001
    max_weights = df.groupby("strategy")["weight"].max()
    assert max_weights["S1"] <= 0.040
    assert max_weights["S2"] <= 0.1501
    assert max_weights["S3"] <= 0.1501
    assert max_weights["S4"] <= 0.1501
    assert max_weights["S5"] <= 0.2001
    assert max_weights["S6"] <= 0.1501
    assert max_weights["S7"] <= 0.2001


def test_data_schema_and_date_synchronization():
    """Verify raw data schemas and synchronization across stages."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prices_path = os.path.join(base_dir, "data", "raw", "prices.csv")
    index_path = os.path.join(base_dir, "data", "raw", "index.csv")
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")
    daily_ret_path = os.path.join(base_dir, "results", "daily_returns.csv")
    
    # 1. Prices schema
    pdf = pd.read_csv(prices_path)
    assert list(pdf.columns) == ["date", "ticker", "adj_close", "volume", "filled"]
    assert pdf["ticker"].nunique() == 27
    assert pdf["filled"].dtype == bool or set(pdf["filled"].unique()).issubset({True, False})
    
    # 2. Index schema
    idf = pd.read_csv(index_path)
    assert list(idf.columns) == ["date", "adj_close"]
    
    # 3. Synchronized max dates
    fdf = pd.read_csv(features_path)
    ddf = pd.read_csv(daily_ret_path)
    
    p_max = pdf["date"].max()
    f_max = fdf["date"].max()
    d_max = ddf["date"].max()
    assert p_max == f_max == d_max, f"Date mismatch: prices={p_max}, features={f_max}, daily_rets={d_max}"
