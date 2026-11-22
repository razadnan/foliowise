"""
tests/test_backtest.py
======================
Unit tests for Milestone M5: Backtesting Engine and Cost Model.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.cost_model import IndianTransactionCostModel
from src.backtesting.engine import BacktestEngine


def test_cost_model_breakdown():
    cost_model = IndianTransactionCostModel()
    trade_val = 1_000_000.0  # 10 Lakhs INR
    
    buy_itemized = cost_model.get_itemized_breakdown(trade_val, is_buy=True)
    assert buy_itemized["stt"] == pytest.approx(1000.0) # 10 bps STT
    assert buy_itemized["stamp_duty"] == pytest.approx(150.0) # 1.5 bps stamp duty
    assert buy_itemized["exchange_charges"] == pytest.approx(32.50) # 0.00325%
    assert buy_itemized["grand_total_inr"] > 0
    assert buy_itemized["effective_bps"] > 10.0 # Statutory + slippage > 10 bps
    
    sell_itemized = cost_model.get_itemized_breakdown(trade_val, is_buy=False)
    assert sell_itemized["stamp_duty"] == 0.0 # Stamp duty is buy only


def test_turnover_computation():
    engine = BacktestEngine()
    w1 = np.array([0.5, 0.5, 0.0])
    w2 = np.array([0.0, 0.5, 0.5])
    # 0.5 * (|0 - 0.5| + |0.5 - 0.5| + |0.5 - 0|) = 0.5 * (0.5 + 0 + 0.5) = 0.5
    turnover = engine.compute_turnover(w2, w1)
    assert turnover == pytest.approx(0.50)


def test_weight_drift():
    engine = BacktestEngine()
    w0 = np.array([0.5, 0.5])
    # Stock 0 doubles (+100%), Stock 1 stays flat (0%)
    rets = np.array([1.0, 0.0])
    w_drift = engine.drift_weights(w0, rets)
    # unnorm = [0.5*2, 0.5*1] = [1.0, 0.5], total = 1.5 -> [2/3, 1/3]
    assert w_drift[0] == pytest.approx(2.0 / 3.0)
    assert w_drift[1] == pytest.approx(1.0 / 3.0)
    assert np.sum(w_drift) == pytest.approx(1.0)


def test_next_day_execution_timing():
    """
    Verify that if a signal is generated at T, the return on T+1 is earned by the
    OLD portfolio, and the new target weights only apply after T+1 close.
    """
    engine = BacktestEngine()
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    
    # 2 assets: A and B
    # Day 0: Signal generated at close to allocate 100% to Asset A
    # Day 1: Asset A drops -10%, Asset B rises +10%
    # If Next-Day execution is enforced:
    # On Day 1, portfolio was not yet invested in Asset A until Day 1 close!
    # So Day 1 return is 0.0 (or cash/previous weight), NOT Asset A return.
    returns_df = pd.DataFrame({
        "A": [0.0, -0.10, 0.05, 0.02, 0.01],
        "B": [0.0,  0.10, -0.05, 0.01, 0.02]
    }, index=dates)
    
    monthly_weights = {
        dates[0]: {"A": 1.0, "B": 0.0}
    }
    
    df_daily, _, _ = engine.simulate_strategy("Test_Strategy", returns_df, monthly_weights)
    
    # Day 0 return should be 0 (uninvested)
    assert df_daily.loc[0, "gross_return"] == 0.0
    # Day 1 return should still be 0 (rebalance executes at Day 1 close)
    assert df_daily.loc[1, "gross_return"] == 0.0
    # Day 2 return should be +0.05 (Asset A return on Day 2)
    assert df_daily.loc[2, "gross_return"] == pytest.approx(0.05)


def test_metrics_calculation():
    engine = BacktestEngine(rf=0.06)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    # Constant 0.08% daily return (~22% annualized)
    net_rets = np.full(252, 0.0008)
    gross_rets = net_rets.copy()
    
    df_daily = pd.DataFrame({
        "date": dates,
        "strategy": "Dummy",
        "gross_return": gross_rets,
        "net_return": net_rets,
        "turnover": np.zeros(252),
        "cost": np.zeros(252),
        "cum_gross": (1.0 + gross_rets).cumprod() * 100.0,
        "cum_net": (1.0 + net_rets).cumprod() * 100.0
    })
    
    metrics = engine.calculate_metrics(df_daily, "Dummy")
    assert metrics["cagr"] > 20.0
    assert metrics["sharpe_ratio"] > 1.0
    assert metrics["max_drawdown"] == 0.0  # Monotonically increasing
