"""
tests/test_e2e.py
=================
Milestone M10: End-to-End Integration and System Regression Test.

Validates the full integrated pipeline in a single deterministic test flow:
Universe/Data -> Features -> ML Models -> Risk & Portfolio Optimization ->
Next-Day Backtest -> Adaptive Feedback -> Allocation Explainability.
"""

import numpy as np
import pandas as pd
import pytest

from src.universe.resolver import UniverseResolver
from src.features.pipeline import FeaturePipeline
from src.features.target import TargetGenerator
from src.models.ridge import RidgeModel
from src.risk.covariance import CovarianceEstimator
from src.optimization.engine import PortfolioOptimizer
from src.backtesting.cost_model import IndianTransactionCostModel
from src.backtesting.engine import BacktestEngine
from src.feedback.engine import FeedbackEngine
from src.explainability.allocator import AllocationExplainer


def test_full_pipeline_e2e(tmp_path):
    # -------------------------------------------------------------------------
    # 1. Universe & Data Stage
    # -------------------------------------------------------------------------
    resolver = UniverseResolver()
    universe_2020 = resolver.get_constituents_df("2020-01-31")
    universe_2024 = resolver.get_constituents_df("2024-01-31")
    assert len(universe_2020) > 0
    assert len(universe_2024) > 0
    
    # -------------------------------------------------------------------------
    # 2. Synthetic Market Generation (2 Tickers, 400 business days)
    # -------------------------------------------------------------------------
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    rng = np.random.default_rng(123)
    
    # Stock 1 (Trending) & Stock 2 (Mean-reverting)
    p1 = 100.0 * np.cumprod(1.0 + rng.normal(0.0008, 0.012, len(dates)))
    p2 = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.015, len(dates)))
    vix = 15.0 + rng.normal(0.0, 1.0, len(dates)).clip(10, 30)
    idx_p = 1000.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.010, len(dates)))
    
    piv_prices = pd.DataFrame({"AAA": p1, "BBB": p2}, index=dates)
    piv_volumes = pd.DataFrame({"AAA": 1000000.0, "BBB": 500000.0}, index=dates)
    nifty_s = pd.Series(idx_p, index=dates)
    vix_s = pd.Series(vix, index=dates)
    
    # -------------------------------------------------------------------------
    # 3. Features & Target Pipeline
    # -------------------------------------------------------------------------
    feat_pipe = FeaturePipeline()
    labeled_df = feat_pipe.build_features_table(
        prices_df=piv_prices,
        volume_df=piv_volumes,
        nifty_prices=nifty_s,
        vix_series=vix_s,
        min_history=100
    )
    assert not labeled_df.empty
    assert "target_rank_20d" in labeled_df.columns
    
    # -------------------------------------------------------------------------
    # 4. ML Walk-Forward Model Training
    # -------------------------------------------------------------------------
    feature_cols = [c for c in labeled_df.columns if c.startswith("f_")]
    valid_df = labeled_df.dropna(subset=["target_rank_20d"] + feature_cols).copy()
    assert len(valid_df) > 50
    
    split_idx = int(len(valid_df) * 0.6)
    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df["target_rank_20d"].values
    X_test = test_df[feature_cols].values
    
    ridge = RidgeModel(alphas=[1.0])
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    assert len(preds) == len(test_df)
    
    # -------------------------------------------------------------------------
    # 5. Risk Estimation & Portfolio Optimization
    # -------------------------------------------------------------------------
    cov_est = CovarianceEstimator(lookback_days=100)
    piv_ret = piv_prices.pct_change().dropna()
    lw_cov, _, _ = cov_est.estimate_ledoit_wolf_covariance(piv_ret.iloc[:200])
    
    optimizer = PortfolioOptimizer(default_stock_cap=0.60, default_sector_cap=0.80)
    expected_alpha = pd.Series({"AAA": float(np.mean(preds[:10])), "BBB": float(np.mean(preds[10:]))})
    
    weights = optimizer.optimize_max_sharpe(
        expected_returns=expected_alpha,
        cov_matrix=lw_cov,
        tickers=["AAA", "BBB"],
        stock_cap=0.60
    )
    assert weights["AAA"] + weights["BBB"] == pytest.approx(1.0, abs=1e-4)
    assert weights["AAA"] <= 0.60 + 1e-4
    assert weights["BBB"] <= 0.60 + 1e-4
    
    # -------------------------------------------------------------------------
    # 6. Next-Day Backtesting Engine
    # -------------------------------------------------------------------------
    cost_model = IndianTransactionCostModel(base_rate=0.0020)
    engine = BacktestEngine(cost_model=cost_model, rf=0.06)
    
    test_ret_piv = piv_ret.iloc[200:]
    rebal_dates = [test_ret_piv.index[0]]
    monthly_weights = {rebal_dates[0]: weights}
    
    daily_results, _, metrics = engine.simulate_strategy(
        strategy_name="E2E_Test_Strategy",
        returns_df=test_ret_piv,
        monthly_target_weights=monthly_weights
    )
    assert not daily_results.empty
    assert "net_return" in daily_results.columns
    assert "cagr" in metrics
    assert "sharpe_ratio" in metrics
    
    # -------------------------------------------------------------------------
    # 7. Adaptive Feedback & Allocation Explainability
    # -------------------------------------------------------------------------
    explainer = AllocationExplainer(stock_cap=0.60, sector_cap=0.80)
    df_exp = explainer.explain_portfolio(
        weights=weights,
        expected_returns=expected_alpha,
        cov_matrix=lw_cov,
        sector_mapping={"AAA": "Tech", "BBB": "Energy"}
    )
    assert len(df_exp) == 2
    assert "pct_risk_contrib" in df_exp.columns
    assert "explanation" in df_exp.columns
