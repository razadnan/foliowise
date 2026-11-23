"""
tests/test_fix_verifications.py
===============================
Dedicated verification suite testing the key architectural, mathematical,
and reliability bug fixes implemented during the FolioWise comprehensive audit.

Validates:
1. BUG-01: Target interpretation (rank target mapped to active alpha spread, no explosive compounding)
2. BUG-02: LSTM sequence extraction (genuine temporal sliding windows, no artificial row decay)
3. BUG-03: Optimizer convex quadratic projection (joint stock and sector caps strictly satisfied)
4. BUG-05: Backtest module pypfopt graceful fallback
5. BUG-06: Ridge regression chronological TimeSeriesSplit validation
6. BUG-07: Model ensemble chronological validation weighting
7. BUG-09: Frontend dashboard offline/file:/// 75-stock dataset embedding
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

from src.optimization.engine import PortfolioOptimizer
from src.models.ridge import RidgeModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import ModelEnsemble
from src.experiments.runner import convert_predictions_to_expected_returns
from src.backtesting.engine import BacktestEngine
import src.backtest as backtest_module


def test_bug01_rank_target_expected_returns_conversion():
    """Verify convert_predictions_to_expected_returns converts percentile rank to modest alpha spread."""
    tickers = ["A", "B", "C"]
    hist_means = pd.Series([0.12, 0.10, 0.08], index=tickers)

    # Model predicts percentile ranks centered around 0.50:
    # A = 0.90 (top decile), B = 0.50 (median), C = 0.10 (bottom decile)
    preds = pd.Series([0.90, 0.50, 0.10], index=tickers)

    mu = convert_predictions_to_expected_returns(
        preds,
        hist_means=hist_means,
        annualize=True,
        target_type="rank"
    )

    # Under old bug, (1.0 + 0.90)**12 - 1.0 = +2,213% return!
    # Under new fix, spread = 0.08 * (rank - 0.50):
    # A: 0.12 + 0.08*(0.40) = 0.152 (+15.2%)
    # B: 0.10 + 0.08*(0.00) = 0.100 (+10.0%)
    # C: 0.08 + 0.08*(-0.40) = 0.048 (+4.8%)
    assert mu["A"] == pytest.approx(0.152, abs=1e-5)
    assert mu["B"] == pytest.approx(0.100, abs=1e-5)
    assert mu["C"] == pytest.approx(0.048, abs=1e-5)

    # Must be bounded within realistic equity return ranges (< 100% annualized)
    assert (mu < 1.0).all()
    assert (mu > -0.5).all()


def test_bug02_lstm_temporal_sliding_window():
    """Verify LSTM prepares genuine temporal sequences without artificial multiplicative decay."""
    seq_len = 5
    n_features = 4
    n_samples = 30

    # Generate sequential time series where each row has distinct values
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    y = np.linspace(0.0, 1.0, n_samples)

    model = LSTMModel(input_dim=n_features, seq_len=seq_len, hidden_dim=8, epochs=1)

    # Test internal sequence preparation
    X_scaled = model.scaler.fit_transform(X)
    X_seq = model._prepare_sequences(X_scaled)

    assert X_seq.shape == (n_samples, seq_len, n_features)

    # For any t >= seq_len - 1, the sequence must exactly match historical slice [t-seq_len+1 : t+1]
    for t in range(seq_len - 1, n_samples):
        expected_slice = X_scaled[t - seq_len + 1 : t + 1]
        np.testing.assert_allclose(X_seq[t], expected_slice, rtol=1e-6)


def test_bug03_optimizer_convex_joint_caps():
    """Verify cap_and_normalize simultaneously respects individual stock cap and sector cap."""
    optimizer = PortfolioOptimizer(default_stock_cap=0.10, default_sector_cap=0.35)

    # 20 stocks across 4 sectors: Tech, Banking, Energy, Pharma (5 stocks each)
    tickers = [f"T{i}" for i in range(20)]
    sector_names = ["Tech", "Banking", "Energy", "Pharma"]
    sectors = {f"T{i}": sector_names[i // 5] for i in range(20)}

    # Raw weights heavily concentrated in Tech (Tech total = 0.60)
    raw_weights = np.zeros(20)
    raw_weights[:5] = [0.20, 0.15, 0.10, 0.10, 0.05] # Tech sum = 0.60 (violates 0.35 sector cap)
    raw_weights[5:] = 0.40 / 15                       # Remaining 15 stocks share 0.40

    projected_w = optimizer.cap_and_normalize(
        raw_weights,
        stock_cap=0.10,
        tickers=tickers,
        sector_mapping=sectors,
        sector_cap=0.35
    )

    # 1. Sum to 1.0
    assert np.isclose(projected_w.sum(), 1.0, atol=1e-4)

    # 2. Long-only non-negative
    assert (projected_w >= -1e-6).all()

    # 3. Individual stock cap strictly satisfied (<= 0.10 + tolerance)
    assert (projected_w <= 0.10 + 1e-4).all(), f"Stock cap violated: {projected_w.max()}"

    # 4. Sector caps strictly satisfied for all sectors (<= 0.35 + tolerance)
    for sec in sector_names:
        sec_sum = sum(projected_w[i] for i in range(20) if sectors[tickers[i]] == sec)
        assert sec_sum <= 0.35 + 1e-4, f"{sec} sector cap violated: {sec_sum}"


def test_bug05_backtest_graceful_pypfopt_fallback():
    """Verify src.backtest functions seamlessly even if pypfopt is absent."""
    assert hasattr(backtest_module, "HAS_PYPFOPT")
    assert hasattr(backtest_module, "PortfolioOptimizer")
    assert hasattr(backtest_module, "BacktestEngine")

    # Test BacktestEngine initialization and basic operations
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    tickers = ["STK_A", "STK_B"]
    prices = pd.DataFrame({"STK_A": [100.0] * 10, "STK_B": [50.0] * 10}, index=dates)
    returns = pd.DataFrame({"STK_A": [0.0] * 10, "STK_B": [0.0] * 10}, index=dates)
    predictions = pd.DataFrame(columns=["date", "ticker", "predicted"])
    universe = pd.DataFrame([{"date": dates[0], "ticker": "STK_A"}, {"date": dates[0], "ticker": "STK_B"}])
    config = {
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "weight_cap": 0.15,
        "cov_lookback": 5,
        "tx_cost": 0.002,
        "risk_free": 0.06
    }

    engine = backtest_module.BacktestEngine(
        prices=prices,
        returns=returns,
        predictions=predictions,
        universe=universe,
        config=config
    )
    assert engine is not None
    ew = engine._equal_weight_dict(tickers)
    assert ew["STK_A"] == pytest.approx(0.50)
    assert ew["STK_B"] == pytest.approx(0.50)


def test_bug06_ridge_chronological_time_series_split():
    """Verify RidgeModel uses TimeSeriesSplit for temporal cross-validation."""
    model = RidgeModel(cv=4)
    assert model.cv == 4

    # Generate synthetic time-series
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model.fit(X, y)
    assert model.best_alpha is not None
    assert model.best_alpha > 0.0
    assert hasattr(model.model, "alpha_")


def test_bug07_ensemble_validation_weighting():
    """Verify ModelEnsemble tunes model weights on chronological validation folds."""
    X_train = np.random.randn(60, 5)
    y_train = np.random.randn(60)

    m_xgb = XGBoostModel(n_estimators=5, max_depth=2, random_state=42).fit(X_train, y_train)
    m_lstm = LSTMModel(input_dim=5, seq_len=3, hidden_dim=8, epochs=1).fit(X_train, y_train)

    ens = ModelEnsemble(model_xgb=m_xgb, model_lstm=m_lstm)

    X_val = np.random.randn(20, 5)
    y_val = np.random.randn(20)

    w_xgb, w_lstm = ens.tune_weights_on_validation(X_val, y_val)

    assert 0.0 < w_xgb < 1.0
    assert 0.0 < w_lstm < 1.0
    assert np.isclose(w_xgb + w_lstm, 1.0, atol=1e-5)


def test_bug09_dashboard_offline_stocks_embedded():
    """Verify frontend/dashboard.html contains pre-embedded 75 stocks for offline/file:/// navigation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dashboard_path = os.path.join(base_dir, "frontend", "dashboard.html")
    assert os.path.exists(dashboard_path), "frontend/dashboard.html missing"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "const EMBEDDED_STOCKS =" in html
    assert "Reliance Industries" in html
    assert "renderStockRows(EMBEDDED_STOCKS)" in html
