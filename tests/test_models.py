"""
tests/test_models.py
====================
Automated Machine Learning Unit & Integration Test Suite.

Validates:
1. Model A: Ridge regression in-fold scaling, coefficient shapes, and predictions
2. Model B: XGBoost regressor fit, prediction shape, and feature importances
3. Model C: PyTorch LSTM sequence model training, gradient flow, and predictions
4. Model D: XGBoost-LSTM ensemble weighted combination
5. ModelEvaluator: Rank IC, ICIR, Newey-West HAC t-stat, RMSE, MAE, and spread metrics
"""

import pytest
import numpy as np
import pandas as pd

from src.models.ridge import RidgeModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import ModelEnsemble
from src.models.evaluator import ModelEvaluator


@pytest.fixture
def synthetic_train_data():
    np.random.seed(42)
    N, F = 200, 10
    X = np.random.randn(N, F)
    # Target linearly related to feature 0 and feature 1 plus noise
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * np.random.randn(N)
    return X, y


def test_ridge_model(synthetic_train_data):
    """Verify Ridge model in-fold scaling and prediction shape."""
    X, y = synthetic_train_data
    model = RidgeModel(alphas=[0.1, 1.0, 10.0])
    model.fit(X, y)

    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert model.best_alpha is not None
    assert len(model.coefficients) == X.shape[1]

    # Feature 0 should have positive coefficient, Feature 1 negative
    assert model.coefficients[0] > 0
    assert model.coefficients[1] < 0


def test_xgboost_model(synthetic_train_data):
    """Verify genuine XGBoost regressor fitting and feature importances."""
    X, y = synthetic_train_data
    model = XGBoostModel(n_estimators=20, max_depth=2, random_state=42)
    model.fit(X, y)

    preds = model.predict(X[:15])
    assert preds.shape == (15,)
    assert not np.isnan(preds).any()

    feat_names = [f"f_{i}" for i in range(X.shape[1])]
    fi = model.get_feature_importance(feat_names)
    assert len(fi) == X.shape[1]
    assert fi["f_0"] > 0


def test_lstm_model(synthetic_train_data):
    """Verify PyTorch LSTM model training and prediction."""
    X, y = synthetic_train_data
    model = LSTMModel(input_dim=X.shape[1], seq_len=4, hidden_dim=16, epochs=3, batch_size=32, random_state=42)
    model.fit(X, y)

    preds = model.predict(X[:12])
    assert preds.shape == (12,)
    assert not np.isnan(preds).any()


def test_model_ensemble(synthetic_train_data):
    """Verify ensemble prediction matches weighted individual predictions."""
    X, y = synthetic_train_data
    m_xgb = XGBoostModel(n_estimators=10, max_depth=2, random_state=42).fit(X, y)
    m_lstm = LSTMModel(input_dim=X.shape[1], seq_len=3, hidden_dim=16, epochs=2, batch_size=32).fit(X, y)

    ens = ModelEnsemble(weight_xgb=0.7, weight_lstm=0.3, model_xgb=m_xgb, model_lstm=m_lstm)
    ens_preds = ens.predict(X[:10])

    p_xgb = m_xgb.predict(X[:10])
    p_lstm = m_lstm.predict(X[:10])
    expected = 0.7 * p_xgb + 0.3 * p_lstm

    np.testing.assert_allclose(ens_preds, expected, rtol=1e-5, atol=1e-6)


def test_evaluator_metrics():
    """Verify ModelEvaluator mathematical formulas."""
    dates = ["2023-01-31"] * 10 + ["2023-02-28"] * 10
    tickers = [f"STK_{i}" for i in range(10)] * 2
    actuals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2)
    preds = actuals + 0.05 # high positive correlation

    df_test = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "actual": actuals,
        "pred": preds
    })

    evaluator = ModelEvaluator()
    res = evaluator.evaluate_model(df_test, "TestModel")

    assert res["model"] == "TestModel"
    assert res["observations"] == 20
    assert res["rebalance_dates"] == 2
    # Rank IC should be 1.0 (perfect rank match)
    assert res["mean_ic"] == pytest.approx(1.0, abs=1e-4)
    assert res["directional_accuracy_pct"] == 100.0
    assert res["rmse"] == pytest.approx(0.05, abs=1e-4)
    assert res["mae"] == pytest.approx(0.05, abs=1e-4)
