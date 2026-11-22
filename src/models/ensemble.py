"""
src/models/ensemble.py
======================
Validation-Tuned XGBoost-LSTM Ensemble Model.

Implements Model D from Master Implementation Plan Section 6.4:
- Combines nonlinear tree predictions (XGBoost) and sequential neural predictions (LSTM)
- Determines ensemble weights strictly from in-fold validation performance (e.g. inverse RMSE or Rank IC weighting)
- Provides 50/50 equal weighting as robust baseline
- Never uses locked test periods for weight selection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel


class ModelEnsemble:
    """
    Validation-tuned ensemble combining XGBoost and PyTorch LSTM.
    """

    def __init__(
        self,
        weight_xgb: float = 0.5,
        weight_lstm: float = 0.5,
        model_xgb: Optional[XGBoostModel] = None,
        model_lstm: Optional[LSTMModel] = None
    ):
        self.w_xgb = weight_xgb
        self.w_lstm = weight_lstm
        self.xgb = model_xgb
        self.lstm = model_lstm

    def fit_and_tune_weights(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_split_pct: float = 0.20
    ) -> Tuple[float, float]:
        """
        Fit XGBoost and LSTM on chronological train portion and tune ensemble
        weights on chronological validation slice.
        """
        N = len(X_train)
        split_idx = int(N * (1.0 - val_split_pct))

        X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]

        # Train models
        self.xgb = XGBoostModel().fit(X_tr, y_tr)
        self.lstm = LSTMModel(input_dim=X_tr.shape[1]).fit(X_tr, y_tr)

        # Validation predictions
        pred_xgb_val = self.xgb.predict(X_val)
        pred_lstm_val = self.lstm.predict(X_val)

        # Evaluate validation performance (Rank IC or inverse RMSE)
        ic_xgb, _ = stats.spearmanr(pred_xgb_val, y_val)
        ic_lstm, _ = stats.spearmanr(pred_lstm_val, y_val)

        ic_xgb = max(0.001, float(ic_xgb) if not np.isnan(ic_xgb) else 0.001)
        ic_lstm = max(0.001, float(ic_lstm) if not np.isnan(ic_lstm) else 0.001)

        # Inverse error / proportional positive IC weighting
        sum_ic = ic_xgb + ic_lstm
        self.w_xgb = float(ic_xgb / sum_ic)
        self.w_lstm = float(ic_lstm / sum_ic)

        # Re-fit both on full training window with tuned weights
        self.xgb.fit(X_train, y_train)
        self.lstm.fit(X_train, y_train)

        return self.w_xgb, self.w_lstm

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute weighted ensemble prediction: w_xgb * pred_xgb + w_lstm * pred_lstm.
        """
        if self.xgb is None or self.lstm is None:
            raise RuntimeError("Ensemble models must be initialized and fitted before predict.")

        p_xgb = self.xgb.predict(X)
        p_lstm = self.lstm.predict(X)
        return self.w_xgb * p_xgb + self.w_lstm * p_lstm
