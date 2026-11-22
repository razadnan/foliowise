"""
src/models/xgboost_model.py
===========================
XGBoost Primary Candidate Model.

Implements Model B from Master Implementation Plan Section 6.1:
- Uses genuine `xgboost.XGBRegressor` (not sklearn GradientBoosting)
- Tree depth regularization (max_depth=3) to control overfitting in low signal-to-noise equity returns
- Feature importance extraction
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from xgboost import XGBRegressor


class XGBoostModel:
    """
    Genuine XGBoost Regressor with controlled tree regularization.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": -1
        }
        self.model: Optional[XGBRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Fit XGBoost model."""
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns/ranks."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Return tree feature importances."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        return {f: float(imp) for f, imp in zip(feature_names, self.model.feature_importances_)}
