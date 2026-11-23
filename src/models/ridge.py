"""
src/models/ridge.py
===================
Ridge Regression Baseline Model.

Implements Model A from Master Implementation Plan Section 6.1:
- Linear shrinkage baseline with L2 penalty
- In-fold StandardScaler fitted strictly inside training window
- Cross-validated hyperparameter alpha selection
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class RidgeModel:
    """
    Ridge Regression with in-fold standard scaling and chronological TimeSeriesSplit cross-validated alpha.
    """

    def __init__(self, alphas: Optional[List[float]] = None, cv: Union[int, Any] = 3):
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.cv = cv
        self.scaler = StandardScaler()
        self.model: Optional[RidgeCV] = None
        self.coefficients: Optional[np.ndarray] = None
        self.best_alpha: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        """
        Fit scaler and Ridge model on training data using chronological cross-validation.
        """
        X_scaled = self.scaler.fit_transform(X)
        cv_splitter = TimeSeriesSplit(n_splits=self.cv) if isinstance(self.cv, int) and self.cv > 1 else self.cv
        self.model = RidgeCV(alphas=self.alphas, cv=cv_splitter)
        self.model.fit(X_scaled, y)
        self.coefficients = self.model.coef_
        self.best_alpha = float(self.model.alpha_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict returns/ranks using fitted model.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Return absolute standardized coefficient magnitude.
        """
        if self.coefficients is None:
            return {}
        return {f: float(abs(c)) for f, c in zip(feature_names, self.coefficients)}
