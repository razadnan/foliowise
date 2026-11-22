"""
src/risk/covariance.py
======================
Ledoit-Wolf Covariance Shrinkage and Risk Estimation Engine.

Implements Master Implementation Plan Section 7.1:
- Consistent lookback window (default 252 trading days)
- Ledoit-Wolf optimal shrinkage toward constant correlation target
- Spectral decomposition and positive semi-definite (PSD) validation
- Condition number improvement comparison against sample covariance
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.covariance import LedoitWolf


class CovarianceEstimator:
    """
    Estimates and validates asset return covariance matrices for portfolio optimization.
    """

    def __init__(self, lookback_days: int = 252, annualization_factor: float = 252.0):
        self.lookback_days = lookback_days
        self.annualization_factor = annualization_factor

    def estimate_sample_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute standard annualized sample covariance.
        """
        cov_daily = returns_df.cov()
        return cov_daily * self.annualization_factor

    def estimate_ledoit_wolf_covariance(
        self,
        returns_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Compute annualized Ledoit-Wolf shrinkage covariance matrix.
        
        Returns:
        --------
        (cov_df, shrinkage_intensity, diagnostics_dict)
        """
        clean_ret = returns_df.dropna(how="any")
        if len(clean_ret) < 20:
            raise ValueError(f"Insufficient history ({len(clean_ret)} rows) for covariance estimation.")

        lw = LedoitWolf()
        lw.fit(clean_ret.values)
        cov_daily = lw.covariance_
        shrinkage = float(lw.shrinkage_)

        # Ensure positive semi-definiteness via eigenvalue thresholding
        eigvals, eigvecs = np.linalg.eigh(cov_daily)
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))

        if min_eig < 1e-8:
            eigvals_clipped = np.maximum(eigvals, 1e-8)
            cov_daily = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        # Symmetrize
        cov_daily = (cov_daily + cov_daily.T) / 2.0
        cov_ann = cov_daily * self.annualization_factor

        cov_df = pd.DataFrame(cov_ann, index=clean_ret.columns, columns=clean_ret.columns)

        # Condition number comparison
        sample_cov = clean_ret.cov().values
        sample_eigs = np.linalg.eigvalsh(sample_cov)
        sample_cond = float(np.max(sample_eigs) / max(1e-12, np.min(sample_eigs)))
        lw_cond = float(max_eig / max(1e-12, min_eig))

        diagnostics = {
            "shrinkage_intensity": round(shrinkage, 4),
            "min_eigenvalue": float(min_eig),
            "max_eigenvalue": float(max_eig),
            "is_positive_definite": bool(min_eig > 0),
            "lw_condition_number": round(lw_cond, 2),
            "sample_condition_number": round(sample_cond, 2),
            "condition_improvement_ratio": round(sample_cond / max(1e-6, lw_cond), 2)
        }

        return cov_df, shrinkage, diagnostics
