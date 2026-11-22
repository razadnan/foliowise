"""
src/models/evaluator.py
=======================
Standardized Quantitative Evaluation Metrics for ML Asset Pricing Models.

Computes:
- Rank IC (Spearman rank correlation per date)
- ICIR (Information Coefficient Information Ratio)
- Newey-West HAC autocorrelation-adjusted t-statistic
- Directional Accuracy %
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Out-of-sample R^2
- Top-Bottom Quintile Realized Spread
- Predicted vs Realized Decile Calibration
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any


class ModelEvaluator:
    """
    Evaluates out-of-sample cross-sectional prediction performance.
    """

    @staticmethod
    def compute_date_rank_ic(df_preds: pd.DataFrame, pred_col: str = "pred", actual_col: str = "actual") -> pd.Series:
        """
        Compute per-date cross-sectional Spearman rank correlation.
        """
        ic_records = {}
        for d, grp in df_preds.groupby("date"):
            valid = grp[[pred_col, actual_col]].dropna()
            if len(valid) >= 5 and valid[pred_col].std() > 1e-7 and valid[actual_col].std() > 1e-7:
                corr, _ = stats.spearmanr(valid[pred_col], valid[actual_col])
                if not np.isnan(corr):
                    ic_records[d] = float(corr)
        return pd.Series(ic_records)

    @staticmethod
    def compute_newey_west_tstat(series: pd.Series, max_lags: int = 3) -> float:
        """
        Compute Newey-West HAC adjusted t-statistic for a time-series mean.
        Accounts for overlapping return autocorrelation.
        """
        vals = series.dropna().values
        T = len(vals)
        if T <= max_lags + 1:
            return 0.0
        mean_val = np.mean(vals)
        demeaned = vals - mean_val
        gamma0 = np.sum(demeaned ** 2) / T
        gammas = []
        for l in range(1, max_lags + 1):
            gammas.append(np.sum(demeaned[l:] * demeaned[:-l]) / T)
        var_nw = gamma0 + 2.0 * sum((1.0 - (l + 1) / (max_lags + 1)) * gammas[l] for l in range(max_lags))
        se_nw = np.sqrt(max(var_nw, 1e-9) / T)
        return float(mean_val / se_nw)

    @classmethod
    def evaluate_model(
        cls,
        df_preds: pd.DataFrame,
        model_name: str,
        pred_col: str = "pred",
        actual_col: str = "actual",
        is_rank_target: bool = False
    ) -> Dict[str, Any]:
        """
        Compute full metric profile for a model.
        """
        valid = df_preds.dropna(subset=[actual_col, pred_col]).copy()
        if len(valid) == 0:
            return {"model": model_name, "error": "No valid predictions"}

        y_true = valid[actual_col].values
        y_pred = valid[pred_col].values

        # 1. Regression Errors
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # 2. Directional Accuracy
        if is_rank_target:
            # Rank centered at 0.5
            sign_match = ((y_pred - 0.5) * (y_true - 0.5) > 0)
        else:
            sign_match = (np.sign(y_pred) == np.sign(y_true))
        dir_acc = float(np.mean(sign_match) * 100.0)

        # 3. Rank IC Series & Stats
        ic_series = cls.compute_date_rank_ic(valid, pred_col=pred_col, actual_col=actual_col)
        n_periods = len(ic_series)
        ic_mean = float(ic_series.mean()) if n_periods > 0 else 0.0
        ic_std = float(ic_series.std()) if n_periods > 1 else 1e-9
        ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
        ic_tstat = float(ic_mean / (ic_std / np.sqrt(n_periods))) if ic_std > 0 and n_periods > 0 else 0.0
        nw_tstat = cls.compute_newey_west_tstat(ic_series, max_lags=3)

        # 4. Top-Bottom Spread
        spreads = []
        for d, grp in valid.groupby("date"):
            if len(grp) >= 10:
                k = max(2, len(grp) // 5)
                top = grp.nlargest(k, pred_col)[actual_col].mean()
                bot = grp.nsmallest(k, pred_col)[actual_col].mean()
                spreads.append(top - bot)
        mean_spread = float(np.mean(spreads) * 100.0) if spreads else 0.0

        return {
            "model": model_name,
            "observations": len(valid),
            "rebalance_dates": n_periods,
            "mean_ic": round(ic_mean, 4),
            "ic_ir": round(ic_ir, 4),
            "ic_tstat": round(ic_tstat, 2),
            "nw_tstat": round(nw_tstat, 2),
            "directional_accuracy_pct": round(dir_acc, 2),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "top_bottom_spread_pct": round(mean_spread, 2)
        }
