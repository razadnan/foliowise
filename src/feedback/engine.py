"""
src/feedback/engine.py
======================
Adaptive Prediction and Portfolio Feedback Engine.

Implements Master Implementation Plan Section 10.1:
1. Prediction Feedback Logger:
   - Matches ex-ante model predictions against subsequent realized returns.
   - Computes prediction error, directional accuracy, and cross-sectional Rank IC per period.
   - Evaluates rolling decay or improvement in factor efficacy.
2. Portfolio Feedback Logger:
   - Contrasts ex-ante expected portfolio returns and risk against ex-post realized results.
   - Tracks volatility surprises (ex-post realized / ex-ante forecasted).
   - Exports immutable feedback logs to results/feedback/.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple, Any


class FeedbackEngine:
    """
    Automates feedback loop tracking for financial machine learning and portfolio optimization.
    """

    def __init__(self, output_dir: str = "results/feedback"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_prediction_feedback(
        self,
        predictions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        forward_days: int = 20
    ) -> pd.DataFrame:
        """
        Match model predictions with realized forward returns.
        
        Parameters:
        -----------
        predictions_df: DataFrame with [date, ticker, model, pred]
        prices_df: DataFrame with [date, ticker, adj_close]
        forward_days: Holding horizon (default 20 business days)
        
        Returns:
        --------
        DataFrame with prediction error, directional accuracy, and realized values.
        """
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        predictions_df = predictions_df.copy()
        predictions_df["date"] = pd.to_datetime(predictions_df["date"])
        
        piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        all_dates = piv_prices.index
        
        records = []
        unique_dates = predictions_df["date"].unique()
        
        for dt_val in unique_dates:
            dt = pd.to_datetime(dt_val)
            if dt not in all_dates:
                continue
            
            # Find date at dt + forward_days
            curr_loc = all_dates.get_loc(dt)
            target_loc = curr_loc + forward_days
            if target_loc >= len(all_dates):
                continue
            
            future_dt = all_dates[target_loc]
            curr_px = piv_prices.loc[dt]
            future_px = piv_prices.loc[future_dt]
            realized_ret = (future_px / curr_px) - 1.0
            
            sub_preds = predictions_df[predictions_df["date"] == dt_val]
            for _, row in sub_preds.iterrows():
                tk = row["ticker"]
                model_name = row["model"]
                pred_val = float(row["pred"])
                
                if tk in realized_ret.index and not np.isnan(realized_ret[tk]):
                    actual_ret = float(realized_ret[tk])
                    err = actual_ret - pred_val
                    dir_correct = (actual_ret >= 0 and pred_val >= 0) or (actual_ret < 0 and pred_val < 0)
                    
                    records.append({
                        "signal_date": dt.strftime("%Y-%m-%d"),
                        "evaluation_date": future_dt.strftime("%Y-%m-%d"),
                        "model": model_name,
                        "ticker": tk,
                        "predicted_return": round(pred_val, 6),
                        "realized_return": round(actual_ret, 6),
                        "prediction_error": round(err, 6),
                        "abs_error": round(abs(err), 6),
                        "direction_correct": dir_correct
                    })
                    
        df_feedback = pd.DataFrame(records)
        feedback_path = os.path.join(self.output_dir, "prediction_feedback.csv")
        df_feedback.to_csv(feedback_path, index=False)
        return df_feedback

    def compute_rolling_ic_summary(self, feedback_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute period-by-period cross-sectional Rank IC for each model.
        """
        summary_records = []
        grouped = feedback_df.groupby(["model", "signal_date"])
        
        for (model_name, sig_date), group in grouped:
            if len(group) >= 5:
                ic, _ = spearmanr(group["predicted_return"], group["realized_return"])
                if np.isnan(ic):
                    ic = 0.0
                dir_acc = group["direction_correct"].mean() * 100.0
                rmse = np.sqrt(np.mean(group["prediction_error"] ** 2))
                
                summary_records.append({
                    "model": model_name,
                    "signal_date": sig_date,
                    "n_assets": len(group),
                    "rank_ic": round(float(ic), 4),
                    "directional_accuracy_pct": round(float(dir_acc), 2),
                    "rmse": round(float(rmse), 4)
                })
                
        df_summary = pd.DataFrame(summary_records)
        summary_path = os.path.join(self.output_dir, "rolling_ic_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        return df_summary

    def generate_portfolio_feedback(
        self,
        daily_returns_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate ex-ante expected risk against ex-post realized portfolio risk.
        """
        records = []
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        all_dates = piv_prices.index
        
        for strat in weights_df["strategy" if "strategy" in weights_df.columns else "experiment"].unique():
            strat_weights = weights_df[weights_df["strategy" if "strategy" in weights_df.columns else "experiment"] == strat]
            strat_daily = daily_returns_df[daily_returns_df["strategy"] == strat].copy()
            strat_daily["date"] = pd.to_datetime(strat_daily["date"])
            strat_daily = strat_daily.set_index("date")
            
            rebal_dates = sorted(strat_weights["date"].unique())
            for i in range(len(rebal_dates) - 1):
                d_start = pd.to_datetime(rebal_dates[i])
                d_end = pd.to_datetime(rebal_dates[i+1])
                
                # Realized daily returns in this holding window
                sub_rets = strat_daily.loc[d_start:d_end, "net_return"] if "net_return" in strat_daily.columns else strat_daily.loc[d_start:d_end, "net_ret"]
                if len(sub_rets) < 5:
                    continue
                    
                realized_ann_vol = float(sub_rets.std() * np.sqrt(252))
                realized_cum_ret = float((1.0 + sub_rets).prod() - 1.0)
                
                # Trailing historical asset returns for ex-ante risk estimate
                hist_window = piv_prices.loc[:d_start].iloc[-126:].pct_change().dropna()
                w_sub = strat_weights[strat_weights["date"] == rebal_dates[i]]
                w_vec = pd.Series(dict(zip(w_sub["ticker"], w_sub["weight"]))).reindex(hist_window.columns).fillna(0.0)
                
                ex_ante_vol = float(np.sqrt(w_vec.values @ (hist_window.cov().values * 252) @ w_vec.values))
                vol_surprise = realized_ann_vol / max(ex_ante_vol, 1e-4)
                
                records.append({
                    "strategy": strat,
                    "cycle_start": rebal_dates[i],
                    "cycle_end": rebal_dates[i+1],
                    "ex_ante_volatility": round(ex_ante_vol * 100.0, 2),
                    "realized_volatility": round(realized_ann_vol * 100.0, 2),
                    "volatility_surprise_ratio": round(vol_surprise, 3),
                    "realized_cycle_return_pct": round(realized_cum_ret * 100.0, 2)
                })
                
        df_port = pd.DataFrame(records)
        port_path = os.path.join(self.output_dir, "portfolio_feedback.csv")
        df_port.to_csv(port_path, index=False)
        return df_port
