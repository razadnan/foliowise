"""
src/explainability/allocator.py
==============================
Allocation Explainability and Decision Attribution Engine.

Implements Master Implementation Plan Section 7.4 & Milestone M8:
- Decomposes optimal weights into quantifiable drivers:
  1. Predicted expected return / excess return alpha rank
  2. Marginal Risk Contribution (MRC) and Percentage Risk Contribution (PRC)
  3. Diversification benefit: 1 - (MRC / standalone asset vol)
  4. Sector exposure and sector cap constraints
  5. Single-asset concentration cap constraints
  6. Top predictive feature/SHAP drivers (MACD, volatility, distance to 52-week high, market correlation)
- Synthesizes transparent rule-based natural language rationales using genuine quantitative outputs.
- Exports comprehensive explanations in both JSON and CSV formats.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class AllocationExplainer:
    """
    Computes stock-level portfolio attribution metrics, factor drivers, and deterministic rationales.
    """

    def __init__(self, stock_cap: float = 0.10, sector_cap: float = 0.35):
        self.stock_cap = stock_cap
        self.sector_cap = sector_cap

    def explain_portfolio(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        sector_mapping: Dict[str, str],
        stock_features: Optional[Dict[str, Dict[str, float]]] = None,
        top_shap_features: Optional[List[str]] = None,
        stock_cap: Optional[float] = None,
        sector_cap: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate detailed attribution metrics and transparent rationales for each stock.
        """
        cap_stock = stock_cap if stock_cap is not None else self.stock_cap
        cap_sector = sector_cap if sector_cap is not None else self.sector_cap

        tickers = [t for t in weights.keys() if t in cov_matrix.columns]
        w_arr = np.array([weights[t] for t in tickers])
        cov = cov_matrix.loc[tickers, tickers].values

        # Portfolio volatility
        port_vol = float(np.sqrt(np.maximum(w_arr @ cov @ w_arr, 1e-8)))

        # Marginal risk contribution: (Sigma w) / port_vol
        mrc = (cov @ w_arr) / port_vol
        # Percentage risk contribution: w * MRC / port_vol (sums to 100%)
        prc = (w_arr * mrc) / port_vol

        # Standalone volatility of each asset
        standalone_vol = np.sqrt(np.maximum(np.diag(cov), 1e-8))
        # Diversification benefit: 1 - MRC / standalone_vol
        div_benefit = 1.0 - (mrc / standalone_vol)

        # Sector weights
        sector_totals = {}
        for t, w in zip(tickers, w_arr):
            sec = sector_mapping.get(t, "Other")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w

        records = []
        for i, t in enumerate(tickers):
            w = float(w_arr[i])
            sec = sector_mapping.get(t, "Other")
            sec_w = float(sector_totals.get(sec, 0.0))
            mu_val = float(expected_returns.get(t, 0.0))
            s_vol = float(standalone_vol[i])
            m_rc = float(mrc[i])
            p_rc = float(prc[i])
            d_ben = float(div_benefit[i])

            # Deterministic rationale synthesis
            rationale_parts = []
            if w >= cap_stock - 1e-4:
                rationale_parts.append(f"Capped at max single-stock limit ({cap_stock:.1%})")
            elif w > 0.05:
                rationale_parts.append(f"Strong positive allocation ({w:.1%})")

            if mu_val > 0.6:
                rationale_parts.append(f"high predicted alpha rank ({mu_val:.2f})")
            elif mu_val < 0.4:
                rationale_parts.append(f"lower return forecast ({mu_val:.2f}) offset by risk reduction")

            # Check feature driver if available
            top_driver_str = ""
            if stock_features and t in stock_features:
                feats = stock_features[t]
                # Identify notable signals
                drivers = []
                if feats.get("f_macd_sig", 0.0) > 0.005:
                    drivers.append("positive MACD signal momentum")
                if feats.get("f_dist_52wh", -1.0) > -0.05:
                    drivers.append("near 52-week high breakout zone")
                if feats.get("f_vol_21", 0.5) < 0.18:
                    drivers.append("favorable low realized volatility")
                if drivers:
                    top_driver_str = ", ".join(drivers)
                    rationale_parts.append(f"supported by {top_driver_str}")

            if d_ben > 0.25:
                rationale_parts.append(f"high diversification discount ({d_ben:.1%})")
            elif d_ben < 0.10:
                rationale_parts.append("closely co-moves with portfolio core")

            if sec_w >= cap_sector - 0.02:
                rationale_parts.append(f"bounded by {sec} sector cap ({cap_sector:.1%})")

            if rationale_parts:
                rationale_parts[0] = rationale_parts[0][0].upper() + rationale_parts[0][1:] if len(rationale_parts[0]) > 0 else ""
                explanation_text = "; ".join(rationale_parts) + "."
            else:
                explanation_text = "Balanced risk-adjusted weight."

            records.append({
                "ticker": t,
                "weight": round(w, 4),
                "sector": sec,
                "expected_return_or_rank": round(mu_val, 4),
                "standalone_vol": round(s_vol, 4),
                "marginal_risk_contrib": round(m_rc, 4),
                "pct_risk_contrib": round(p_rc, 4),
                "diversification_benefit": round(d_ben, 4),
                "sector_total_weight": round(sec_w, 4),
                "top_factor_driver": top_driver_str if top_driver_str else "Multi-factor balance",
                "is_at_stock_cap": bool(w >= cap_stock - 1e-4),
                "is_at_sector_cap": bool(sec_w >= cap_sector - 1e-4),
                "explanation": explanation_text
            })

        df_exp = pd.DataFrame(records).sort_values("weight", ascending=False).reset_index(drop=True)
        return df_exp

    def generate_and_save_allocation_explanations(
        self,
        weights_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        sector_mapping: Dict[str, str],
        predictions_df: Optional[pd.DataFrame] = None,
        features_df: Optional[pd.DataFrame] = None,
        strategy_name: str = "E3",
        date_filter: Optional[str] = None,
        output_csv_path: str = "results/allocation_explanations.csv",
        output_json_path: str = "results/allocation_explanations.json"
    ) -> pd.DataFrame:
        """
        Generate explanations for the latest (or specified) rebalance cycle and save to files.
        """
        # Filter weights for strategy
        strat_col = "experiment" if "experiment" in weights_df.columns else "strategy"
        strat_weights = weights_df[weights_df[strat_col] == strategy_name]
        if strat_weights.empty:
            strat_weights = weights_df

        available_dates = sorted(strat_weights["date"].unique())
        target_date = date_filter if date_filter else available_dates[-1]

        sub_w = strat_weights[strat_weights["date"] == target_date]
        w_dict = dict(zip(sub_w["ticker"], sub_w["weight"]))

        # Covariance from trailing returns
        dt = pd.to_datetime(target_date)
        returns_df = returns_df.copy()
        returns_df["date"] = pd.to_datetime(returns_df["date"])
        piv_ret = returns_df.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
        hist_ret = piv_ret.loc[:dt].iloc[-252:]
        cov_matrix = hist_ret.cov() * 252.0

        # Expected returns / alpha rank
        exp_ret = hist_ret.mean() * 252.0
        if predictions_df is not None:
            pred_sub = predictions_df[predictions_df["date"] == target_date]
            if not pred_sub.empty:
                exp_ret = pd.Series(dict(zip(pred_sub["ticker"], pred_sub["pred"]))).reindex(cov_matrix.columns).fillna(0.5)

        # Stock features
        stock_feats = {}
        if features_df is not None:
            feat_sub = features_df[features_df["date"] == target_date]
            for _, r in feat_sub.iterrows():
                stock_feats[r["ticker"]] = r.to_dict()

        df_exp = self.explain_portfolio(
            weights=w_dict,
            expected_returns=exp_ret,
            cov_matrix=cov_matrix,
            sector_mapping=sector_mapping,
            stock_features=stock_feats
        )

        df_exp.insert(0, "strategy", strategy_name)
        df_exp.insert(1, "rebalance_date", target_date)

        # Save CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_exp.to_csv(output_csv_path, index=False)

        # Save JSON
        json_payload = {
            "strategy": strategy_name,
            "rebalance_date": target_date,
            "total_assets": len(df_exp),
            "allocated_assets": int((df_exp["weight"] > 1e-4).sum()),
            "allocations": df_exp.to_dict(orient="records")
        }
        with open(output_json_path, "w") as f:
            json.dump(json_payload, f, indent=2)

        return df_exp
