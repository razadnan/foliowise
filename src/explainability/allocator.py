"""
src/explainability/allocator.py
==============================
Allocation Explainability and Decision Attribution Engine.

Implements Master Implementation Plan Section 7.4:
- Decomposes optimal weights into quantifiable drivers:
  1. Predicted expected return / excess return rank
  2. Marginal Risk Contribution (MRC) and Percentage Risk Contribution (PRC)
  3. Diversification benefit: 1 - (MRC / standalone asset vol)
  4. Sector exposure and sector cap constraints
  5. Single-asset concentration cap constraints
- Synthesizes transparent rule-based natural language rationales using genuine quantitative outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class AllocationExplainer:
    """
    Computes stock-level portfolio attribution metrics and deterministic rationales.
    """

    def explain_portfolio(
        self,
        weights: Dict[str, float],
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        sector_mapping: Dict[str, str],
        stock_cap: float = 0.10,
        sector_cap: float = 0.35
    ) -> pd.DataFrame:
        """
        Generate detailed attribution metrics and transparent rationales for each stock.
        
        Returns:
        --------
        DataFrame sorted by weight descending with quantitative attribution columns
        and synthesized explanation.
        """
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
            if w >= stock_cap - 1e-4:
                rationale_parts.append(f"Capped at max single-stock limit ({stock_cap:.1%})")
            elif w > 0.05:
                rationale_parts.append(f"Strong positive allocation ({w:.1%})")

            if mu_val > 0.6:
                rationale_parts.append(f"high predicted forward alpha rank ({mu_val:.2f})")
            elif mu_val < 0.4:
                rationale_parts.append(f"lower return forecast ({mu_val:.2f}) offset by risk reduction")

            if d_ben > 0.25:
                rationale_parts.append(f"high diversification discount ({d_ben:.1%})")
            elif d_ben < 0.10:
                rationale_parts.append("closely co-moves with portfolio core")

            if sec_w >= sector_cap - 0.02:
                rationale_parts.append(f"bounded by {sec} sector cap ({sector_cap:.1%})")

            explanation_text = "; ".join(rationale_parts).capitalize() + "." if rationale_parts else "Balanced risk-adjusted weight."

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
                "is_at_stock_cap": bool(w >= stock_cap - 1e-4),
                "is_at_sector_cap": bool(sec_w >= sector_cap - 1e-4),
                "explanation": explanation_text
            })

        df_exp = pd.DataFrame(records).sort_values("weight", ascending=False).reset_index(drop=True)
        return df_exp
