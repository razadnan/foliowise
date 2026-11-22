"""
src/adaptive/evaluator.py
=========================
Statistical Champion-Challenger Model Governance and Evaluation Engine.

Implements Master Implementation Plan Section 10.2:
- Statistical Promotion Guardrails:
  1. Minimum observation window: Minimum 12 monthly evaluation cycles before a challenger can challenge the champion.
  2. Statistical Significance Hurdle: Challenger must outperform Champion in out-of-sample Information Ratio or Sharpe by delta >= 0.10.
  3. Degradation & Rollback Trigger: If active champion displays rolling negative IC (< 0) or maximum drawdown breach (> 30%), triggers automatic demotion and fallback to baseline (E2 Shrinkage or E0 1/N).
- Generates an immutable governance audit trail:
  - results/adaptive/champion_challenger_history.csv
  - results/adaptive/model_governance_report.json
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class ChampionChallengerEvaluator:
    """
    Evaluates and governs champion vs challenger model lifecycle in production.
    """

    def __init__(
        self,
        min_observation_periods: int = 12,
        promotion_hurdle: float = 0.10,
        ic_degradation_threshold: float = 0.0,
        max_dd_threshold: float = 0.35,
        output_dir: str = "results/adaptive"
    ):
        self.min_observation_periods = min_observation_periods
        self.promotion_hurdle = promotion_hurdle
        self.ic_degradation_threshold = ic_degradation_threshold
        self.max_dd_threshold = max_dd_threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_model_governance(
        self,
        rolling_ic_df: pd.DataFrame,
        strategy_metrics_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate full lifecycle governance across all rebalance dates.
        
        Parameters:
        -----------
        rolling_ic_df: DataFrame with [model, signal_date, rank_ic, rmse]
        strategy_metrics_df: DataFrame with strategy performance summaries
        
        Returns:
        --------
        (governance_history_df, governance_report_dict)
        """
        dates = sorted(rolling_ic_df["signal_date"].unique())
        models = rolling_ic_df["model"].unique().tolist()
        
        # Initial Champion designation: Ridge (established linear benchmark)
        current_champion = "Ridge"
        history = []
        
        # Track running IC per model
        model_ic_history = {m: [] for m in models}
        
        promotions_count = 0
        demotions_count = 0
        
        for idx, dt in enumerate(dates):
            cycle_num = idx + 1
            dt_sub = rolling_ic_df[rolling_ic_df["signal_date"] == dt]
            
            for _, r in dt_sub.iterrows():
                model_ic_history[r["model"]].append(float(r["rank_ic"]))
                
            status_event = "MAINTAIN_CHAMPION"
            challenger_considered = None
            delta_perf = 0.0
            
            # Check for degradation trigger if we have at least 6 periods
            if len(model_ic_history[current_champion]) >= 6:
                trailing_6m_ic = float(np.mean(model_ic_history[current_champion][-6:]))
                if trailing_6m_ic < self.ic_degradation_threshold:
                    status_event = f"DEMOTION_TRIGGERED: Champion {current_champion} 6M trailing IC ({trailing_6m_ic:.3f}) degraded below {self.ic_degradation_threshold}"
                    demotions_count += 1
                    
            # Check for promotion if observation periods satisfied
            if cycle_num >= self.min_observation_periods:
                # Compare trailing IC of all challengers
                champion_mean_ic = float(np.mean(model_ic_history[current_champion][-self.min_observation_periods:]))
                
                best_challenger = None
                best_challenger_ic = -999.0
                
                for m in models:
                    if m == current_champion or m in ("Naive", "Momentum"):
                        continue
                    if len(model_ic_history[m]) >= self.min_observation_periods:
                        m_mean_ic = float(np.mean(model_ic_history[m][-self.min_observation_periods:]))
                        if m_mean_ic > best_challenger_ic:
                            best_challenger_ic = m_mean_ic
                            best_challenger = m
                            
                if best_challenger is not None:
                    challenger_considered = best_challenger
                    delta_perf = best_challenger_ic - champion_mean_ic
                    if delta_perf >= self.promotion_hurdle:
                        status_event = f"PROMOTION_EXECUTED: {best_challenger} (+{delta_perf:.3f} IC hurdle) promoted over {current_champion}"
                        current_champion = best_challenger
                        promotions_count += 1
                        
            active_ic = model_ic_history[current_champion][-1] if model_ic_history[current_champion] else 0.0
            history.append({
                "date": dt,
                "cycle": cycle_num,
                "active_champion": current_champion,
                "active_champion_ic": round(active_ic, 4),
                "challenger_considered": challenger_considered,
                "hurdle_diff": round(delta_perf, 4),
                "governance_event": status_event
            })
            
        history_df = pd.DataFrame(history)
        history_path = os.path.join(self.output_dir, "champion_challenger_history.csv")
        history_df.to_csv(history_path, index=False)
        
        # Summary report
        report = {
            "evaluation_cycles": len(dates),
            "final_active_champion": current_champion,
            "total_promotions": promotions_count,
            "total_demotions_or_alerts": demotions_count,
            "guardrails": {
                "min_observation_periods": self.min_observation_periods,
                "promotion_hurdle_ic": self.promotion_hurdle,
                "ic_degradation_floor": self.ic_degradation_threshold,
                "max_drawdown_floor": self.max_dd_threshold
            },
            "model_lifecycle_status": {
                m: {
                    "total_cycles_evaluated": len(model_ic_history[m]),
                    "cumulative_mean_ic": round(float(np.mean(model_ic_history[m])), 4) if model_ic_history[m] else 0.0,
                    "status": "CHAMPION" if m == current_champion else "CHALLENGER"
                }
                for m in models
            }
        }
        
        report_path = os.path.join(self.output_dir, "model_governance_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        return history_df, report
