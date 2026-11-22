"""
tests/test_adaptive.py
======================
Unit tests for Milestone M7: Adaptive Learning and Champion-Challenger Evaluator.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.feedback.engine import FeedbackEngine
from src.adaptive.evaluator import ChampionChallengerEvaluator


def test_feedback_engine(tmp_path):
    output_dir = str(tmp_path / "feedback")
    engine = FeedbackEngine(output_dir=output_dir)
    
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    tickers = ["T1", "T2", "T3"]
    
    # Mock prices
    price_records = []
    for d in dates:
        for t in tickers:
            price_records.append({
                "date": d,
                "ticker": t,
                "adj_close": 100.0 * (1.0 + 0.001 * (dates.get_loc(d)))
            })
    prices_df = pd.DataFrame(price_records)
    
    # Mock predictions on first date
    preds_df = pd.DataFrame([
        {"date": dates[0], "ticker": "T1", "model": "M1", "pred": 0.02},
        {"date": dates[0], "ticker": "T2", "model": "M1", "pred": 0.01},
        {"date": dates[0], "ticker": "T3", "model": "M1", "pred": -0.01},
    ])
    
    feedback_df = engine.generate_prediction_feedback(preds_df, prices_df, forward_days=10)
    assert not feedback_df.empty
    assert "prediction_error" in feedback_df.columns
    assert "direction_correct" in feedback_df.columns
    assert os.path.exists(os.path.join(output_dir, "prediction_feedback.csv"))


def test_champion_challenger_governance(tmp_path):
    output_dir = str(tmp_path / "adaptive")
    evaluator = ChampionChallengerEvaluator(
        min_observation_periods=3,
        promotion_hurdle=0.05,
        output_dir=output_dir
    )
    
    # Create 5 cycles of rolling IC
    records = []
    dates = ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"]
    for d in dates:
        # Ridge champion starts with 0.02
        records.append({"model": "Ridge", "signal_date": d, "rank_ic": 0.02, "rmse": 0.3})
        # Challenger XGBoost has superior 0.09 IC
        records.append({"model": "XGBoost", "signal_date": d, "rank_ic": 0.09, "rmse": 0.28})
        
    df_rolling = pd.DataFrame(records)
    df_metrics = pd.DataFrame()
    
    history_df, report = evaluator.evaluate_model_governance(df_rolling, df_metrics)
    
    assert len(history_df) == 5
    assert os.path.exists(os.path.join(output_dir, "champion_challenger_history.csv"))
    assert os.path.exists(os.path.join(output_dir, "model_governance_report.json"))
    
    # XGBoost should be promoted after 3 observation periods
    assert report["final_active_champion"] == "XGBoost"
    assert report["total_promotions"] >= 1
