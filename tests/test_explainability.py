"""
tests/test_explainability.py
============================
Unit tests for Milestone M8: Allocation Explainability and Decision Attribution Engine.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.explainability.allocator import AllocationExplainer


def test_explain_portfolio_with_factors():
    explainer = AllocationExplainer(stock_cap=0.10, sector_cap=0.35)
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    weights = {"RELIANCE.NS": 0.10, "TCS.NS": 0.05, "HDFCBANK.NS": 0.08}
    mu = pd.Series([0.65, 0.45, 0.55], index=tickers)
    
    # Simple cov matrix
    cov = pd.DataFrame([
        [0.04, 0.01, 0.01],
        [0.01, 0.03, 0.01],
        [0.01, 0.01, 0.05]
    ], index=tickers, columns=tickers)
    
    sectors = {"RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking"}
    stock_feats = {
        "RELIANCE.NS": {"f_macd_sig": 0.012, "f_vol_21": 0.15, "f_dist_52wh": -0.02}
    }
    
    df_exp = explainer.explain_portfolio(
        weights=weights,
        expected_returns=mu,
        cov_matrix=cov,
        sector_mapping=sectors,
        stock_features=stock_feats
    )
    
    assert len(df_exp) == 3
    assert "marginal_risk_contrib" in df_exp.columns
    assert "pct_risk_contrib" in df_exp.columns
    assert "diversification_benefit" in df_exp.columns
    assert "top_factor_driver" in df_exp.columns
    assert "explanation" in df_exp.columns
    
    # Check that RELIANCE is flagged as at stock cap
    rel_row = df_exp[df_exp["ticker"] == "RELIANCE.NS"].iloc[0]
    assert bool(rel_row["is_at_stock_cap"]) is True
    assert "Capped at max single-stock limit" in rel_row["explanation"]
    assert "positive MACD signal momentum" in rel_row["explanation"]


def test_generate_and_save_allocation_explanations(tmp_path):
    explainer = AllocationExplainer()
    csv_out = str(tmp_path / "allocation_explanations.csv")
    json_out = str(tmp_path / "allocation_explanations.json")
    
    weights_df = pd.DataFrame([
        {"date": "2024-01-31", "experiment": "E3", "ticker": "T1", "weight": 0.10},
        {"date": "2024-01-31", "experiment": "E3", "ticker": "T2", "weight": 0.10}
    ])
    
    dates = pd.date_range("2023-01-01", periods=260, freq="B")
    returns_records = []
    for d in dates:
        for t in ["T1", "T2"]:
            returns_records.append({"date": d, "ticker": t, "ret_1d": 0.0005})
    returns_df = pd.DataFrame(returns_records)
    
    df_exp = explainer.generate_and_save_allocation_explanations(
        weights_df=weights_df,
        returns_df=returns_df,
        sector_mapping={"T1": "Tech", "T2": "Fin"},
        strategy_name="E3",
        date_filter="2024-01-31",
        output_csv_path=csv_out,
        output_json_path=json_out
    )
    
    assert os.path.exists(csv_out)
    assert os.path.exists(json_out)
    
    with open(json_out, "r") as f:
        data = json.load(f)
    assert data["strategy"] == "E3"
    assert data["allocated_assets"] == 2
