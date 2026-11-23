"""
tests/test_metrics_and_frontend.py
==================================
Unit tests for portfolio metrics, evaluation functions, frontend compilation,
and schema invariants across results/frontend.json and strategy artifacts.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.metrics import (
    compute_cagr,
    compute_volatility,
    compute_sharpe,
    compute_max_drawdown,
    compute_calmar,
    compute_ic,
    compute_strategy_metrics
)
from src.stage5_analysis import (
    compute_sortino_ratio,
    compute_max_drawdown_duration,
    compute_tracking_error_and_ir
)


def test_compute_cagr_properties():
    # 1. Flat returns -> CAGR = 0
    flat = pd.Series([0.0] * 252)
    assert compute_cagr(flat) == pytest.approx(0.0, abs=1e-7)

    # 2. Daily 1% over 252 days
    daily_1pct = pd.Series([0.01] * 252)
    expected_cagr = (1.01 ** 252) - 1.0
    assert compute_cagr(daily_1pct) == pytest.approx(expected_cagr, rel=1e-5)

    # 3. Empty series
    assert np.isnan(compute_cagr(pd.Series([], dtype=float)))


def test_compute_volatility_properties():
    # 1. Flat returns -> vol = 0
    flat = pd.Series([0.0] * 252)
    assert compute_volatility(flat) == pytest.approx(0.0, abs=1e-7)

    # 2. Known standard deviation
    np.random.seed(42)
    sample_rets = pd.Series(np.random.normal(0, 0.01, 252))
    expected_vol = sample_rets.std() * np.sqrt(252.0)
    assert compute_volatility(sample_rets) == pytest.approx(expected_vol, rel=1e-5)

    # 3. Single observation -> NaN
    assert np.isnan(compute_volatility(pd.Series([0.01])))


def test_compute_sharpe_properties():
    # 1. Constant return exactly equal to daily risk-free rate -> Sharpe ~ 0
    rf_daily = 0.06 / 252.0
    # Add tiny alternating jitter to give non-zero vol
    jitter = np.array([1e-5, -1e-5] * 126)
    rf_series = pd.Series(rf_daily + jitter)
    sharpe = compute_sharpe(rf_series, risk_free=0.06)
    assert abs(sharpe) < 1e-2

    # 2. Zero volatility -> returns NaN
    flat = pd.Series([0.05 / 252.0] * 252)
    assert np.isnan(compute_sharpe(flat, risk_free=0.06))


def test_compute_max_drawdown_and_duration():
    # 1. Monotonically rising -> zero drawdown and zero duration
    up = pd.Series([0.01] * 50)
    assert compute_max_drawdown(up) == pytest.approx(0.0, abs=1e-7)
    assert compute_max_drawdown_duration(up) == 0

    # 2. 50% drop and recovery
    drop_and_recover = pd.Series([-0.02] * 20 + [0.03] * 30)
    mdd = compute_max_drawdown(drop_and_recover)
    assert mdd < -0.10
    duration = compute_max_drawdown_duration(drop_and_recover)
    assert duration > 0


def test_compute_sortino_ratio():
    # Series with positive returns only -> near zero downside std -> finite high Sortino
    pos_series = pd.Series([0.002] * 252)
    sortino = compute_sortino_ratio(pos_series, rf=0.06)
    assert sortino > 0

    # Series with heavy downside
    bad_series = pd.Series([-0.005] * 252)
    bad_sortino = compute_sortino_ratio(bad_series, rf=0.06)
    assert bad_sortino < 0


def test_compute_tracking_error_and_ir():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    strat = pd.Series(0.001, index=dates)
    bench = pd.Series(0.0005, index=dates)

    te, ir = compute_tracking_error_and_ir(strat, bench)
    assert te >= 0.0
    # Strat outperforms benchmark -> IR >= 0
    assert ir >= 0.0


def test_compute_ic():
    # 1. Perfect correlation (+1.0)
    n = 30
    perfect_df = pd.DataFrame({
        "date": ["2024-01-31"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "predicted": list(range(n)),
        "realized": list(range(n))
    })
    ic_res = compute_ic(perfect_df)
    assert ic_res["mean_ic"] == pytest.approx(1.0, abs=1e-5)

    # 2. Inverse correlation (-1.0)
    inv_df = pd.DataFrame({
        "date": ["2024-01-31"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "predicted": list(range(n)),
        "realized": list(reversed(range(n)))
    })
    inv_ic = compute_ic(inv_df)
    assert inv_ic["mean_ic"] == pytest.approx(-1.0, abs=1e-5)


def test_frontend_json_schema_and_consistency():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fjson_path = os.path.join(base_dir, "results", "frontend.json")
    strat_csv_path = os.path.join(base_dir, "results", "strategy_metrics.csv")
    cost_csv_path = os.path.join(base_dir, "results", "cost_sensitivity.csv")

    assert os.path.exists(fjson_path), f"frontend.json missing at {fjson_path}"
    with open(fjson_path, "r") as f:
        data = json.load(f)

    # Top-level keys
    assert {"stocks", "nifty", "rho", "last_data_date", "test_summary"}.issubset(data.keys())
    assert len(data["stocks"]) > 0
    assert "m" in data["nifty"] and "v" in data["nifty"]
    assert isinstance(data["rho"], (int, float))

    # Cross-file validation with strategy_metrics.csv
    if os.path.exists(strat_csv_path):
        sm_df = pd.read_csv(strat_csv_path).set_index("strategy")
        if "S0" in sm_df.index:
            assert data["test_summary"]["benchmark_sharpe"] == pytest.approx(float(sm_df.loc["S0", "sharpe"]), abs=1e-2)
        if "S1" in sm_df.index:
            assert data["test_summary"]["equal_weight_sharpe"] == pytest.approx(float(sm_df.loc["S1", "sharpe"]), abs=1e-2)
        if "S4" in sm_df.index:
            assert data["test_summary"]["ml_strategy_sharpe"] == pytest.approx(float(sm_df.loc["S4", "sharpe"]), abs=1e-2)

    # Cost sensitivity consistency: higher cost must not yield higher net Sharpe
    cs = data["test_summary"]["cost_sensitivity"]
    sh_10 = cs["s4_sharpe_at_0.1pct"]
    sh_20 = cs["s4_sharpe_at_0.2pct"]
    sh_30 = cs["s4_sharpe_at_0.3pct"]
    assert sh_10 >= sh_20 - 1e-4, f"10 bps Sharpe {sh_10} lower than 20 bps Sharpe {sh_20}"
    assert sh_20 >= sh_30 - 1e-4, f"20 bps Sharpe {sh_20} lower than 30 bps Sharpe {sh_30}"


def test_build_frontend_script():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_html_path = os.path.join(base_dir, "frontend", "index.html")
    assert os.path.exists(frontend_html_path), "frontend/index.html does not exist"

    with open(frontend_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "<!DOCTYPE html>" in html_content
    assert "FolioWise" in html_content
    assert ("const S=" in html_content) or ("const S = " in html_content)
    assert ("NIFTY=" in html_content) or ("NIFTY = " in html_content)
    assert ("RHO=" in html_content) or ("RHO = " in html_content)
