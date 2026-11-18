"""
src/metrics.py
==============
Portfolio performance and model evaluation metrics.

Usage
-----
    from src.metrics import compute_strategy_metrics, compute_ic
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict


# ---------------------------------------------------------------------------
# Portfolio performance metrics
# ---------------------------------------------------------------------------

def compute_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.

    Parameters
    ----------
    returns : pd.Series
        Daily simple (arithmetic) returns.
    periods_per_year : int
        Trading days per year (default 252).

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.12 = 12 %).
    """
    n = len(returns)
    if n == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    years = n / periods_per_year
    return float(cumulative ** (1 / years) - 1)


def compute_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualised volatility (standard deviation of daily returns).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    periods_per_year : int
        Trading days per year (default 252).

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    if len(returns) < 2:
        return np.nan
    return float(returns.std() * np.sqrt(periods_per_year))


def compute_sharpe(
    returns: pd.Series,
    risk_free: float = 0.06,
    periods_per_year: int = 252
) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    risk_free : float
        Annual risk-free rate (default 6 % for India).
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    float
        Sharpe ratio.
    """
    daily_rf = risk_free / periods_per_year
    excess = returns - daily_rf
    vol = compute_volatility(returns, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return np.nan
    ann_excess = excess.mean() * periods_per_year
    return float(ann_excess / vol)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown (most negative peak-to-trough decline).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Max drawdown as a negative decimal (e.g. -0.35 = -35 %).
    """
    if len(returns) == 0:
        return np.nan
    cum = (1 + returns).cumprod()
    rolling_peak = cum.cummax()
    drawdown = (cum - rolling_peak) / rolling_peak
    return float(drawdown.min())


def compute_calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    periods_per_year : int
        Trading days per year.

    Returns
    -------
    float
        Calmar ratio. Returns NaN if max drawdown is 0.
    """
    cagr = compute_cagr(returns, periods_per_year)
    mdd = compute_max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr / abs(mdd))


# ---------------------------------------------------------------------------
# Model evaluation metric
# ---------------------------------------------------------------------------

def compute_ic(predictions_df: pd.DataFrame) -> Dict:
    """
    Compute the Information Coefficient (IC) of model predictions.

    IC = Spearman rank correlation between predicted and realized returns,
         computed cross-sectionally for each rebalance date, then averaged.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain columns: 'date', 'ticker', 'predicted', 'realized'.
        'predicted' : model's predicted 21-day forward return.
        'realized'  : actual 21-day forward return.

    Returns
    -------
    dict
        {
          'mean_ic': float,       # average IC across dates
          'ic_std': float,        # standard deviation of IC
          'ic_ir': float,         # IC / IC_std  (Information Ratio)
          'pct_positive': float,  # fraction of months with IC > 0
          'per_date': pd.Series,  # IC at each date
        }
    """
    required = {"date", "ticker", "predicted", "realized"}
    if not required.issubset(predictions_df.columns):
        raise ValueError(f"predictions_df must have columns: {required}")

    # Drop rows where either prediction or realization is NaN
    df = predictions_df.dropna(subset=["predicted", "realized"])

    per_date = {}
    for date, grp in df.groupby("date"):
        if len(grp) < 3:
            continue
        ic, _ = spearmanr(grp["predicted"], grp["realized"])
        per_date[date] = ic

    ic_series = pd.Series(per_date)
    if ic_series.empty:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "pct_positive": np.nan,
            "per_date": ic_series,
        }

    mean_ic = float(ic_series.mean())
    ic_std = float(ic_series.std())
    ic_ir = mean_ic / ic_std if ic_std > 0 else np.nan
    pct_pos = float((ic_series > 0).mean())

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "pct_positive": pct_pos,
        "per_date": ic_series,
    }


# ---------------------------------------------------------------------------
# Summary table for all strategies
# ---------------------------------------------------------------------------

def compute_strategy_metrics(
    daily_returns_df: pd.DataFrame,
    risk_free: float = 0.06
) -> pd.DataFrame:
    """
    Compute a summary performance table for all strategies.

    Parameters
    ----------
    daily_returns_df : pd.DataFrame
        Each column is a strategy identifier (e.g. 'S0', 'S1', ...).
        Each row is a trading day's simple return for that strategy.
    risk_free : float
        Annual risk-free rate (default 6 %).

    Returns
    -------
    pd.DataFrame
        Rows = strategies, columns = metric names:
        CAGR, Volatility, Sharpe, Max_Drawdown, Calmar
    """
    rows = {}
    for col in daily_returns_df.columns:
        ret = daily_returns_df[col].dropna()
        rows[col] = {
            "CAGR": compute_cagr(ret),
            "Volatility": compute_volatility(ret),
            "Sharpe": compute_sharpe(ret, risk_free=risk_free),
            "Max_Drawdown": compute_max_drawdown(ret),
            "Calmar": compute_calmar(ret),
        }

    metrics_df = pd.DataFrame(rows).T
    metrics_df.index.name = "Strategy"
    return metrics_df


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running metrics unit tests ...")
    errors = []

    # --- cagr ---
    flat = pd.Series([0.0] * 252)
    cagr_flat = compute_cagr(flat)
    assert abs(cagr_flat - 0.0) < 1e-9, f"CAGR of flat series should be 0, got {cagr_flat}"

    # 1 % daily over 252 days → CAGR ≈ (1.01^252)^(1/1) - 1
    daily_1pct = pd.Series([0.01] * 252)
    expected_cagr = 1.01**252 - 1
    computed_cagr = compute_cagr(daily_1pct)
    assert abs(computed_cagr - expected_cagr) < 1e-6, \
        f"CAGR mismatch: expected {expected_cagr:.4f}, got {computed_cagr:.4f}"

    # --- volatility ---
    vol_zero = compute_volatility(flat)
    assert vol_zero == 0.0 or np.isnan(vol_zero), "Volatility of flat returns should be 0"

    # --- sharpe ---
    # With daily return = risk_free/252, excess = 0, Sharpe should be 0
    rf_daily = 0.06 / 252
    rf_series = pd.Series([rf_daily] * 252)
    sharpe_zero = compute_sharpe(rf_series, risk_free=0.06)
    assert abs(sharpe_zero) < 1e-3, f"Sharpe near risk-free should be ~0, got {sharpe_zero}"

    # --- max drawdown ---
    # Series that always goes up → no drawdown
    up_series = pd.Series([0.01] * 100)
    mdd_up = compute_max_drawdown(up_series)
    assert mdd_up == 0.0 or mdd_up > -1e-9, f"Max drawdown of rising series should be 0, got {mdd_up}"

    # 50 % drop followed by recovery
    drop_then_recover = pd.Series([-0.01] * 70 + [0.01] * 70)
    mdd_drop = compute_max_drawdown(drop_then_recover)
    assert mdd_drop < 0, f"Max drawdown should be negative, got {mdd_drop}"

    # --- ic ---
    # Perfect positive rank correlation
    n = 20
    pred_df = pd.DataFrame({
        "date": ["2020-01-31"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "predicted": list(range(n)),
        "realized": list(range(n)),
    })
    ic_result = compute_ic(pred_df)
    assert abs(ic_result["mean_ic"] - 1.0) < 1e-6, \
        f"Perfect rank corr IC should be 1.0, got {ic_result['mean_ic']}"

    # --- strategy_metrics table ---
    ret_df = pd.DataFrame({
        "S0": pd.Series([0.001] * 252),
        "S1": pd.Series([0.0005] * 252),
    })
    metrics = compute_strategy_metrics(ret_df)
    assert set(metrics.columns) == {"CAGR", "Volatility", "Sharpe", "Max_Drawdown", "Calmar"}, \
        "Missing columns in strategy metrics"
    assert set(metrics.index) == {"S0", "S1"}, "Missing strategy rows"

    print(f"All {5 + 2} tests passed.")
    sys.exit(0)
