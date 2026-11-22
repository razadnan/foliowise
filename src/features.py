"""
src/features.py
===============
Backward-compatible wrapper for FolioWise Feature Engineering.
Delegates to modular classes in `src.features` while preserving all
legacy helper functions and signatures.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# Ensure base dir in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features.pipeline import FeaturePipeline
from src.features.target import TargetGenerator
from src.features.leakage_test import LeakageTester

# Individual feature functions exposed for backward compatibility
compute_past_returns = FeaturePipeline.compute_past_returns
compute_rolling_volatility = FeaturePipeline.compute_rolling_volatility
compute_momentum = FeaturePipeline.compute_momentum
compute_rsi = FeaturePipeline.compute_rsi
compute_macd = FeaturePipeline.compute_macd
compute_moving_average_ratios = FeaturePipeline.compute_moving_average_ratios
compute_roc = FeaturePipeline.compute_roc
compute_volume_change = FeaturePipeline.compute_volume_change
compute_dist_52wh = FeaturePipeline.compute_dist_52wh
compute_rel_nifty = FeaturePipeline.compute_rel_nifty
compute_mkt_volatility = FeaturePipeline.compute_mkt_volatility
compute_mkt_correlation = FeaturePipeline.compute_mkt_correlation


def build_target(prices: pd.Series, horizon: int = 21) -> pd.Series:
    """Legacy forward return target builder."""
    gen = TargetGenerator(horizon=horizon)
    return gen.compute_forward_returns(prices, horizon)


def build_features_table(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    nifty_series: pd.Series,
    vix_series: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Build complete feature matrix. Accepts either NIFTY returns or NIFTY prices.
    """
    pipeline = FeaturePipeline()
    # Check if nifty_series is returns or prices
    if (nifty_series.abs() < 0.5).all():
        # Passed returns -> reconstruct normalized price series
        nifty_prices = (1.0 + nifty_series).cumprod() * 10000.0
    else:
        nifty_prices = nifty_series

    return pipeline.build_features_table(
        prices_df=prices_df,
        volume_df=volume_df,
        nifty_prices=nifty_prices,
        vix_series=vix_series
    )


def run_leakage_test(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    nifty_series: pd.Series,
    truncation_dates: List[str],
    vix_series: Optional[pd.Series] = None
) -> Dict:
    """
    Verify feature computation is free of look-ahead bias across truncation dates.
    """
    tester = LeakageTester()
    if (nifty_series.abs() < 0.5).all():
        nifty_prices = (1.0 + nifty_series).cumprod() * 10000.0
    else:
        nifty_prices = nifty_series

    res = tester.run_multi_date_leakage_test(
        prices_df=prices_df,
        volume_df=volume_df,
        nifty_prices=nifty_prices,
        vix_series=vix_series,
        truncation_dates=truncation_dates
    )
    return {
        "passed": res["passed"],
        "details": res["discrepancies"],
        "report_df": res["report_df"]
    }
