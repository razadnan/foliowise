"""
src/features/pipeline.py
========================
Standardized Feature Engineering Pipeline for NIFTY 50 Equities.

Implements all 14 quantitative features specified in Master Implementation Plan Section 5.1:
1.  f_ret_1: 1-day daily return
2.  f_ret_5: 5-day return
3.  f_ret_21: 21-day (monthly) return
4.  f_ret_63: 63-day (quarterly) return
5.  f_vol_21: 21-day rolling volatility
6.  f_vol_63: 63-day rolling volatility
7.  f_mom_126_21: Medium-term momentum (126-day lookback skipping recent 21 days)
8.  f_rsi_14: Wilder's 14-day Relative Strength Index
9.  f_macd: Moving Average Convergence Divergence normalized by price
10. f_macd_sig: MACD 9-day Signal line
11. f_macd_hist: MACD Histogram (f_macd - f_macd_sig)
12. f_sma_50_ratio: Price distance to 50-day Simple Moving Average (P/SMA50 - 1)
13. f_ema_20_ratio: Price distance to 20-day Exponential Moving Average (P/EMA20 - 1)
14. f_roc_10: 10-day Rate of Change
15. f_volchg: Relative volume surge (21-day avg / 63-day avg - 1)
16. f_dist_52wh: Distance from rolling 252-day (52-week) high
17. f_rel_nifty: 1-day return relative to NIFTY 50
18. f_mkt_vol_21: 21-day rolling volatility of NIFTY 50
19. f_mkt_corr_63: 63-day rolling correlation of stock return with NIFTY 50
20. f_vix_level: India VIX normalized level (VIX / 100.0)
21. f_vix_chg_21: 21-day percentage change in India VIX

ALL calculations are purely backward-looking: at row t, only information <= t is used.
"""

import os
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.features.target import TargetGenerator


class FeaturePipeline:
    """
    Constructs leak-free feature matrices and standardized prediction targets.
    """

    FEATURE_COLS = [
        "f_ret_1", "f_ret_5", "f_ret_21", "f_ret_63",
        "f_vol_21", "f_vol_63",
        "f_mom_126_21",
        "f_rsi_14",
        "f_macd", "f_macd_sig", "f_macd_hist",
        "f_sma_50_ratio", "f_ema_20_ratio",
        "f_roc_10",
        "f_volchg",
        "f_dist_52wh",
        "f_rel_nifty",
        "f_mkt_vol_21",
        "f_mkt_corr_63",
        "f_vix_level",
        "f_vix_chg_21"
    ]

    TARGET_COLS = [
        "target_20d_ret",
        "target_excess_20d",
        "target_rank_20d",
        "target_21d"
    ]

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.target_gen = TargetGenerator(horizon=20)

    # ---------------------------------------------------------------------------
    # Technical indicator calculators (purely backward-looking)
    # ---------------------------------------------------------------------------

    @staticmethod
    def compute_past_returns(prices: pd.Series, windows: List[int] = [1, 5, 21, 63]) -> pd.DataFrame:
        """Rolling backward-looking percentage returns."""
        result = {}
        for w in windows:
            result[f"f_ret_{w}"] = prices.pct_change(w)
        return pd.DataFrame(result, index=prices.index)

    @staticmethod
    def compute_rolling_volatility(daily_returns: pd.Series, windows: List[int] = [21, 63]) -> pd.DataFrame:
        """Rolling standard deviation of returns."""
        result = {}
        for w in windows:
            result[f"f_vol_{w}"] = daily_returns.rolling(window=w, min_periods=w).std()
        return pd.DataFrame(result, index=daily_returns.index)

    @staticmethod
    def compute_momentum(prices: pd.Series, long_window: int = 126, skip: int = 21) -> pd.Series:
        """Medium-term momentum skipping recent short-term reversal window."""
        mom = prices.shift(skip) / prices.shift(long_window) - 1.0
        mom.name = f"f_mom_{long_window}_{skip}"
        return mom

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Wilder's 14-day Relative Strength Index."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.name = f"f_rsi_{period}"
        return rsi

    @staticmethod
    def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD Line, Signal Line, and Histogram, normalized by price for cross-sectional comparability.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow) / prices
        macd_sig = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_sig
        return pd.DataFrame({
            "f_macd": macd_line,
            "f_macd_sig": macd_sig,
            "f_macd_hist": macd_hist
        }, index=prices.index)

    @staticmethod
    def compute_moving_average_ratios(prices: pd.Series) -> pd.DataFrame:
        """SMA 50 ratio (price / SMA50 - 1) and EMA 20 ratio (price / EMA20 - 1)."""
        sma_50 = prices.rolling(window=50, min_periods=50).mean()
        ema_20 = prices.ewm(span=20, adjust=False).mean()
        sma_ratio = prices / sma_50 - 1.0
        ema_ratio = prices / ema_20 - 1.0
        return pd.DataFrame({
            "f_sma_50_ratio": sma_ratio,
            "f_ema_20_ratio": ema_ratio
        }, index=prices.index)

    @staticmethod
    def compute_roc(prices: pd.Series, period: int = 10) -> pd.Series:
        """Rate of change over given period."""
        roc = prices / prices.shift(period) - 1.0
        roc.name = f"f_roc_{period}"
        return roc

    @staticmethod
    def compute_volume_change(volume: pd.Series, short_w: int = 21, long_w: int = 63) -> pd.Series:
        """Relative volume surge (21-day avg / 63-day avg - 1)."""
        short_avg = volume.rolling(window=short_w, min_periods=short_w).mean()
        long_avg = volume.rolling(window=long_w, min_periods=long_w).mean()
        volchg = short_avg / long_avg.replace(0, np.nan) - 1.0
        volchg.name = "f_volchg"
        return volchg

    @staticmethod
    def compute_dist_52wh(prices: pd.Series) -> pd.Series:
        """Distance from 252-day rolling high (always <= 0)."""
        rolling_max = prices.rolling(window=252, min_periods=252).max()
        dist = prices / rolling_max - 1.0
        dist.name = "f_dist_52wh"
        return dist

    @staticmethod
    def compute_rel_nifty(daily_returns: pd.Series, nifty_returns: pd.Series) -> pd.Series:
        """Daily return minus benchmark return."""
        rel = daily_returns - nifty_returns
        rel.name = "f_rel_nifty"
        return rel

    @staticmethod
    def compute_mkt_volatility(nifty_returns: pd.Series, window: int = 21) -> pd.Series:
        """Rolling volatility of benchmark returns."""
        vol = nifty_returns.rolling(window=window, min_periods=window).std()
        vol.name = f"f_mkt_vol_{window}"
        return vol

    @staticmethod
    def compute_mkt_correlation(daily_returns: pd.Series, nifty_returns: pd.Series, window: int = 63) -> pd.Series:
        """Rolling correlation between stock and market."""
        corr = daily_returns.rolling(window=window, min_periods=window).corr(nifty_returns).fillna(0.0)
        corr.name = f"f_mkt_corr_{window}"
        return corr

    # ---------------------------------------------------------------------------
    # Table builder
    # ---------------------------------------------------------------------------

    def build_features_table(
        self,
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        nifty_prices: pd.Series,
        vix_series: Optional[pd.Series] = None,
        min_history: int = 260
    ) -> pd.DataFrame:
        """
        Build the full feature and target matrix.
        
        Parameters:
        -----------
        prices_df: Wide DataFrame of adjusted close prices (columns = tickers).
        volume_df: Wide DataFrame of trading volumes.
        nifty_prices: Benchmark price index series.
        vix_series: Optional India VIX series.
        
        Returns:
        --------
        Long-format DataFrame sorted by (date, ticker).
        """
        tickers = prices_df.columns.tolist()
        calendar = prices_df.index
        nifty_ret = nifty_prices.pct_change()
        mkt_vol_21 = self.compute_mkt_volatility(nifty_ret, window=21)

        # Process VIX features
        if vix_series is not None:
            vix_aligned = vix_series.reindex(calendar).ffill()
            f_vix_level = vix_aligned / 100.0
            f_vix_chg_21 = vix_aligned / vix_aligned.shift(21) - 1.0
        else:
            f_vix_level = pd.Series(0.165, index=calendar)
            f_vix_chg_21 = pd.Series(0.0, index=calendar)

        # Build standardized targets
        targets_table = self.target_gen.build_targets_table(prices_df, nifty_prices)

        all_records = []
        for ticker in tickers:
            p = prices_df[ticker].dropna()
            if len(p) < min_history:
                continue

            v = volume_df[ticker].reindex(p.index).ffill()
            n_ret = nifty_ret.reindex(p.index)
            d_ret = p.pct_change()

            past_ret = self.compute_past_returns(p, windows=[1, 5, 21, 63])
            rolling_vol = self.compute_rolling_volatility(d_ret, windows=[21, 63])
            mom = self.compute_momentum(p, long_window=126, skip=21)
            rsi = self.compute_rsi(p, period=14)
            macd_df = self.compute_macd(p, fast=12, slow=26, signal=9)
            ma_ratios = self.compute_moving_average_ratios(p)
            roc = self.compute_roc(p, period=10)
            volchg = self.compute_volume_change(v, short_w=21, long_w=63)
            dist_52wh = self.compute_dist_52wh(p)
            rel_nifty = self.compute_rel_nifty(d_ret, n_ret)
            mkt_corr = self.compute_mkt_correlation(d_ret, n_ret, window=63)
            m_vol = mkt_vol_21.reindex(p.index)
            v_lvl = f_vix_level.reindex(p.index)
            v_chg = f_vix_chg_21.reindex(p.index)

            ticker_feats = pd.concat([
                past_ret, rolling_vol, mom, rsi,
                macd_df, ma_ratios, roc, volchg,
                dist_52wh, rel_nifty, m_vol, mkt_corr,
                pd.Series(v_lvl, name="f_vix_level"),
                pd.Series(v_chg, name="f_vix_chg_21")
            ], axis=1)

            ticker_feats.index.name = "date"
            ticker_feats = ticker_feats.reset_index()
            ticker_feats.insert(1, "ticker", ticker)

            # Drop warm-up rows where any FEATURE is NaN
            feat_mask = ticker_feats[self.FEATURE_COLS].notna().all(axis=1)
            ticker_feats = ticker_feats[feat_mask].copy()

            all_records.append(ticker_feats)

        if not all_records:
            raise ValueError("No tickers had sufficient observations to build features.")

        full_features = pd.concat(all_records, ignore_index=True)

        # Merge targets
        # Standardize date format for merge
        full_features["date"] = pd.to_datetime(full_features["date"]).dt.strftime("%Y-%m-%d")
        targets_table["date"] = pd.to_datetime(targets_table["date"]).dt.strftime("%Y-%m-%d")

        merged = pd.merge(full_features, targets_table, on=["date", "ticker"], how="left")
        merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
        return merged
