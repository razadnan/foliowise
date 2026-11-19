"""
src/features.py
===============
Leak-free feature engineering for the NIFTY 50 ML portfolio project.

ALL computations are purely backward-looking: at time t, only data
up to and including t is used. No future information leaks in.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------

def compute_past_returns(
    prices: pd.Series,
    windows: List[int] = [1, 5, 21, 63]
) -> pd.DataFrame:
    """
    Compute past (backward-looking) returns over given rolling windows.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices, indexed by date.
    windows : list of int
        Look-back windows in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: f_ret_{w} for each window w.
        NaN for the first w rows of each column (warm-up period).
    """
    result = {}
    for w in windows:
        # prices.pct_change(w) gives (prices[t] - prices[t-w]) / prices[t-w]
        # This is fully backward-looking.
        result[f"f_ret_{w}"] = prices.pct_change(w)
    return pd.DataFrame(result, index=prices.index)


def compute_rolling_volatility(
    returns: pd.Series,
    windows: List[int] = [21, 63]
) -> pd.DataFrame:
    """
    Compute rolling standard deviation of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (pct_change of prices).
    windows : list of int
        Rolling windows in trading days.

    Returns
    -------
    pd.DataFrame
        Columns: f_vol_{w} for each window w.
    """
    result = {}
    for w in windows:
        # min_periods=w ensures NaN until we have a full window.
        result[f"f_vol_{w}"] = returns.rolling(window=w, min_periods=w).std()
    return pd.DataFrame(result, index=returns.index)


def compute_momentum(
    prices: pd.Series,
    long_window: int = 126,
    skip: int = 21
) -> pd.Series:
    """
    Compute medium-term momentum skipping the most recent period.

    Formula: prices[t - skip] / prices[t - long_window] - 1

    This avoids short-term reversal contamination and is fully
    backward-looking.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    long_window : int
        Total look-back horizon (default 126 ~ 6 months).
    skip : int
        Recent days to skip (default 21 ~ 1 month).

    Returns
    -------
    pd.Series
        Named 'f_mom_{long_window}_{skip}'.
    """
    # prices.shift(skip) gives prices[t - skip] at row t
    # prices.shift(long_window) gives prices[t - long_window] at row t
    mom = prices.shift(skip) / prices.shift(long_window) - 1
    mom.name = f"f_mom_{long_window}_{skip}"
    return mom


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS)
    RS  = avg_gain / avg_loss  (Wilder's smoothed average)

    Fully backward-looking. NaN for first `period` rows.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    period : int
        Look-back period (default 14 days).

    Returns
    -------
    pd.Series
        Named 'f_rsi_{period}', values in [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothed average: initialise with simple mean, then ewm
    # We use adjust=False and com=(period-1) which matches Wilder's formula.
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi.name = f"f_rsi_{period}"
    return rsi


def compute_volume_change(
    volume: pd.Series,
    short_w: int = 21,
    long_w: int = 63
) -> pd.Series:
    """
    Compute relative volume: short-window avg / long-window avg - 1.

    Positive values indicate recent volume surge; negative = drought.
    Fully backward-looking.

    Parameters
    ----------
    volume : pd.Series
        Daily trading volume.
    short_w : int
        Short rolling window (default 21).
    long_w : int
        Long rolling window (default 63).

    Returns
    -------
    pd.Series
        Named 'f_volchg'.
    """
    short_avg = volume.rolling(window=short_w, min_periods=short_w).mean()
    long_avg = volume.rolling(window=long_w, min_periods=long_w).mean()
    volchg = short_avg / long_avg.replace(0, np.nan) - 1
    volchg.name = "f_volchg"
    return volchg


def compute_dist_52wh(prices: pd.Series) -> pd.Series:
    """
    Distance from the 52-week (252 trading-day) rolling high.

    Formula: price[t] / max(price[t-252 : t]) - 1

    Result is always <= 0 (stock is at or below its 252-day high).
    Fully backward-looking; uses a *closed* window ending at t.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.

    Returns
    -------
    pd.Series
        Named 'f_dist_52wh', values <= 0.
    """
    # rolling(252).max() computes max over [t-251, t], i.e. including t itself.
    rolling_max = prices.rolling(window=252, min_periods=252).max()
    dist = prices / rolling_max - 1
    dist.name = "f_dist_52wh"
    return dist


def compute_rel_nifty(
    stock_returns: pd.Series,
    nifty_returns: pd.Series
) -> pd.Series:
    """
    Compute stock excess return relative to NIFTY 50.

    Formula: stock_return[t] - nifty_return[t]

    Both must be 1-day pct_change series aligned on the same index.

    Parameters
    ----------
    stock_returns : pd.Series
        1-day percentage returns for the stock.
    nifty_returns : pd.Series
        1-day percentage returns for NIFTY 50.

    Returns
    -------
    pd.Series
        Named 'f_rel_nifty'.
    """
    rel = stock_returns - nifty_returns
    rel.name = "f_rel_nifty"
    return rel


def compute_mkt_volatility(
    nifty_returns: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Compute backward-looking rolling market volatility of NIFTY 50 returns.
    """
    vol = nifty_returns.rolling(window=window, min_periods=window).std()
    vol.name = f"f_mkt_vol_{window}"
    return vol


def compute_mkt_correlation(
    stock_returns: pd.Series,
    nifty_returns: pd.Series,
    window: int = 63
) -> pd.Series:
    """
    Compute backward-looking rolling correlation between stock and market.
    """
    corr = stock_returns.rolling(window=window, min_periods=window).corr(nifty_returns).fillna(0.0)
    corr.name = f"f_mkt_corr_{window}"
    return corr


def build_target(prices: pd.Series, horizon: int = 21) -> pd.Series:
    """
    Compute the forward return target for supervised learning.

    Formula: prices[t + horizon] / prices[t] - 1

    **Important**: The last `horizon` rows will be NaN because there is
    no future price available. These rows must be excluded from model
    training (but features can still be computed for prediction).

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices.
    horizon : int
        Forward look-ahead in trading days (default 21 ~ 1 month).

    Returns
    -------
    pd.Series
        Named 'target_{horizon}d'.
    """
    # shift(-horizon) brings future price to current row
    target = prices.shift(-horizon) / prices - 1
    target.name = f"target_{horizon}d"
    return target


# ---------------------------------------------------------------------------
# Leakage test
# ---------------------------------------------------------------------------

def run_leakage_test(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    nifty_returns: pd.Series,
    truncation_dates: List[str]
) -> Dict:
    """
    Verify that feature computation is free of look-ahead bias.

    For each truncation date `d`:
      - Compute features on FULL data.
      - Compute features on data TRUNCATED to date d.
      - Features must agree at date d (within floating-point tolerance).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide-format closing prices; columns = tickers.
    volume_df : pd.DataFrame
        Wide-format volumes; columns = tickers.
    nifty_returns : pd.Series
        NIFTY 50 1-day returns.
    truncation_dates : list of str
        ISO date strings, e.g. ['2019-06-28', '2021-12-31'].

    Returns
    -------
    dict
        {
          'passed': bool,
          'details': [
            {'date': ..., 'ticker': ..., 'feature': ...,
             'full_value': ..., 'trunc_value': ..., 'match': bool},
            ...
          ]
        }
    """
    tickers = prices_df.columns.tolist()
    details = []
    all_passed = True

    # Compute features on full data
    full_features = build_features_table(prices_df, volume_df, nifty_returns)

    for d_str in truncation_dates:
        d = pd.Timestamp(d_str)

        # Truncate data at date d (inclusive)
        prices_trunc = prices_df.loc[:d]
        volume_trunc = volume_df.loc[:d]
        nifty_trunc = nifty_returns.loc[:d]

        trunc_features = build_features_table(prices_trunc, volume_trunc, nifty_trunc)

        feature_cols = [c for c in full_features.columns
                        if c not in ("date", "ticker", "target_21d")]

        for ticker in tickers:
            # Row in full features at date d for this ticker
            full_row = full_features[
                (full_features["date"] == d) & (full_features["ticker"] == ticker)
            ]
            trunc_row = trunc_features[
                (trunc_features["date"] == d) & (trunc_features["ticker"] == ticker)
            ]

            if full_row.empty or trunc_row.empty:
                # Date might be in warm-up period — skip
                continue

            for feat in feature_cols:
                fv = full_row[feat].values[0]
                tv = trunc_row[feat].values[0]
                # Both NaN → match
                if pd.isna(fv) and pd.isna(tv):
                    match = True
                elif pd.isna(fv) or pd.isna(tv):
                    match = False
                else:
                    match = bool(np.isclose(fv, tv, rtol=1e-6, atol=1e-10))

                if not match:
                    all_passed = False
                    details.append({
                        "date": d_str,
                        "ticker": ticker,
                        "feature": feat,
                        "full_value": fv,
                        "trunc_value": tv,
                        "match": False,
                    })

    return {"passed": all_passed, "details": details}


# ---------------------------------------------------------------------------
# Full feature table builder
# ---------------------------------------------------------------------------

def build_features_table(
    prices_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    nifty_returns: pd.Series
) -> pd.DataFrame:
    """
    Build the complete, long-format feature table for all tickers.

    Columns in output:
        date, ticker,
        f_ret_1, f_ret_5, f_ret_21, f_ret_63,
        f_vol_21, f_vol_63,
        f_mom_126_21,
        f_rsi_14,
        f_volchg,
        f_dist_52wh,
        f_rel_nifty,
        target_21d

    Rules:
    - Rows where ANY feature column is NaN are dropped (warm-up rows).
    - Target is NaN for the last 21 rows of each ticker (no future price).

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide-format adjusted closing prices; columns = ticker symbols.
    volume_df : pd.DataFrame
        Wide-format volume; same shape as prices_df.
    nifty_returns : pd.Series
        NIFTY 50 1-day percentage returns, same date index.

    Returns
    -------
    pd.DataFrame
        Long-format feature table, sorted by (date, ticker).
    """
    all_records = []
    tickers = prices_df.columns.tolist()

    feature_cols = [
        "f_ret_1", "f_ret_5", "f_ret_21", "f_ret_63",
        "f_vol_21", "f_vol_63",
        "f_mom_126_21",
        "f_rsi_14",
        "f_volchg",
        "f_dist_52wh",
        "f_rel_nifty",
        "f_mkt_vol_21",
        "f_mkt_corr_63",
    ]

    for ticker in tickers:
        prices = prices_df[ticker].dropna()
        if len(prices) < 300:
            # Not enough history
            continue

        # Align volume and nifty to this ticker's price index
        volume = volume_df[ticker].reindex(prices.index).ffill()
        nifty_ret = nifty_returns.reindex(prices.index)

        daily_returns = prices.pct_change()

        # Build individual features
        past_ret = compute_past_returns(prices, windows=[1, 5, 21, 63])
        rolling_vol = compute_rolling_volatility(daily_returns, windows=[21, 63])
        mom = compute_momentum(prices, long_window=126, skip=21)
        rsi = compute_rsi(prices, period=14)
        volchg = compute_volume_change(volume, short_w=21, long_w=63)
        dist_52wh = compute_dist_52wh(prices)
        rel_nifty = compute_rel_nifty(daily_returns, nifty_ret)
        mkt_vol = compute_mkt_volatility(nifty_ret, window=21)
        mkt_corr = compute_mkt_correlation(daily_returns, nifty_ret, window=63)
        target = build_target(prices, horizon=21)

        # Combine into a ticker-level DataFrame
        ticker_df = pd.concat(
            [past_ret, rolling_vol, mom, rsi, volchg, dist_52wh, rel_nifty, mkt_vol, mkt_corr, target],
            axis=1
        )
        ticker_df.columns = feature_cols + ["target_21d"]
        ticker_df.index.name = "date"
        ticker_df = ticker_df.reset_index()
        ticker_df.insert(1, "ticker", ticker)

        # Drop warm-up rows where any FEATURE (not target) is NaN
        feature_mask = ticker_df[feature_cols].notna().all(axis=1)
        ticker_df = ticker_df[feature_mask].copy()

        all_records.append(ticker_df)

    if not all_records:
        raise ValueError("No tickers had enough data to build features.")

    result = pd.concat(all_records, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    return result
