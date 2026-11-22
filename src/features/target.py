"""
src/features/target.py
======================
Standardized Prediction Target Generator.

Constructs:
1. 20-trading-day forward price return: R_{i, t->t+20} = P_{i, t+20} / P_{i, t} - 1
2. 20-trading-day benchmark excess return: R^{excess}_{i, t} = R_{i, t->t+20} - R_{mkt, t->t+20}
3. Cross-sectional rank target: Normalized percentile rank in [0.0, 1.0] across active universe on date t
4. Legacy 21-day return target for backward compatibility
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TargetGenerator:
    """
    Constructs standardized prediction targets for machine learning models.
    """

    def __init__(self, horizon: int = 20):
        self.horizon = horizon

    def compute_forward_returns(self, prices: pd.Series, horizon: Optional[int] = None) -> pd.Series:
        """
        Compute forward lookahead return: P[t + horizon] / P[t] - 1.
        The last `horizon` rows are NaN.
        """
        h = horizon or self.horizon
        fwd_ret = prices.shift(-h) / prices - 1.0
        fwd_ret.name = f"target_{h}d_ret"
        return fwd_ret

    def compute_excess_returns(
        self,
        stock_prices: pd.Series,
        benchmark_prices: pd.Series,
        horizon: Optional[int] = None
    ) -> pd.Series:
        """
        Compute forward excess return over benchmark:
        R^{excess}_{i, t} = R_{i, t->t+h} - R_{mkt, t->t+h}
        """
        h = horizon or self.horizon
        stock_fwd = self.compute_forward_returns(stock_prices, h)
        bmk_fwd = self.compute_forward_returns(benchmark_prices, h)
        bmk_aligned = bmk_fwd.reindex(stock_fwd.index)
        excess = stock_fwd - bmk_aligned
        excess.name = f"target_excess_{h}d"
        return excess

    def compute_cross_sectional_rank(self, excess_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional percentile rank across active stocks on each date t:
        Rank is normalized to [0.0, 1.0] where 0.0 is lowest excess return and 1.0 is highest.
        
        Parameters:
        -----------
        excess_df: Wide DataFrame with date index and ticker columns containing forward excess returns.
        
        Returns:
        --------
        Wide DataFrame of cross-sectional ranks.
        """
        ranks = excess_df.rank(axis=1, ascending=True, method="average", pct=True)
        # Scale pct rank from (1/N, 1.0) to (0.0, 1.0)
        # pct = (rank) / N -> to get (rank - 1)/(N - 1):
        def scale_row(row):
            valid = row.dropna()
            n = len(valid)
            if n <= 1:
                return row.apply(lambda x: 0.5 if pd.notna(x) else np.nan)
            min_v = valid.min()
            max_v = valid.max()
            if max_v == min_v:
                return row.apply(lambda x: 0.5 if pd.notna(x) else np.nan)
            return (row - min_v) / (max_v - min_v)

        scaled_ranks = ranks.apply(scale_row, axis=1)
        return scaled_ranks

    def build_targets_table(
        self,
        prices_df: pd.DataFrame,
        benchmark_prices: pd.Series
    ) -> pd.DataFrame:
        """
        Build complete long-format target table for all stocks.
        
        Columns:
        - date
        - ticker
        - target_20d_ret
        - target_excess_20d
        - target_rank_20d (Primary standardized target)
        - target_21d (Legacy backward-compatible target)
        """
        tickers = prices_df.columns.tolist()
        bmk = benchmark_prices.reindex(prices_df.index)

        # 1. Compute wide raw forward returns
        fwd_20 = prices_df.shift(-20) / prices_df - 1.0
        fwd_21 = prices_df.shift(-21) / prices_df - 1.0
        bmk_20 = bmk.shift(-20) / bmk - 1.0

        # 2. Compute wide excess returns
        excess_20 = fwd_20.sub(bmk_20, axis=0)

        # 3. Compute wide cross-sectional rank
        rank_20 = self.compute_cross_sectional_rank(excess_20)

        # 4. Melt into long format
        records = []
        for ticker in tickers:
            df_t = pd.DataFrame({
                "date": prices_df.index,
                "ticker": ticker,
                "target_20d_ret": fwd_20[ticker].values,
                "target_excess_20d": excess_20[ticker].values,
                "target_rank_20d": rank_20[ticker].values,
                "target_21d": fwd_21[ticker].values
            })
            records.append(df_t)

        df_all = pd.concat(records, ignore_index=True)
        return df_all.sort_values(["date", "ticker"]).reset_index(drop=True)
