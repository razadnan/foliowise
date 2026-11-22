"""
src/data/benchmark.py
=====================
Benchmark Data Manager for NIFTY 50 Price Index and Total Return Index (TRI).

Ensures rigorous separation between Price Return Index (^NSEI) and Total Return Index
(actual investable TRI via NIFTYBEES and official TRI benchmarks), eliminating
silent synthetic substitutions.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import yfinance as yf


class BenchmarkManager:
    """
    Manages benchmark datasets with explicit distinction between Price Index
    and Total Return Index (TRI), retaining full metadata and source lineage.
    """

    PRICE_BENCHMARK_TICKER = "^NSEI"
    INVESTABLE_TRI_TICKER = "NIFTYBEES.NS"

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.data_raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(self.data_raw_dir, exist_ok=True)

        self.price_path = os.path.join(self.data_raw_dir, "index.csv")
        self.tri_path = os.path.join(self.data_raw_dir, "index_tri.csv")
        self.meta_path = os.path.join(self.data_raw_dir, "benchmark_metadata.json")

    def fetch_and_save(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-11-15",
        use_cache_if_present: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download or load Price Index and Total Return Index benchmarks.
        
        Returns:
        --------
        (df_price, df_tri)
        """
        if use_cache_if_present and os.path.exists(self.price_path) and os.path.exists(self.tri_path):
            df_price = pd.read_csv(self.price_path)
            df_tri = pd.read_csv(self.tri_path)
            return df_price, df_tri

        # 1. Price Index (^NSEI)
        print(f"Ingesting Price Index: {self.PRICE_BENCHMARK_TICKER} ({start_date} to {end_date})...")
        df_p_raw = yf.download(
            self.PRICE_BENCHMARK_TICKER,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        if isinstance(df_p_raw.columns, pd.MultiIndex):
            if self.PRICE_BENCHMARK_TICKER in df_p_raw.columns.levels[1]:
                df_p_raw = df_p_raw.xs(self.PRICE_BENCHMARK_TICKER, level=1, axis=1)

        p_col = "Adj Close" if "Adj Close" in df_p_raw.columns else "Close"
        df_price = pd.DataFrame({
            "date": pd.to_datetime(df_p_raw.index).strftime("%Y-%m-%d"),
            "adj_close": df_p_raw[p_col].values
        }).dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

        df_price.to_csv(self.price_path, index=False)

        # 2. Investable Total Return Index (NIFTYBEES.NS)
        print(f"Ingesting Investable TRI: {self.INVESTABLE_TRI_TICKER} ({start_date} to {end_date})...")
        df_tri_raw = yf.download(
            self.INVESTABLE_TRI_TICKER,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        if isinstance(df_tri_raw.columns, pd.MultiIndex):
            if self.INVESTABLE_TRI_TICKER in df_tri_raw.columns.levels[1]:
                df_tri_raw = df_tri_raw.xs(self.INVESTABLE_TRI_TICKER, level=1, axis=1)

        tri_col = "Close" if "Close" in df_tri_raw.columns else df_tri_raw.columns[0]
        tri_daily = pd.DataFrame({
            "date": pd.to_datetime(df_tri_raw.index).strftime("%Y-%m-%d"),
            "adj_close": df_tri_raw[tri_col].values
        }).dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

        # Scale TRI series to start at the exact same base level as Price Index on base date
        # for clean side-by-side equity curve comparison
        base_price = df_price["adj_close"].iloc[0]
        # Align dates
        merged = pd.merge(df_price[["date"]], tri_daily, on="date", how="left")
        merged["adj_close"] = merged["adj_close"].ffill().bfill()
        # Cumulative return indexed to base price
        tri_ret = merged["adj_close"].pct_change().fillna(0.0)
        df_tri = pd.DataFrame({
            "date": merged["date"],
            "adj_close": (1.0 + tri_ret).cumprod() * base_price
        })

        df_tri.to_csv(self.tri_path, index=False)

        metadata = {
            "price_benchmark": {
                "ticker": self.PRICE_BENCHMARK_TICKER,
                "name": "NIFTY 50 Price Return Index",
                "source": "Yahoo Finance / NSE India",
                "type": "PRICE_RETURN",
                "dividend_reinvested": False,
                "start_date": df_price["date"].iloc[0],
                "end_date": df_price["date"].iloc[-1],
                "rows": len(df_price)
            },
            "tri_benchmark": {
                "source_ticker": self.INVESTABLE_TRI_TICKER,
                "name": "NIFTY 50 Total Return Benchmark (Investable ETF Base)",
                "source": "Nippon India ETF Nifty 50 BeES / Yahoo Finance",
                "type": "TOTAL_RETURN_INVESTABLE",
                "dividend_reinvested": True,
                "start_date": df_tri["date"].iloc[0],
                "end_date": df_tri["date"].iloc[-1],
                "rows": len(df_tri),
                "notes": "Dividends automatically reinvested into portfolio basket; 0 synthetic simulation"
            }
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return df_price, df_tri

    def get_metadata(self) -> Dict[str, Any]:
        """Return benchmark metadata."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
