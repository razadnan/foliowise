"""
src/data/market_data.py
=======================
Market Data Pipeline for FolioWise.

Fetches, aligns, cleans, validates, and stores:
1. Daily stock OHLCV for all point-in-time verified historical NIFTY 50 constituents
2. NIFTY 50 Benchmark Price Index (^NSEI)
3. Investable NIFTY 50 Total Return Index Proxy (NIFTYBEES.NS)
4. India VIX (^INDIAVIX) daily volatility index
5. Sector classifications and metadata
6. Apache Parquet and SQLite persistence

Enforces:
- Point-in-time universe resolution (no hardcoded 27-survivor restriction)
- Forward fill limit = 3 days (FR-1.5)
- Strict `filled` tracking column
- Zero backward filling
- Zero silent synthetic substitutions
- Never interprets a missing price as a zero return
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

# Ensure repository root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.benchmark import BenchmarkManager
from src.data.validator import DataValidator
from src.data.storage import DataStorage
from src.universe.resolver import UniverseResolver


class MarketDataPipeline:
    """
    Core market data pipeline orchestrator for FolioWise.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.data_raw_dir = os.path.join(base_dir, "data", "raw")
        self.data_proc_dir = os.path.join(base_dir, "data", "processed")
        self.results_dir = os.path.join(base_dir, "results")

        os.makedirs(self.data_raw_dir, exist_ok=True)
        os.makedirs(self.data_proc_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.benchmark_mgr = BenchmarkManager(base_dir=base_dir)
        self.validator = DataValidator(base_dir=base_dir)
        self.storage = DataStorage(base_dir=base_dir)
        self.resolver = UniverseResolver(base_dir=base_dir)

    def fetch_india_vix(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-11-20",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch and clean daily India VIX volatility index.
        """
        vix_path = os.path.join(self.data_raw_dir, "india_vix.csv")
        if use_cache and os.path.exists(vix_path):
            return pd.read_csv(vix_path)

        print(f"Ingesting India VIX (^INDIAVIX) from {start_date} to {end_date}...")
        df_vix_raw = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
        if isinstance(df_vix_raw.columns, pd.MultiIndex):
            if "^INDIAVIX" in df_vix_raw.columns.levels[1]:
                df_vix_raw = df_vix_raw.xs("^INDIAVIX", level=1, axis=1)

        vix_col = "Close" if "Close" in df_vix_raw.columns else df_vix_raw.columns[0]
        df_vix = pd.DataFrame({
            "date": pd.to_datetime(df_vix_raw.index).strftime("%Y-%m-%d"),
            "vix": df_vix_raw[vix_col].values
        }).dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)

        df_vix.to_csv(vix_path, index=False)
        print(f"Saved India VIX to {vix_path} ({len(df_vix)} rows)")
        return df_vix

    def get_historical_universe_spec(self) -> Dict[str, Tuple[str, str]]:
        """
        Dynamically extract all historical constituent tickers, names, and sectors
        from the point-in-time universe database.
        
        Returns:
        --------
        Dict[trading_ticker -> (company_name, sector)]
        """
        df_pit = self.resolver.df_pit
        universe_spec = {}
        for _, row in df_pit.iterrows():
            current_t = row.get("current_ticker", row["ticker"])
            trading_t = self.resolver.resolve_ticker(current_t)
            name = row.get("company_name", row.get("name", trading_t))
            sector = row.get("sector", "Equity")
            universe_spec[trading_t] = (name, sector)
        return universe_spec

    def run_pipeline(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-11-20",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full data engineering pipeline:
        - Ingest Benchmark Price Index (^NSEI) & Investable ETF TRI Proxy (NIFTYBEES.NS)
        - Ingest India VIX (^INDIAVIX)
        - Ingest historical constituent prices for ALL point-in-time verified members
        - Forward fill limit = 3, filled boolean tracking, zero backward filling
        - Save universe, returns, parquet, sqlite
        - Validate datasets and return quality report
        """
        print("=" * 60)
        print("FOLIO-WISE DATA PIPELINE: HISTORICAL RECONSTRUCTION")
        print("=" * 60)

        # 1. Benchmarks
        df_price_idx, df_tri_idx = self.benchmark_mgr.fetch_and_save(
            start_date=start_date,
            end_date=end_date,
            use_cache_if_present=use_cache
        )
        calendar = pd.to_datetime(df_price_idx["date"]).sort_values().unique()

        # 2. India VIX
        df_vix = self.fetch_india_vix(start_date=start_date, end_date=end_date, use_cache=use_cache)

        # 3. Dynamic Historical Universe Extraction
        universe_spec = self.get_historical_universe_spec()
        print(f"Reconstructed point-in-time candidate universe: {len(universe_spec)} constituents")

        prices_path = os.path.join(self.data_raw_dir, "prices.csv")
        cached_prices_df = None
        cached_tickers = set()
        if os.path.exists(prices_path):
            cached_prices_df = pd.read_csv(prices_path)
            cached_tickers = set(cached_prices_df["ticker"].unique())
            print(f"Existing prices cache contains {len(cached_tickers)} tickers ({len(cached_prices_df)} rows)")

        # Download / collect prices for all tickers in universe_spec
        all_prices = []
        kept_universe = {}

        # First add cached tickers if valid
        if cached_prices_df is not None:
            for ticker in cached_tickers:
                if ticker in universe_spec:
                    t_df = cached_prices_df[cached_prices_df["ticker"] == ticker].copy()
                    all_prices.append(t_df)
                    kept_universe[ticker] = universe_spec[ticker]

        # Determine tickers that need to be fetched
        needed_tickers = [t for t in universe_spec.keys() if t not in kept_universe]
        print(f"Tickers needing historical market data ingestion: {len(needed_tickers)}")

        for ticker in needed_tickers:
            name, sector = universe_spec[ticker]
            print(f"Downloading {ticker} ({name})...")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
                if df.empty or len(df) < 50:
                    print(f"  -> Insufficient/no data for {ticker} ({len(df)} rows)")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    if ticker in df.columns.levels[1]:
                        df = df.xs(ticker, level=1, axis=1)

                adj_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                vol_col = "Volume" if "Volume" in df.columns else None

                stock_df = pd.DataFrame({
                    "date": pd.to_datetime(df.index),
                    "adj_close": df[adj_col].values,
                    "volume": df[vol_col].values if vol_col else 0
                }).dropna(subset=["adj_close"]).drop_duplicates(subset=["date"])

                # Align against market calendar
                # Only align dates within the stock's actual trading span (do not backward fill before IPO/listing!)
                stock_min_date = stock_df["date"].min()
                stock_max_date = stock_df["date"].max()
                active_calendar = [d for d in calendar if stock_min_date <= d <= stock_max_date]

                stock_df = stock_df.set_index("date").reindex(active_calendar)
                raw_na = stock_df["adj_close"].isna()

                # FR-1.5: Forward-fill limit=3, record filled flag, zero backward fill
                stock_df["adj_close"] = stock_df["adj_close"].ffill(limit=3)
                stock_df["volume"] = stock_df["volume"].fillna(0)
                stock_df["filled"] = (raw_na & stock_df["adj_close"].notna()).astype(bool)

                # Keep strictly valid observations
                valid_mask = stock_df["adj_close"].notna()
                clean_df = stock_df[valid_mask].reset_index()
                clean_df["ticker"] = ticker
                clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

                all_prices.append(clean_df[["date", "ticker", "adj_close", "volume", "filled"]])
                kept_universe[ticker] = (name, sector)
                print(f"  -> Successfully ingested {ticker}: {len(clean_df)} daily observations ({clean_df['date'].min()} to {clean_df['date'].max()})")
            except Exception as e:
                print(f"  -> Error fetching {ticker}: {e}")

        if not all_prices:
            raise RuntimeError("No stock prices collected!")

        all_prices_df = pd.concat(all_prices, ignore_index=True).drop_duplicates(subset=["date", "ticker"]).sort_values(["ticker", "date"]).reset_index(drop=True)
        all_prices_df.to_csv(prices_path, index=False)
        print(f"Saved consolidated historical stock prices to {prices_path} ({len(all_prices_df)} rows, {all_prices_df['ticker'].nunique()} tickers)")

        # 4. Save Universe CSV
        universe_df = pd.DataFrame([
            {"ticker": t, "name": name, "sector": sec}
            for t, (name, sec) in sorted(kept_universe.items())
        ])
        universe_path = os.path.join(self.data_raw_dir, "universe.csv")
        universe_df.to_csv(universe_path, index=False)
        print(f"Saved investable universe to {universe_path} ({len(universe_df)} stocks)")

        # 5. Compute and Save Daily Returns (data/processed/returns.csv)
        # Never interpret missing prices as zero returns; pct_change within ticker
        ret_df = all_prices_df.sort_values(["ticker", "date"]).copy()
        ret_df["ret_1d"] = ret_df.groupby("ticker")["adj_close"].pct_change()
        # Drop the first unobservable NaN return per ticker
        ret_clean = ret_df.dropna(subset=["ret_1d"])[["date", "ticker", "ret_1d"]].reset_index(drop=True)
        returns_path = os.path.join(self.data_proc_dir, "returns.csv")
        ret_clean.to_csv(returns_path, index=False)
        print(f"Saved daily returns to {returns_path} ({len(ret_clean)} rows)")

        # 6. Save Parquet and SQLite formats
        parquet_path = os.path.join(self.data_proc_dir, "prices.parquet")
        self.storage.save_parquet(all_prices_df, parquet_path)
        self.storage.save_sqlite(all_prices_df, "prices")
        self.storage.save_sqlite(universe_df, "universe")
        self.storage.save_sqlite(df_price_idx, "index_price")
        self.storage.save_sqlite(df_tri_idx, "index_tri")
        self.storage.save_sqlite(df_vix, "india_vix")
        print("Exported Apache Parquet and SQLite transactional datasets successfully.")

        # 7. Run Data Quality Validation Audit
        audit_report = self.validator.run_full_audit(
            prices_path="data/raw/prices.csv",
            index_path="data/raw/index.csv",
            tri_path="data/raw/index_tri.csv",
            vix_path="data/raw/india_vix.csv"
        )
        print(f"Data Quality Audit Status: {'PASSED' if audit_report['overall_passed'] else 'FAILED'}")

        return {
            "all_prices_df": all_prices_df,
            "universe_df": universe_df,
            "index_price_df": df_price_idx,
            "index_tri_df": df_tri_idx,
            "vix_df": df_vix,
            "audit_report": audit_report
        }
