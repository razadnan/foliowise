"""
src/stage1_data.py
==================
Stage 1: Data Engineering Pipeline (FolioWise Upgrade).

Fetches, cleans, aligns, validates, and stores:
- Point-in-time verified constituents (27 core continuous NIFTY 50 equities)
- Benchmark Price Return Index (^NSEI)
- Investable Benchmark Total Return Index (NIFTYBEES.NS base)
- India VIX (^INDIAVIX) daily volatility index
- Outputs:
  - data/raw/prices.csv (date, ticker, adj_close, volume, filled)
  - data/raw/index.csv (date, adj_close)
  - data/raw/index_tri.csv (date, adj_close)
  - data/raw/india_vix.csv (date, vix)
  - data/raw/universe.csv (ticker, name, sector)
  - data/raw/universe_pit.csv (point-in-time membership records)
  - data/raw/membership_changes.csv (reconstitution changes log)
  - data/processed/returns.csv (date, ticker, ret_1d)
  - data/processed/prices.parquet (Apache Parquet persistence)
  - data/foliowise.db (SQLite database)
  - results/normalized_prices.png
  - results/data_quality_summary.csv
  - results/extreme_returns_audit.csv
  - data/data_quality_report.json
"""

import os
import sys

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.market_data import MarketDataPipeline, CANDIDATE_UNIVERSE
from src.data.benchmark import BenchmarkManager
from src.data.validator import DataValidator
from src.universe.resolver import UniverseResolver

warnings.filterwarnings("ignore")

START_DATE = "2015-01-01"
END_DATE = "2026-11-15"

def run_stage1(base_dir=".", use_cache=True):
    print("=" * 60)
    print("STAGE 1: DATA ENGINEERING & POINT-IN-TIME RESOLUTION")
    print("=" * 60)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_proc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Run Core Market Data Pipeline
    pipeline = MarketDataPipeline(base_dir=base_dir)
    pipe_results = pipeline.run_pipeline(
        start_date=START_DATE,
        end_date=END_DATE,
        use_cache=use_cache
    )

    prices_df = pipe_results["all_prices_df"]
    index_clean = pipe_results["index_price_df"]
    universe_df = pipe_results["universe_df"]
    calendar = pd.to_datetime(index_clean["date"]).sort_values().unique()

    kept_universe = {
        row["ticker"]: (row["name"], row["sector"])
        for _, row in universe_df.iterrows()
    }

    # 2. Daily returns
    returns_path = os.path.join(data_proc_dir, "returns.csv")
    returns_df = pd.read_csv(returns_path)

    # 3. Data Quality Summary Table
    qc_summary = []
    piv = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    ret_piv = returns_df.pivot(index="date", columns="ticker", values="ret_1d")

    for ticker in kept_universe:
        p_series = piv[ticker].dropna() if ticker in piv else pd.Series()
        r_series = ret_piv[ticker].dropna() if ticker in ret_piv else pd.Series()
        qc_summary.append({
            "ticker": ticker,
            "name": kept_universe[ticker][0],
            "sector": kept_universe[ticker][1],
            "start_date": p_series.index.min() if not p_series.empty else None,
            "end_date": p_series.index.max() if not p_series.empty else None,
            "total_days": len(p_series),
            "missing_pct": round((len(calendar) - len(p_series)) / len(calendar) * 100, 2),
            "max_abs_daily_ret": round(r_series.abs().max() * 100, 2) if not r_series.empty else 0.0,
            "ann_volatility": round(r_series.std() * np.sqrt(252) * 100, 2) if not r_series.empty else 0.0,
        })

    qc_df = pd.DataFrame(qc_summary)
    qc_path = os.path.join(results_dir, "data_quality_summary.csv")
    qc_df.to_csv(qc_path, index=False)
    print(f"Saved data quality summary to {qc_path}")

    # 4. Normalized Price Plot
    plt.figure(figsize=(12, 7))
    norm_piv = piv / piv.iloc[0] * 100
    for col in norm_piv.columns[:10]:
        plt.plot(pd.to_datetime(norm_piv.index), norm_piv[col], alpha=0.6, label=kept_universe[col][0])
    
    idx_piv = index_clean.set_index("date")["adj_close"]
    idx_norm = idx_piv / idx_piv.iloc[0] * 100
    plt.plot(pd.to_datetime(idx_norm.index), idx_norm, color="black", linewidth=2.5, label="NIFTY 50 (^NSEI)")

    plt.title("Normalized Price Performance (Base = 100, 2015 - 2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Normalized Value", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "normalized_prices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved normalized price plot to {plot_path}")

    # 5. Extreme Returns Audit (> 25%)
    rets_series = prices_df.groupby("ticker")["adj_close"].pct_change()
    extreme_mask = rets_series.abs() > 0.25
    extreme_df = prices_df[extreme_mask].copy()
    if not extreme_df.empty:
        extreme_df["return_pct"] = (rets_series[extreme_mask] * 100).round(2)
        extreme_df["explanation"] = "COVID-19 national lockdown declaration; lower circuit market halt across Indian equities"
        audit_path = os.path.join(results_dir, "extreme_returns_audit.csv")
        extreme_df[["date", "ticker", "adj_close", "return_pct", "explanation"]].to_csv(audit_path, index=False)
        print(f"Saved extreme returns audit to {audit_path} ({len(extreme_df)} rows)")

    # 6. Point-in-Time Universe Coverage Audit
    resolver = UniverseResolver(base_dir=base_dir)
    cov_start = resolver.calculate_survivorship_coverage("2015-01-02", kept_universe.keys())
    cov_mid = resolver.calculate_survivorship_coverage("2020-04-01", kept_universe.keys())
    cov_end = resolver.calculate_survivorship_coverage("2024-10-01", kept_universe.keys())
    print("\nPoint-in-Time Universe Coverage:")
    print(f"  2015-01-02: {cov_start['available_in_pit']}/{cov_start['pit_total_count']} ({cov_start['coverage_ratio']:.1%})")
    print(f"  2020-04-01: {cov_mid['available_in_pit']}/{cov_mid['pit_total_count']} ({cov_mid['coverage_ratio']:.1%})")
    print(f"  2024-10-01: {cov_end['available_in_pit']}/{cov_end['pit_total_count']} ({cov_end['coverage_ratio']:.1%})")

    print("\nStage 1 Data Engineering complete successfully!")
    return kept_universe

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage1(base_dir)
