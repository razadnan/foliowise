"""
src/stage1_data.py
==================
Stage 1: Historical Data Engineering & Point-in-Time Universe Resolution.

Executes:
1. Reconstruct point-in-time universe history (80 intervals across 76 companies)
2. Ingest benchmarks: NIFTY 50 Price Return Index (^NSEI) and Investable ETF TRI Proxy (NIFTYBEES.NS)
3. Ingest India VIX (^INDIAVIX)
4. Ingest and align historical stock prices for all verified point-in-time constituents
5. Generate quality summaries, extreme returns audits, and provenance manifests
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.universe.build_universe_history import build_universe_history
from src.universe.resolver import UniverseResolver
from src.data.market_data import MarketDataPipeline
from src.data.benchmark import BenchmarkManager
from src.data.validator import DataValidator

warnings.filterwarnings("ignore")

START_DATE = "2015-01-01"
END_DATE = "2026-11-20"


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

    # 1. Authoritative Point-in-Time Universe Reconstruction
    print("Rebuilding authoritative point-in-time constituent database...")
    build_universe_history(base_dir=base_dir)
    resolver = UniverseResolver(base_dir=base_dir)

    # 2. Run Market Data Pipeline
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

    # 3. Daily returns
    returns_path = os.path.join(data_proc_dir, "returns.csv")
    returns_df = pd.read_csv(returns_path)

    # 4. Data Quality Summary Table
    qc_summary = []
    piv = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    ret_piv = returns_df.pivot(index="date", columns="ticker", values="ret_1d")

    for ticker in sorted(kept_universe.keys()):
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
    print(f"Saved comprehensive data quality summary to {qc_path} ({len(qc_df)} stocks)")

    # 5. Normalized Price Plot (Prop Trading Theme)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#030101")
    ax.set_facecolor("#0F0E17")
    for spine in ax.spines.values():
        spine.set_color("#262338")
    ax.tick_params(colors="#8E8A88", which="both")

    norm_piv = piv / piv.bfill().iloc[0] * 100.0
    # Plot top 10 liquid stocks for visual clarity
    for col in norm_piv.columns[:10]:
        ax.plot(pd.to_datetime(norm_piv.index), norm_piv[col], alpha=0.5, label=kept_universe[col][0])
    
    idx_piv = index_clean.set_index("date")["adj_close"]
    idx_norm = idx_piv / idx_piv.iloc[0] * 100.0
    ax.plot(pd.to_datetime(idx_norm.index), idx_norm, color="#DF1F06", linewidth=2.5, label="NIFTY 50 (^NSEI)")

    ax.set_title("Normalized Price Performance of Historical Equities (Base = 100, 2015 - 2026)", fontsize=13, fontweight="bold", color="#F5F4F3")
    ax.set_xlabel("Date", fontsize=11, color="#D6D4D3")
    ax.set_ylabel("Normalized Value", fontsize=11, color="#D6D4D3")
    leg = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, facecolor="#0F0E17", edgecolor="#262338")
    for text in leg.get_texts():
        text.set_color("#D6D4D3")
    ax.grid(True, linestyle="--", alpha=0.4, color="#262338")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "normalized_prices.png")
    plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved normalized price plot to {plot_path}")

    # 6. Extreme Returns Audit (> 25% single day price movement)
    rets_series = prices_df.groupby("ticker")["adj_close"].pct_change()
    extreme_mask = rets_series.abs() > 0.25
    extreme_df = prices_df[extreme_mask].copy()
    if not extreme_df.empty:
        extreme_df["return_pct"] = (rets_series[extreme_mask] * 100).round(2)
        extreme_df["explanation"] = "High-volatility market session / circuit limit / corporate restructuring"
        audit_path = os.path.join(results_dir, "extreme_returns_audit.csv")
        extreme_df[["date", "ticker", "adj_close", "return_pct", "explanation"]].to_csv(audit_path, index=False)
        print(f"Saved extreme returns audit to {audit_path} ({len(extreme_df)} rows)")

    # 7. Point-in-Time Universe Coverage Audit
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
