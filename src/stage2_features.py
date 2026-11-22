"""
src/stage2_features.py
======================
Stage 2: Feature Engineering & Leakage Verification (FolioWise Upgrade).

Standardizes 21 quantitative equity features and the 20-day cross-sectional
excess-return rank target, executes multi-date truncation leakage tests across
6 historical cutoff dates, and exports feature tables and diagnostics.

Outputs:
- data/processed/features.csv
- data/processed/features.parquet
- results/leakage_test_report.csv
- results/feature_descriptive_stats.csv
- results/feature_correlation_heatmap.png
"""

import os
import sys

# Ensure repository root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.pipeline import FeaturePipeline
from src.features.leakage_test import LeakageTester
from src.data.storage import DataStorage


def run_stage2(base_dir="."):
    print("=" * 60)
    print("STAGE 2: FEATURE ENGINEERING & LEAKAGE VERIFICATION")
    print("=" * 60)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(data_proc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    prices_path = os.path.join(data_raw_dir, "prices.csv")
    index_path = os.path.join(data_raw_dir, "index.csv")
    vix_path = os.path.join(data_raw_dir, "india_vix.csv")

    prices_df = pd.read_csv(prices_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    index_df = pd.read_csv(index_path)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date").reset_index(drop=True)
    nifty_prices = index_df.set_index("date")["adj_close"]

    vix_series = None
    if os.path.exists(vix_path):
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_col = "vix" if "vix" in vix_df.columns else "adj_close"
        vix_series = vix_df.set_index("date")[vix_col]

    # Pivot prices and volume
    piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    piv_volume = prices_df.pivot(index="date", columns="ticker", values="volume")

    pipeline = FeaturePipeline(base_dir=base_dir)
    tester = LeakageTester(pipeline=pipeline)
    storage = DataStorage(base_dir=base_dir)

    # 1. Multi-Date Leakage Test across 6 diverse cutoff dates
    truncation_dates = [
        "2018-12-31",
        "2019-12-31",
        "2021-06-30",
        "2022-12-30",
        "2023-12-29",
        "2024-12-31"
    ]
    print(f"Running multi-date leakage test across {len(truncation_dates)} cutoff dates...")
    leakage_result = tester.run_multi_date_leakage_test(
        prices_df=piv_prices,
        volume_df=piv_volume,
        nifty_prices=nifty_prices,
        vix_series=vix_series,
        truncation_dates=truncation_dates
    )

    leak_report = leakage_result["report_df"]
    leak_path = os.path.join(results_dir, "leakage_test_report.csv")
    leak_report.to_csv(leak_path, index=False)
    print(f"Saved leakage report to {leak_path}")

    if leakage_result["passed"]:
        print(f">>> LEAKAGE AUDIT PASSED: {leakage_result['total_checks']} feature checks matched exactly! <<<")
    else:
        print(f"!!! LEAKAGE AUDIT FAILED with {leakage_result['discrepancies_count']} discrepancies !!!")
        for err in leakage_result["discrepancies"][:5]:
            print(" ", err)
        raise RuntimeError("Leakage detected in feature engineering!")

    # 2. Build full feature matrix
    print("Building full standardized features table...")
    features_table = pipeline.build_features_table(
        prices_df=piv_prices,
        volume_df=piv_volume,
        nifty_prices=nifty_prices,
        vix_series=vix_series
    )

    # Save features.csv and features.parquet
    features_path = os.path.join(data_proc_dir, "features.csv")
    features_table.to_csv(features_path, index=False)
    parquet_path = os.path.join(data_proc_dir, "features.parquet")
    storage.save_parquet(features_table, parquet_path)
    storage.save_sqlite(features_table, "features")
    print(f"Saved features table to {features_path} ({len(features_table)} rows, {len(features_table.columns)} columns)")

    # 3. Descriptive statistics
    feat_cols = pipeline.FEATURE_COLS
    stat_cols = feat_cols + ["target_rank_20d", "target_excess_20d", "target_21d"]
    desc_stats = features_table[stat_cols].describe().T
    desc_stats["median"] = features_table[stat_cols].median()
    desc_stats["skew"] = features_table[stat_cols].skew()
    desc_stats["kurtosis"] = features_table[stat_cols].kurtosis()
    desc_path = os.path.join(results_dir, "feature_descriptive_stats.csv")
    desc_stats.to_csv(desc_path)
    print(f"Saved feature descriptive statistics to {desc_path}")

    # 4. Correlation heatmap
    plt.figure(figsize=(14, 11))
    corr = features_table[feat_cols + ["target_rank_20d"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        mask=mask,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7}
    )
    plt.title("Feature & Target Correlation Heatmap (NIFTY 50 Universe)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(results_dir, "feature_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_path}")

    print("\nStage 2 Feature Engineering complete successfully!")
    return features_table


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage2(base_dir)
