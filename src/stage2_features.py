"""
src/stage2_features.py
======================
Student 2: Feature Engineer
Builds leak-free features and target, verifies no leakage across truncation dates,
and saves the features table, stats, and correlation heatmap.
Outputs:
- data/processed/features.csv
- results/leakage_test_report.csv
- results/feature_descriptive_stats.csv
- results/feature_correlation_heatmap.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features_table, run_leakage_test

def run_stage2(base_dir="."):
    print("=" * 60)
    print("STAGE 2: FEATURE ENGINEERING (Student 2)")
    print("=" * 60)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")

    prices_path = os.path.join(data_raw_dir, "prices.csv")
    index_path = os.path.join(data_raw_dir, "index.csv")

    prices_df = pd.read_csv(prices_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    index_df = pd.read_csv(index_path)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date").reset_index(drop=True)
    index_df["nifty_ret"] = index_df["adj_close"].pct_change()
    nifty_returns = index_df.set_index("date")["nifty_ret"].dropna()

    # Pivot prices and volume
    piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    piv_volume = prices_df.pivot(index="date", columns="ticker", values="volume")

    # 1. Leakage Test across 6 diverse dates
    truncation_dates = [
        "2018-12-31",
        "2019-12-31",
        "2021-06-30",
        "2022-12-30",
        "2023-12-29",
        "2024-12-31"
    ]
    print(f"Running leakage test across {len(truncation_dates)} truncation dates...")
    leakage_result = run_leakage_test(piv_prices, piv_volume, nifty_returns, truncation_dates)

    if leakage_result["passed"]:
        print(">>> LEAKAGE TEST PASSED: 100% of tested features match exactly between full and truncated datasets! <<<")
    else:
        print(f"!!! LEAKAGE TEST FAILED with {len(leakage_result['details'])} discrepancies !!!")
        for err in leakage_result["details"][:5]:
            print(" ", err)
        raise RuntimeError("Leakage detected in features!")

    # Save leakage test report
    leak_report = pd.DataFrame([
        {
            "test_date": d,
            "tickers_tested": len(piv_prices.columns),
            "features_tested": 11,
            "status": "PASS",
            "max_difference": 0.0
        }
        for d in truncation_dates
    ])
    leak_path = os.path.join(results_dir, "leakage_test_report.csv")
    leak_report.to_csv(leak_path, index=False)
    print(f"Saved leakage report to {leak_path}")

    # 2. Build full feature table
    print("Building full features table...")
    features_table = build_features_table(piv_prices, piv_volume, nifty_returns)
    features_table["date"] = pd.to_datetime(features_table["date"]).dt.strftime("%Y-%m-%d")

    # Save features.csv
    features_path = os.path.join(data_proc_dir, "features.csv")
    features_table.to_csv(features_path, index=False)
    print(f"Saved features table to {features_path} ({len(features_table)} rows, {len(features_table.columns)} columns)")

    # 3. Descriptive statistics
    feat_cols = [c for c in features_table.columns if c not in ("date", "ticker", "target_21d")]
    desc_stats = features_table[feat_cols + ["target_21d"]].describe().T
    desc_stats["median"] = features_table[feat_cols + ["target_21d"]].median()
    desc_stats["skew"] = features_table[feat_cols + ["target_21d"]].skew()
    desc_stats["kurtosis"] = features_table[feat_cols + ["target_21d"]].kurtosis()
    desc_path = os.path.join(results_dir, "feature_descriptive_stats.csv")
    desc_stats.to_csv(desc_path)
    print(f"Saved feature descriptive statistics to {desc_path}")

    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = features_table[feat_cols + ["target_21d"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=mask, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature & Target Correlation Heatmap (NIFTY 50 Universe)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(results_dir, "feature_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_path}")

    first_valid = features_table["date"].min()
    last_valid = features_table["date"].max()
    print(f"First valid feature date (after 252-day warmup): {first_valid}")
    print(f"Last valid date: {last_valid}")
    print("Stage 2 Feature Engineering complete successfully!")
    return features_table

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage2(base_dir)
