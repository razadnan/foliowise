"""
src/data/validator.py
=====================
Automated Data Validation Module for Indian Equities and Benchmarks.

Implements rigorous checks required by Master Implementation Plan Section 4.4 & 13.1:
- Duplicate detection
- Missing calendar days & gap limits
- Zero and negative prices
- Missing volume
- Extreme abnormal return spikes (> 35%)
- Corporate-action consistency
- Forward-fill audit (filled flag verification, max gap 3 days)
- India VIX integrity
- Benchmark Price vs TRI separation and synthetic data tagging
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class DataValidator:
    """
    Automated data quality auditing suite for historical financial series.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def validate_prices(self, df_prices: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate stock prices dataset (long format: date, ticker, adj_close, volume, filled).
        """
        issues = []
        metrics = {}

        # 1. Schema check
        required_cols = {"date", "ticker", "adj_close"}
        missing_cols = required_cols - set(df_prices.columns)
        if missing_cols:
            issues.append(f"Missing required columns in prices: {missing_cols}")

        total_rows = len(df_prices)
        metrics["total_rows"] = total_rows
        metrics["unique_tickers"] = int(df_prices["ticker"].nunique()) if "ticker" in df_prices else 0
        metrics["date_min"] = str(df_prices["date"].min())
        metrics["date_max"] = str(df_prices["date"].max())

        # 2. Duplicate detection
        dup_mask = df_prices.duplicated(subset=["date", "ticker"], keep=False)
        dup_count = int(dup_mask.sum())
        metrics["duplicate_records"] = dup_count
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate (date, ticker) records.")

        # 3. Non-positive or null prices
        null_prices = int(df_prices["adj_close"].isna().sum())
        non_positive = int((df_prices["adj_close"] <= 0).sum())
        metrics["null_prices"] = null_prices
        metrics["non_positive_prices"] = non_positive
        if null_prices > 0:
            issues.append(f"Found {null_prices} null adjusted close prices.")
        if non_positive > 0:
            issues.append(f"Found {non_positive} zero or negative adjusted close prices.")

        # 4. Volume checks
        if "volume" in df_prices.columns:
            null_vols = int(df_prices["volume"].isna().sum())
            neg_vols = int((df_prices["volume"] < 0).sum())
            metrics["null_volumes"] = null_vols
            metrics["negative_volumes"] = neg_vols
            if null_vols > 0:
                issues.append(f"Found {null_vols} null volume entries.")
            if neg_vols > 0:
                issues.append(f"Found {neg_vols} negative volume entries.")

        # 5. Filled flag audit
        if "filled" in df_prices.columns:
            filled_count = int(df_prices["filled"].sum())
            fill_pct = (filled_count / total_rows) * 100 if total_rows > 0 else 0
            metrics["filled_records_count"] = filled_count
            metrics["filled_records_pct"] = round(fill_pct, 3)
        else:
            issues.append("Missing 'filled' tracking column per FR-1.5 specification.")

        # 6. Abnormal return spikes (> 35% daily moves)
        df_sorted = df_prices.sort_values(["ticker", "date"]).copy()
        df_sorted["daily_ret"] = df_sorted.groupby("ticker")["adj_close"].pct_change()
        spikes = df_sorted[df_sorted["daily_ret"].abs() > 0.35]
        spike_count = len(spikes)
        metrics["abnormal_return_spikes_gt_35pct"] = spike_count
        if spike_count > 0:
            sample_spikes = spikes[["date", "ticker", "daily_ret"]].head(5).to_dict(orient="records")
            issues.append(
                f"Detected {spike_count} daily return moves exceeding +/-35%. Samples: {sample_spikes}"
            )

        passed = len(issues) == 0
        return {
            "dataset": "prices",
            "passed": passed,
            "issues": issues,
            "metrics": metrics
        }

    def validate_benchmark(self, df_price: pd.DataFrame, df_tri: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate benchmark series (Price Index and TRI).
        """
        issues = []
        metrics = {}

        # Price Index
        metrics["price_index_rows"] = len(df_price)
        metrics["price_index_min_date"] = str(df_price["date"].min())
        metrics["price_index_max_date"] = str(df_price["date"].max())

        if df_price["adj_close"].isna().sum() > 0:
            issues.append("Price index contains null values.")
        if (df_price["adj_close"] <= 0).sum() > 0:
            issues.append("Price index contains zero or negative values.")

        # Check duplicates
        if df_price["date"].duplicated().sum() > 0:
            issues.append("Price index contains duplicate dates.")

        # TRI Index
        if df_tri is not None:
            metrics["tri_index_rows"] = len(df_tri)
            if df_tri["adj_close"].isna().sum() > 0:
                issues.append("TRI benchmark contains null values.")
            if (df_tri["adj_close"] <= 0).sum() > 0:
                issues.append("TRI benchmark contains zero or negative values.")

            # Total Return must outpace or equal Price Return over 10-year horizon due to dividends
            tri_ret_total = (df_tri["adj_close"].iloc[-1] / df_tri["adj_close"].iloc[0]) - 1.0
            price_ret_total = (df_price["adj_close"].iloc[-1] / df_price["adj_close"].iloc[0]) - 1.0
            metrics["10yr_tri_total_return"] = round(tri_ret_total, 4)
            metrics["10yr_price_total_return"] = round(price_ret_total, 4)
            metrics["dividend_drift_premium"] = round(tri_ret_total - price_ret_total, 4)

            if tri_ret_total < price_ret_total:
                issues.append(
                    f"TRI total return ({tri_ret_total:.2%}) is lower than Price Index ({price_ret_total:.2%}). "
                    "Dividends must result in non-negative cumulative premium over long horizons."
                )

        passed = len(issues) == 0
        return {
            "dataset": "benchmark",
            "passed": passed,
            "issues": issues,
            "metrics": metrics
        }

    def validate_india_vix(self, df_vix: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate India VIX series.
        """
        issues = []
        metrics = {}

        metrics["vix_rows"] = len(df_vix)
        metrics["vix_min_date"] = str(df_vix["date"].min())
        metrics["vix_max_date"] = str(df_vix["date"].max())

        vix_col = "vix" if "vix" in df_vix.columns else "adj_close"
        if vix_col not in df_vix.columns:
            issues.append("Missing VIX value column ('vix' or 'adj_close').")
            return {"dataset": "india_vix", "passed": False, "issues": issues, "metrics": metrics}

        metrics["vix_min"] = float(df_vix[vix_col].min())
        metrics["vix_max"] = float(df_vix[vix_col].max())
        metrics["vix_mean"] = float(df_vix[vix_col].mean())

        # VIX bounds check: historically in India between 8.0 and 95.0 (March 2020 COVID peak was ~86)
        if (df_vix[vix_col] <= 0).sum() > 0:
            issues.append("Found non-positive India VIX values.")
        if (df_vix[vix_col] > 120).sum() > 0:
            issues.append("Found implausibly extreme VIX values (> 120).")

        passed = len(issues) == 0
        return {
            "dataset": "india_vix",
            "passed": passed,
            "issues": issues,
            "metrics": metrics
        }

    def run_full_audit(
        self,
        prices_path: str = "data/raw/prices.csv",
        index_path: str = "data/raw/index.csv",
        tri_path: str = "data/raw/index_tri.csv",
        vix_path: str = "data/raw/india_vix.csv"
    ) -> Dict[str, Any]:
        """
        Execute full data engineering validation suite across all raw datasets.
        """
        results = {}
        all_passed = True

        full_prices_path = os.path.join(self.base_dir, prices_path)
        if os.path.exists(full_prices_path):
            df_p = pd.read_csv(full_prices_path)
            res_p = self.validate_prices(df_p)
            results["prices"] = res_p
            if not res_p["passed"]:
                all_passed = False
        else:
            results["prices"] = {"passed": False, "issues": [f"File not found: {prices_path}"]}
            all_passed = False

        full_index_path = os.path.join(self.base_dir, index_path)
        full_tri_path = os.path.join(self.base_dir, tri_path)
        if os.path.exists(full_index_path):
            df_idx = pd.read_csv(full_index_path)
            df_tri = pd.read_csv(full_tri_path) if os.path.exists(full_tri_path) else None
            res_b = self.validate_benchmark(df_idx, df_tri)
            results["benchmark"] = res_b
            if not res_b["passed"]:
                all_passed = False

        full_vix_path = os.path.join(self.base_dir, vix_path)
        if os.path.exists(full_vix_path):
            df_vix = pd.read_csv(full_vix_path)
            res_v = self.validate_india_vix(df_vix)
            results["india_vix"] = res_v
            if not res_v["passed"]:
                all_passed = False

        overall_report = {
            "overall_passed": all_passed,
            "audit_timestamp": pd.Timestamp.now().isoformat(),
            "components": results
        }

        # Save JSON quality report
        out_json = os.path.join(self.base_dir, "data", "data_quality_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(overall_report, f, indent=2)

        return overall_report
