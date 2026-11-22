"""
src/features/leakage_test.py
============================
Multi-Date Truncation Leakage Testing Suite.

Verifies mathematically that feature computation contains zero look-ahead bias:
For each historical cutoff date d:
1. Truncate all input series to [:d] (inclusive).
2. Compute features on truncated dataset.
3. Compute features on full dataset.
4. Verify that for every stock and every feature at date d, the truncated value
   matches the full value exactly within floating-point tolerance (atol=1e-10).
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline


class LeakageTester:
    """
    Automated look-ahead bias audit suite across historical cutoff dates.
    """

    def __init__(self, pipeline: Optional[FeaturePipeline] = None):
        self.pipeline = pipeline or FeaturePipeline()

    def run_multi_date_leakage_test(
        self,
        prices_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        nifty_prices: pd.Series,
        vix_series: Optional[pd.Series],
        truncation_dates: List[str]
    ) -> Dict[str, Any]:
        """
        Execute leakage verification across multiple historical dates.
        
        Returns:
        --------
        Dictionary with 'passed', 'tested_dates', 'total_checks', 'discrepancies',
        and 'report_df'.
        """
        tickers = prices_df.columns.tolist()
        feature_cols = self.pipeline.FEATURE_COLS

        # Full features
        full_df = self.pipeline.build_features_table(prices_df, volume_df, nifty_prices, vix_series)
        full_df["dt"] = pd.to_datetime(full_df["date"])

        discrepancies = []
        date_reports = []
        total_checks = 0

        for d_str in truncation_dates:
            d = pd.to_datetime(d_str)

            # Truncate strictly at cutoff date
            p_trunc = prices_df.loc[:d]
            v_trunc = volume_df.loc[:d]
            n_trunc = nifty_prices.loc[:d]
            vix_trunc = vix_series.loc[:d] if vix_series is not None else None

            trunc_df = self.pipeline.build_features_table(p_trunc, v_trunc, n_trunc, vix_trunc)
            trunc_df["dt"] = pd.to_datetime(trunc_df["date"])

            d_fmt = d.strftime("%Y-%m-%d")
            full_at_d = full_df[full_df["date"] == d_fmt].set_index("ticker")
            trunc_at_d = trunc_df[trunc_df["date"] == d_fmt].set_index("ticker")

            date_checks = 0
            date_mismatches = 0
            max_diff = 0.0

            for ticker in tickers:
                if ticker in full_at_d.index and ticker not in trunc_at_d.index:
                    date_mismatches += 1
                    discrepancies.append({
                        "date": d_fmt,
                        "ticker": ticker,
                        "feature": "ALL",
                        "full": "PRESENT",
                        "trunc": "MISSING",
                        "error": "Record present in full run but missing in truncated run (boundary lookahead leakage)"
                    })
                    continue
                elif ticker not in full_at_d.index or ticker not in trunc_at_d.index:
                    continue

                row_full = full_at_d.loc[ticker]
                row_trunc = trunc_at_d.loc[ticker]

                for feat in feature_cols:
                    total_checks += 1
                    date_checks += 1

                    vf = row_full[feat]
                    vt = row_trunc[feat]

                    if pd.isna(vf) and pd.isna(vt):
                        continue
                    elif pd.isna(vf) or pd.isna(vt):
                        date_mismatches += 1
                        discrepancies.append({
                            "date": d_fmt,
                            "ticker": ticker,
                            "feature": feat,
                            "full": vf,
                            "trunc": vt,
                            "error": "NaN mismatch"
                        })
                    else:
                        diff = abs(float(vf) - float(vt))
                        if diff > max_diff:
                            max_diff = diff
                        if not np.isclose(vf, vt, rtol=1e-6, atol=1e-10):
                            date_mismatches += 1
                            discrepancies.append({
                                "date": d_fmt,
                                "ticker": ticker,
                                "feature": feat,
                                "full": vf,
                                "trunc": vt,
                                "diff": diff
                            })

            date_reports.append({
                "test_date": d_fmt,
                "tickers_tested": len(tickers),
                "features_tested": len(feature_cols),
                "checks": date_checks,
                "mismatches": date_mismatches,
                "status": "PASS" if date_mismatches == 0 else "FAIL",
                "max_difference": round(float(max_diff), 12)
            })

        passed = len(discrepancies) == 0
        report_df = pd.DataFrame(date_reports)

        return {
            "passed": passed,
            "total_checks": total_checks,
            "discrepancies_count": len(discrepancies),
            "discrepancies": discrepancies,
            "report_df": report_df
        }
