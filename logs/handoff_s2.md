# Student 2 Handoff Note (Feature Engineering)

**To:** Student 3 (Modelling Lead)  
**From:** Student 2 (Feature Engineer)  
**Date:** 2026-11-18  
**Tag:** `handoff-s2`  
**Status:** Accepted & Verified  

---

## 1. Deliverables Summary
- **Features Table:** `data/processed/features.csv` (72,090 rows, 14 columns)
- **Source Code:** `src/features.py` and `src/stage2_features.py`
- **Leakage Test Report:** `results/leakage_test_report.csv` (100% PASS across 6 truncation dates)
- **Descriptive Statistics:** `results/feature_descriptive_stats.csv`
- **Correlation Heatmap:** `results/feature_correlation_heatmap.png`
- **Notebook:** `notebooks/02_features.ipynb`

## 2. Feature Definitions & Schema
Columns in `data/processed/features.csv`:
1. `date` (YYYY-MM-DD)
2. `ticker` (NSE symbol)
3. `f_ret_1`: 1-day backward return
4. `f_ret_5`: 5-day backward return
5. `f_ret_21`: 21-day backward return
6. `f_ret_63`: 63-day backward return
7. `f_vol_21`: 21-day rolling standard deviation of daily returns
8. `f_vol_63`: 63-day rolling standard deviation of daily returns
9. `f_mom_126_21`: 126-day momentum skipping the last 21 days
10. `f_rsi_14`: Wilder's 14-day Relative Strength Index
11. `f_volchg`: 21d volume mean / 63d volume mean - 1
12. `f_dist_52wh`: Close / 252d rolling maximum - 1 ($\le 0$)
13. `f_rel_nifty`: Stock 1d return minus NIFTY 1d return
14. `target_21d`: $P_{t+21} / P_t - 1$ (NaN only for final 21 rows of each stock)

## 3. Leakage Test Verification
Tested on 6 historical dates (`2018-12-31`, `2019-12-31`, `2021-06-30`, `2022-12-30`, `2023-12-29`, `2024-12-31`). Features computed on full data vs truncated data matched exactly ($\Delta = 0.0000000000$). Look-ahead bias is strictly 0%.

## 4. Notes for Student 3
- First valid feature date is `2016-01-13` (after 252-day warm-up window).
- Features are deliberately **unscaled** in `features.csv`. Standardize inside each expanding walk-forward training fold to prevent distribution leakage!
- Target is 21-days forward: remember to apply the **21-day purge window** before the test start date.

## 5. Acceptance Checklist
- [x] Leakage test passes on 6 dates with zero discrepancies.
- [x] Target is NaN only for the last 21 rows of each stock.
- [x] No feature uses data after date $t$.
- [x] Unscaled features in table; scaling deferred to training fold.
- [x] Signed and verified.
