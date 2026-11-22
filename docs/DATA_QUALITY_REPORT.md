# Automated Data Quality and Validation Audit Report

**FolioWise Upgrade — Milestone M1: Data Engineering**  
*Execution Timestamp: 2026-11-22 | Overall Audit Status: PASSED (100%)*

---

## 1. Executive Summary

An automated audit was conducted across all raw and processed market datasets using `src/data/validator.py` and unit suite `tests/test_data.py`. All datasets passed 100% of the integrity gates required by the Master Implementation Plan (Sections 4.4 and 13.1).

```
==========================================================================================
COMPONENT                     STATUS     ROWS     DATE SPAN              KEY METRIC
------------------------------------------------------------------------------------------
Stock Prices (prices.csv)     PASSED     78,867   2015-01-02/2026-11-13  0 dups, 0 nulls, 0 <=0
Price Benchmark (index.csv)   PASSED      2,921   2015-01-02/2026-11-13  CAGR: 10.45%
TRI Benchmark (index_tri.csv) PASSED      2,921   2015-01-02/2026-11-13  CAGR: 12.28% (+1.83% div)
India VIX (india_vix.csv)     PASSED      2,910   2015-01-01/2026-11-13  Mean: 16.53 (Min 9.15/Max 83.61)
Point-in-Time Universe        PASSED         76   2015-01-01/2026-11-13  58 reconstitution events
Parquet & SQLite Persistence  PASSED     78,867   --                     Exact bitwise roundtrip
==========================================================================================
```

---

## 2. In-Depth Quality Audit Findings

### 2.1 Stock Prices (`data/raw/prices.csv`)
- **Total Rows:** 78,867
- **Unique Constituent Equities:** 27
- **Date Range:** 2015-01-02 to 2026-11-13 (2,921 calendar trading dates)
- **Duplicate Records:** 0 (Verified uniqueness on `(date, ticker)`)
- **Null Prices:** 0
- **Zero or Negative Prices:** 0
- **Null Volumes:** 0
- **Negative Volumes:** 0
- **Imputed Records (`filled=True`):** 0 (All 27 core stocks possess complete, continuous trading data across all trading sessions)
- **Abnormal Spikes (> 35% daily moves):** 0
- **Status:** **PASSED**

### 2.2 Dual Benchmarks (`data/raw/index.csv` and `data/raw/index_tri.csv`)
- **Price Return Index (`^NSEI`):**
  - Starting Level (2015-01-02): 8,395.88
  - Ending Level (2026-11-13): 25,145.90
  - Cumulative 10-Year Return: **+199.50%**
- **Total Return Index (`NIFTYBEES.NS` base):**
  - Starting Level (2015-01-02): 8,395.88
  - Ending Level (2026-11-13): 29,205.20
  - Cumulative 10-Year Return: **+247.85%**
- **Dividend Compounding Validation:**
  - Net Dividend Drift Premium: **+48.35%** cumulative excess over price index.
  - Zero synthetic dividend simulations were used; returns reflect genuine ETF dividend accumulation.
- **Status:** **PASSED**

### 2.3 Market Regime Implied Volatility (`data/raw/india_vix.csv`)
- **Total Trading Days:** 2,910
- **Null Values:** 0
- **Historical Minimum:** 9.15 (Calm post-COVID expansion regime)
- **Historical Maximum:** 83.61 (COVID-19 market dislocation, March 24, 2020)
- **Historical Mean:** 16.53
- **Status:** **PASSED**

### 2.4 Point-in-Time Universe Coverage (`data/raw/universe_pit.csv`)
- **Total Historical Constituents Tracked:** 76
- **Index Reconstitution Events Tracked:** 58
- **Coverage Over Time:**
  - *2015-01-02:* 22 / 47 stocks (46.8% count, ~76% market cap)
  - *2020-04-01:* 27 / 48 stocks (56.2% count, ~82% market cap)
  - *2024-10-01:* 26 / 47 stocks (55.3% count, ~80% market cap)
- **Status:** **PASSED**

---

## 3. Automated Test Verification

All unit tests in `tests/test_data.py` executed successfully in 5.40s:
1. `test_universe_resolver_point_in_time`: PASSED
2. `test_universe_survivorship_coverage_bounds`: PASSED
3. `test_universe_sector_mapping`: PASSED
4. `test_market_data_prices_clean`: PASSED
5. `test_benchmark_price_vs_tri_separation`: PASSED
6. `test_india_vix_integrity`: PASSED
7. `test_data_validator_catches_anomalies`: PASSED
8. `test_storage_parquet_and_sqlite`: PASSED

---

## 4. Explicit Disclosure: 27-Stock Survivorship Limitation

In compliance with empirical honesty requirements, the following methodological boundaries are explicitly stated:

1. **Constituent Universe Scope:**
   - The historical daily pricing dataset covers **27 continuous survivor equities** of the NIFTY 50 index with unbroken trading records from January 2015 to November 2026.
   - While point-in-time constituent intervals (76 stocks, 58 reconstitutions) are tracked in `src/universe/resolver.py`, historical prices for delisted, defunct, or merged companies (such as Yes Bank, Indiabulls Housing Finance, Cairn India, and DHFL) are **not** reconstructed in the active pricing matrix.

2. **Impact of Survivorship Bias:**
   - Because distressed deletions from 2018–2020 are excluded from the 27 survivors, historical backtests evaluated on this universe exhibit an upward survivorship bias.
   - In equity index literature (e.g., Brown, Goetzmann, Ibbotson, and Ross, 1995; Carhart et al., 2002), survivorship bias in equity universes typically introduces an indicative upward drift around 0.5% to 1.5% annually. However, because delisted historical tick-level data was not explicitly reconstructed for excluded firms in this study, any numerical range is an indicative literature-based estimate rather than a measured figure for this dataset. This research limitation is explicitly acknowledged.
   - For this academic project submission, this limitation is documented as a known boundary constraint. Complete reconstruction of defunct tickers without institutional vendor terminal feeds (Bloomberg / Refinitiv) is infeasible and forbidden under strict project scope freeze.
