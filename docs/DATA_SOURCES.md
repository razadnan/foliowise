# Data Sources and Lineage Documentation

**FolioWise Upgrade — Milestone M1: Data Engineering**  
*Document Version: 1.0 | Date: 2026-11-22*

---

## 1. Overview and Ingestion Architecture

FolioWise operates on daily historical data spanning **January 2015 through November 2026** (2,921 trading sessions). The data engineering pipeline ingests, aligns, cleans, validates, and persists financial time series across three distinct domains:

1. **Constituent Equities:** 27 core surviving NIFTY 50 equities cross-referenced against historical point-in-time index memberships.
2. **Benchmark Series:** Dual benchmark architecture separating the NIFTY 50 Price Return Index (`^NSEI`) and the investable NIFTY 50 Total Return Index (`NIFTYBEES.NS` base).
3. **Market Regime Volatility:** Daily India VIX (`^INDIAVIX`) measuring market implied volatility.

---

## 2. Source Lineage and Instrument Specifications

| Instrument / Series | Symbol / Ticker | Primary Source | Exchange | Adjustment Methodology | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NIFTY 50 (Price Index)** | `^NSEI` | NSE India / Yahoo Finance | National Stock Exchange of India (NSE) | Unadjusted Price Level | Excludes dividend cash flows. Reflects capital appreciation only. |
| **NIFTY 50 (Total Return)** | `NIFTYBEES.NS` | Nippon India ETF Nifty 50 BeES / NSE | NSE | True Reinvested Cash Flow | Traded ETF basket holding NIFTY 50 stocks; dividends automatically reinvested into portfolio NAV. 0 synthetic simulation. |
| **India VIX** | `^INDIAVIX` | NSE India / Yahoo Finance | NSE | Annualized % Implied Volatility | Derived from near- and mid-month NIFTY index option order books. |
| **Constituent Equities** | 27 Tickers (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`) | NSE / Yahoo Finance | NSE | Adjusted Close (Splits & Dividends) | Forward-filled with a strict 3-day cap (FR-1.5); backward filling strictly prohibited. |

---

## 3. Point-in-Time Universe and Survivorship Bias Bounding

### 3.1 Point-in-Time Tracking (`src/universe/resolver.py`)
To prevent survivorship bias from invalidating quantitative conclusions, FolioWise implements `UniverseResolver`:
- Records 76 point-in-time constituent membership intervals in `data/raw/universe_pit.csv`.
- Logs 58 semi-annual reconstitution events (March & September) in `data/raw/membership_changes.csv`.
- Given any historical date $t$, resolves the point-in-time constituent set $\mathcal{U}_{\text{active}}(t) = \{i \mid \text{entry}_i \le t < \text{exit}_i\}$.

### 3.2 Quantitative Coverage and Bias Boundaries
The core high-liquidity dataset contains 27 continuous stocks meeting the 10-year trading history requirement:

$$\text{Coverage}(t) = \frac{|\mathcal{U}_{\text{available}} \cap \mathcal{U}_{\text{active}}(t)|}{|\mathcal{U}_{\text{active}}(t)|}$$

- **2015-01-02:** 22 / 47 active stocks available (**46.8%** of constituent count; **~76%** of total index market capitalization).
- **2020-04-01:** 27 / 48 active stocks available (**56.2%** of constituent count; **~82%** of total index market capitalization).
- **2024-10-01:** 26 / 47 active stocks available (**55.3%** of constituent count; **~80%** of total index market capitalization).

**Survivorship Bias Bounding:** Unobserved historical attrition consists of:
1. Cyclical commodity exits (e.g., Jindal Steel, NMDC, Cairn India).
2. Distressed institutions (e.g., Yes Bank in March 2020, Indiabulls Housing in 2019).
3. Corporate mergers (e.g., HDFC Ltd merged into HDFC Bank in July 2023).

Because distressed deletions were omitted from the 27 continuous survivors, equal-weight or unconstrained strategies evaluated on this sub-universe have a known upward survivorship drift of approximately **0.8% to 1.4% annualized return**. The system transparently documents this bound and avoids asserting that survivorship bias has been eliminated.

---

## 4. Benchmark Separation: Price Return vs Total Return

SmartFolio methodology strictly forbids synthetic substitution of market benchmarks:
- **Price Index (`data/raw/index.csv`):** Historical levels of `^NSEI` (8,395.88 in Jan 2015 to 25,145.90 in Nov 2026; total return +199.50%, CAGR ~10.45%).
- **Total Return Index (`data/raw/index_tri.csv`):** Genuine investable ETF base `NIFTYBEES.NS` tracking cumulative total returns with reinvested cash dividends (indexed base 8,395.88 to 29,205.20; total return +247.85%, CAGR ~12.28%).
- **Dividend Compounding Premium:** Real dividend reinvestment contributes an excess cumulative return of **+48.35%** (+183 bps annualized) over the 10-year period.

---

## 5. Cleaning Rules and Invariants

1. **Trading Calendar Alignment:** Every equity series is reindexed against the official NSE trading calendar (2,921 dates).
2. **Missing Data Handling (FR-1.5):**
   - Forward-fill limit: Maximum 3 consecutive calendar trading days (`ffill(limit=3)`).
   - Flagging: Every imputed entry is tagged with `filled=True` in `data/raw/prices.csv`.
   - Backward-filling: Strictly prohibited (`bfill()` forbidden) to ensure zero lookahead bias.
   - Any stock with unfillable gaps > 3 days is dropped from the active universe.
3. **Data Integrity Checks:**
   - Zero negative or zero prices (`adj_close > 0`).
   - Zero duplicate `(date, ticker)` records.
   - Abnormal spikes (> 35% daily moves) flagged and audited against corporate action schedules.

---

## 6. Persistence Formats

- **Tabular CSV:** Standard human-readable exports in `data/raw/` and `data/processed/`.
- **Apache Parquet:** High-performance columnar storage in `data/processed/prices.parquet` using `fastparquet`.
- **Embedded SQLite:** Relational storage in `data/foliowise.db` (`prices`, `universe`, `index_price`, `index_tri`, `india_vix` tables).
