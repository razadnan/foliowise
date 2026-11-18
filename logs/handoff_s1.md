# Student 1 Handoff Note (Data Engineering)

**To:** Student 2 (Feature Engineer)  
**From:** Student 1 (Data Engineer)  
**Date:** 2026-11-18  
**Tag:** `handoff-s1`  
**Status:** Accepted & Verified  

---

## 1. Deliverables Summary
- **Raw Prices:** `data/raw/prices.csv` (78,867 rows, columns: `date, ticker, adj_close, volume`)
- **Raw Index:** `data/raw/index.csv` (2,921 rows, columns: `date, adj_close`)
- **Universe Metadata:** `data/raw/universe.csv` (27 stocks, columns: `ticker, name, sector`)
- **Daily Returns:** `data/processed/returns.csv` (78,840 rows, columns: `date, ticker, ret_1d`)
- **Data Quality Summary:** `results/data_quality_summary.csv`
- **Normalized Price Plot:** `results/normalized_prices.png`
- **Notebook:** `notebooks/01_data.ipynb` and reusable source `src/stage1_data.py`

## 2. Key Numbers & Characteristics
- **Date Range:** 2015-01-01 to 2026-11-13 (2,921 trading days aligned to NIFTY 50 calendar).
- **Universe Size:** 27 stocks representing 8 sectors (Banking, IT, Energy, FMCG, Pharma, Auto, Industrials, Consumer). All target criteria met (>20 stocks, history from 2015).
- **Missing Data Treatment:** Aligned against NIFTY index calendar; forward-filled short gaps ($\le 3$ days); zero missing values remain in cleaned prices.
- **Extreme Returns:** All daily returns checked and verified within realistic market circuit filter limits ($\le 35\%$).

## 3. Notes for Student 2
- The dataset starts on `2015-01-01`. Because 52-week rolling high and 126-day momentum require lookbacks of up to 252 trading days, expect warm-up rows until approximately `2016-01-13`.
- Volume for some market trading sessions can occasionally be zero on holiday half-days; handled via forward fill.

## 4. Acceptance Checklist
- [x] Universe has $\ge 20$ stocks (achieved 27).
- [x] No missing `adj_close` in cleaned prices.
- [x] No daily return above 35% left unexplained.
- [x] Rerunning `src/stage1_data.py` reproduces identical files.
- [x] Signed and verified.
