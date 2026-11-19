# Defect Log

| Defect ID | Date | File | Symptom | Evidence | Impact | Status | Resolution / Fixed By |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | 2026-11-18 | `src/features.py` | Deprecated pandas method warning | `fillna(method='ffill')` generates deprecation warning in pandas 2.2+ | Low | **Closed** | Replaced with `.ffill()` (Student 2) |
| **DEF-02** | 2026-11-18 | `src/stage4_backtest.py` | PyPortfolioOpt binary incompatibility with Python 3.12 / numpy 2 | Process exit on C-extension import | Critical | **Closed** | Implemented pure `scipy.optimize` (SLSQP), `sklearn.covariance.LedoitWolf`, and native HRP algorithms, guaranteeing 100% stability and exact compliance (Student 4) |
| **DEF-03** | 2026-11-18 | `frontend/index.html` | Stock count mismatch in baseline FAQ | FAQ referenced "13 large companies" | Minor | **Closed** | Injected actual count "27 large companies" via `build_frontend.py` (Student 5) |
| **DEF-04** | 2026-11-19 | `notebooks/01_data.ipynb`, `data/raw/prices.csv`, `data/raw/index.csv` | Schema mismatch | Wide format written by standalone notebook (`date, RELIANCE.NS, ...`) and `Close, Return` in `index.csv` broke downstream pivoting | High | **Closed** | Refactored `01_data.ipynb` and `stage1_data.py` to strictly enforce long format: `date, ticker, adj_close, volume, filled` and `date, adj_close` (Student 1) |
| **DEF-05** | 2026-11-19 | Pipeline data files | Date mismatch | `prices.csv`, `features.csv`, and `daily_returns.csv` max dates differed when partially refreshed | High | **Closed** | Reran full pipeline sequentially; verified all three end dates are identical: `2026-11-13` (Team) |
| **DEF-06** | 2026-11-19 | `src/build_frontend.py` | Modified demo banner & non-permitted lines | Baseline diff showed 8 lines changed instead of permitted 3 lines | Medium | **Closed** | Removed demo banner rewrites; diff against `FolioWise_baseline.html` now shows exactly 3 lines changed (Student 5) |
