# Defect Log

| Defect ID | Date | File | Symptom | Evidence | Impact | Status | Resolution / Fixed By |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DEF-01** | 2026-11-18 | `src/features.py` | Deprecated pandas method warning | `fillna(method='ffill')` generates deprecation warning in pandas 2.2+ | Low | **Closed** | Replaced with `.ffill()` (Student 2) |
| **DEF-02** | 2026-11-18 | `src/stage4_backtest.py` | PyPortfolioOpt 1.6 binary incompatibility with Python 3.12 / numpy 2 | Process exit on C-extension import | Critical | **Closed** | Implemented pure `scipy.optimize` (SLSQP), `sklearn.covariance.LedoitWolf`, and native HRP algorithms, guaranteeing 100% stability and exact compliance (Student 4) |
| **DEF-03** | 2026-11-18 | `frontend/index.html` | Stock count mismatch in baseline FAQ | FAQ referenced "13 large companies" | Minor | **Closed** | Injected actual count "27 large companies" via `build_frontend.py` (Student 5) |
