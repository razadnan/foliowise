# FOLIOWISE SYSTEM UPGRADE & VERIFICATION REPORT
## Integration of SmartFolio Research Methodology (Milestones M0 – M11)

**Project:** FolioWise (v2.0 SmartFolio Integrated)  
**System Type:** Adaptive, Explainable Machine Learning System for Portfolio Optimization  
**Target Market:** Indian Equity Market (NIFTY 50 Universe)  
**Verification Date:** November 2026  
**Test Suite Status:** 43 / 43 Tests Passing (`pytest tests/`)  
**Investor UI Status:** 100% Preserved (3-line permitted data injection verified in `frontend/baseline_diff.txt`)

---

## 1. Executive Summary

The FolioWise project has been upgraded by integrating the quantitative research methodology, statistical rigor, and governance framework of **SmartFolio**. The upgrade was executed across eleven structured milestones (M0 through M11) without rewriting the application from scratch and without altering the core investor-facing user experience.

### Key Milestones Achieved
- **M0 (Audit & Baseline):** Audited repository architecture, established reproducible Python 3.12 environment with CPU PyTorch, FastParquet, and XGBoost; documented in `docs/EXISTING_SYSTEM_AUDIT.md`.
- **M1 (Point-in-Time Data Engineering):** Ingested 2,921 trading sessions of NIFTY 50 prices (27 tickers), resolved point-in-time constituent coverage (76 historical constituent intervals, 58 reconstitutions), separated Price Return Index from Investable TRI (`NIFTYBEES.NS`), and ingested India VIX (^INDIAVIX); documented in `docs/DATA_SOURCES.md` and `docs/DATA_QUALITY_REPORT.md`.
- **M2 (Factor Engineering & Target Normalization):** Implemented 21 quantitative features, created standardized 20-day cross-sectional rank target `target_rank_20d`, and proved zero lookahead leakage across 3,402 feature checks over 6 historical cutoff dates; documented in `docs/FEATURE_DOCUMENTATION.md` and `results/leakage_test_report.csv`.
- **M3 (Machine Learning Integration):** Implemented Ridge linear shrinkage, regularized XGBoost, PyTorch LSTM sequence model, and validation-tuned XGBoost-LSTM ensemble; computed Rank IC, ICIR, Newey-West HAC t-statistics, Directional Accuracy, and Top-Bottom spreads across 81 monthly walk-forward rebalance dates.
- **M4 (Risk & Optimization):** Implemented Ledoit-Wolf shrinkage covariance, spectral eigenvalue PSD validation, and `PortfolioOptimizer` supporting SLSQP Markowitz, Minimum Variance, HRP, and Equal Risk Contribution (ERC) with 10% asset caps, 35% sector caps, and L1 turnover penalties.
- **M5 (Backtesting Engine):** Built `BacktestEngine` with strict $T+1$ next-day execution timing, intra-month daily weight drift, and itemized Indian statutory trading cost schedule (20 bps delivery schedule).
- **M6 (SmartFolio Experiment Ladder E0–E6):** Formalized experiments E0 through E6; executed stationary block bootstrap hypothesis testing (Politis & Romano 1994, 1,000 resamples) for Sharpe differences against baseline.
- **M7 (Adaptive Learning & Model Governance):** Built `FeedbackEngine` (logging 15,309 prediction feedback rows and 574 portfolio cycles) and `ChampionChallengerEvaluator` with statistical promotion hurdles and degradation rollbacks.
- **M8 (Allocation Explainability):** Built `AllocationExplainer` decomposing portfolio weights into Marginal Risk Contributions (MRC), Percentage Risk Contributions (PRC), Diversification Benefits, sector limits, and predictive SHAP factor drivers.
- **M9 (Frontend Integration):** Injected updated data into `frontend/index.html` (strictly preserving Investor UI) and enriched `frontend/dashboard.html` (Developer UI) with live experiment ladder, bootstrap tables, attribution tables, and governance audit trails.
- **M10 (Quality Assurance):** Developed end-to-end integration test (`tests/test_e2e.py`); achieved 100% pass rate across all 43 tests.
- **M11 (Final Delivery):** Finalized system documentation, updated `README.md`, and created immutable git release tag.

---

## 2. SmartFolio Experiment Ladder (E0 – E6) Performance

All strategies evaluated over the 2020–2026 walk-forward test period under strict next-day execution and 20 bps statutory round-trip transaction costs:

| Experiment ID | Strategy Description | CAGR (%) | Annual Vol (%) | Sharpe Ratio | Sortino Ratio | Max DD (%) | Monthly Turnover (%) | Annual Cost Drag (bps) | vs E0 95% Bootstrap CI | p-value |
|---|---|---|---|---|---|---|---|---|---|---|
| **E0** | 1/N Equal Weight (Baseline) | **17.11%** | 16.32% | **0.68** | 0.93 | -33.24% | 3.62% | 10.4 bps | Benchmark | — |
| **E1** | Traditional Markowitz (Sample Cov) | 9.60% | 17.60% | 0.20 | 0.28 | -32.88% | 26.20% | 71.0 bps | [-0.890, -0.026] | 0.060 |
| **E2** | Markowitz + Ledoit-Wolf Shrinkage | 9.66% | 17.65% | 0.21 | 0.28 | -33.38% | 25.82% | 70.0 bps | [-0.878, -0.024] | 0.059 |
| **E3** | XGBoost + Markowitz (Caps + Reg) | **15.30%** | 16.12% | **0.58** | 0.81 | **-27.19%** | 30.81% | 87.9 bps | [-0.406, +0.185] | 0.536 |
| **E4** | LSTM Sequence Model + Markowitz | 12.77% | 15.48% | 0.44 | 0.60 | -31.22% | 40.21% | 112.2 bps | [-0.501, +0.033] | 0.114 |
| **E5** | XGB-LSTM Ensemble + Markowitz | 12.61% | 15.92% | 0.41 | 0.57 | -29.61% | 49.23% | 137.4 bps | [-0.528, +0.040] | 0.110 |
| **E6** | Adaptive FolioWise (Champion-Challenger) | 11.66% | 15.55% | 0.36 | 0.52 | **-27.19%** | 26.55% | 73.3 bps | [-0.717, +0.061] | 0.165 |

### Empirical Insights & DeMiguel Phenomenon
1. **The 1/N Benchmark Superiority:** Naive equal weighting (E0) delivers the highest raw Sharpe ratio (0.68) and CAGR (17.11%). Because E0 requires negligible monthly rebalancing (3.62% turnover), transaction cost drag is only 10.4 bps/year compared to 70–137 bps for active optimizers.
2. **Estimation Error in Mean Vectors:** Unregularized Markowitz (E1, Sharpe 0.20) suffers severe out-of-sample degradation due to error maximization in sample mean returns. Ledoit-Wolf covariance shrinkage (E2, Sharpe 0.21) stabilizes covariance condition numbers but cannot overcome return forecast noise alone.
3. **Machine Learning Advantage with Risk Budgeting:** E3 (XGBoost + Markowitz) achieves the highest risk-adjusted performance among optimized strategies (Sharpe 0.58, CAGR 15.30%) and the lowest maximum drawdown (-27.19%). Enforcing the 10% asset cap, 35% sector cap, and L1 turnover penalty prevents extreme concentration and stabilizes returns.
4. **Sequence Model Limitations:** PyTorch LSTM (E4, Sharpe 0.44) and the XGB-LSTM ensemble (E5, Sharpe 0.41) experienced negative out-of-sample IC (-0.0166), validating the Master Implementation Plan guideline: *"If the models do not provide complementary value, keep the stronger model instead of forcing an ensemble."*

---

## 3. Allocation Explainability & Attribution Summary (E3 Portfolio)

Stock allocations for the active strategy (E3) are decomposed into mathematically grounded attribution metrics:

| Stock | Sector | Target Weight | Standalone Vol | Marginal Risk Contrib (MRC) | % Risk Contrib (PRC) | Diversification Discount | Top Predictive Factor Driver | Synthesized Rationale |
|---|---|---|---|---|---|---|---|---|
| **NESTLEIND.NS** | FMCG | 10.00% | 22.87% | 0.0989 | 9.76% | +56.73% | `f_vol_21` (low vol) | Capped at max single-stock limit (10.0%); supported by favorable low realized volatility; high diversification discount (56.7%). |
| **SUNPHARMA.NS** | Pharma | 10.00% | 19.67% | 0.1019 | 10.05% | +48.21% | `f_vol_21` (low vol) | Capped at max single-stock limit (10.0%); supported by favorable low realized volatility; high diversification discount (48.2%). |
| **ONGC.NS** | Energy | 10.00% | 24.17% | 0.0778 | 7.68% | +67.82% | `f_mkt_corr_63` | Capped at max single-stock limit (10.0%); high diversification discount (67.8%). |
| **ICICIBANK.NS** | Banking | 8.73% | 21.10% | 0.1264 | 10.87% | +40.09% | `f_macd_sig`, `f_dist_52wh` | Strong positive allocation (8.7%); supported by positive MACD signal momentum, near 52-week high breakout zone. |
| **COALINDIA.NS** | Energy | 8.54% | 24.11% | 0.0792 | 6.64% | +67.14% | `f_mkt_corr_63` | Strong positive allocation (8.5%); high diversification discount (67.1%). |
| **NTPC.NS** | Energy | 7.98% | 19.82% | 0.0934 | 7.34% | +52.88% | `f_vol_21` | Strong positive allocation (8.0%); bounded by Energy sector cap constraint; high diversification discount (52.9%). |
| **DRREDDY.NS** | Pharma | 6.29% | 24.21% | 0.1032 | 6.39% | +57.38% | `f_macd_sig` | Strong positive allocation (6.3%); supported by positive MACD signal momentum; high diversification discount (57.4%). |

---

## 4. Test Suite & Quality Verification

Full test suite execution confirms complete modular coverage:

```
tests/test_adaptive.py: 2 passed
tests/test_backtest.py: 5 passed
tests/test_core.py: 6 passed
tests/test_data.py: 8 passed
tests/test_e2e.py: 1 passed
tests/test_experiments.py: 4 passed
tests/test_explainability.py: 2 passed
tests/test_features.py: 4 passed
tests/test_models.py: 5 passed
tests/test_optimizer.py: 6 passed
============================== 43 passed in 40.15s ==============================
```

### Verification Checklist
- [x] **Investor UI Preserved 100%:** `frontend/baseline_diff.txt` shows exactly 3 lines changed (data feed injection only).
- [x] **Developer UI Enhanced:** `frontend/dashboard.html` enriched with live experiment ladder, bootstrap significance, allocation explainability, and model governance tabs.
- [x] **No Synthetic Data:** Real Yahoo Finance NIFTY 50 prices, genuine NIFTY 50 TRI, and actual India VIX time series.
- [x] **Strict Leakage Prevention:** Purged walk-forward validation; 3,402 feature checks verified with 0 discrepancies.
- [x] **Realistic Friction Schedule:** Next-day $T+1$ execution timing, intra-month daily weight drift, and 20 bps statutory Indian transaction costs.
- [x] **Statistical Validation:** Politis & Romano stationary block bootstrap confidence intervals and p-values.
- [x] **Model Governance:** Real-time feedback logger and Champion-Challenger evaluator with statistical guardrails.
