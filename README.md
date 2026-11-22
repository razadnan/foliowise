# FolioWise: Adaptive Machine Learning & Quantitative Portfolio Optimization

> **FolioWise (v2.0 SmartFolio Integrated)** is an institutional-grade, explainable quantitative research platform and portfolio optimization system applied to the Indian Equity Market (**NIFTY 50**).
> It bridges advanced statistical machine learning with constrained convex portfolio optimization, walk-forward out-of-sample backtesting, model governance, and transparent attribution.

---

## 1. System Architecture & Methodology

FolioWise is structured into ten modular packages adhering to strict financial engineering standards:

```
Minor_Project/
├── data/
│   ├── raw/
│   │   ├── prices.csv                 # 27 NIFTY 50 large caps (2015-2026, 78,867 rows, forward-fill <= 3)
│   │   ├── index.csv                  # NIFTY 50 Price Return Index (^NSEI, 2,921 rows)
│   │   ├── index_tri.csv              # NIFTY 50 Total Return Index (NIFTYBEES base, 2,921 rows)
│   │   ├── india_vix.csv              # India VIX daily closing level (^INDIAVIX, 2,910 rows)
│   │   ├── universe_pit.csv           # Point-in-Time membership intervals (76 constituent periods)
│   │   ├── membership_changes.csv     # 58 semi-annual reconstitution events
│   │   └── universe.csv               # Ticker, name, and sector metadata
│   ├── processed/
│   │   ├── returns.csv                # Daily asset returns
│   │   └── features.csv               # 21 quantitative factors + standardized rank targets
│   ├── data_quality_report.json       # Automated data integrity & validation report
│   └── foliowise.db                   # SQLite transactional store
├── src/
│   ├── universe/                      # Point-in-time constituent resolver & coverage bounds
│   │   ├── resolver.py
│   │   └── build_universe_history.py
│   ├── data/                          # Data acquisition, cleaning, TRI benchmarking & storage
│   │   ├── market_data.py
│   │   ├── benchmark.py
│   │   ├── storage.py
│   │   └── validator.py
│   ├── features/                      # 21 factors, standardized rank targets, leakage verification
│   │   ├── pipeline.py
│   │   ├── target.py
│   │   └── leakage_test.py
│   ├── models/                        # ML models (Ridge, XGBoost, PyTorch LSTM, Ensemble)
│   │   ├── ridge.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   ├── ensemble.py
│   │   └── evaluator.py               # Rank IC, ICIR, Newey-West HAC t-statistics, RMSE
│   ├── risk/                          # Ledoit-Wolf shrinkage covariance & spectral PSD checks
│   │   └── covariance.py
│   ├── optimization/                  # Markowitz SLSQP, Min-Var, HRP, ERC, 10% asset / 35% sector caps
│   │   └── engine.py
│   ├── backtesting/                   # Walk-forward daily simulation (T+1 execution, drift, Indian costs)
│   │   ├── engine.py
│   │   └── cost_model.py              # 20 bps statutory delivery fee schedule (STT, Stamp, SEBI, GST)
│   ├── experiments/                   # SmartFolio Experiment Ladder E0-E6 & stationary bootstrap
│   │   ├── runner.py
│   │   └── bootstrap.py               # Politis & Romano (1994) stationary block bootstrap
│   ├── feedback/                      # Prediction feedback loop & ex-ante vs ex-post portfolio risk
│   │   └── engine.py
│   ├── adaptive/                      # Champion-Challenger governance with promotion/rollback hurdles
│   │   └── evaluator.py
│   ├── explainability/                # Stock attribution: MRC, PRC, diversification, SHAP factors
│   │   └── allocator.py
│   ├── stage1_data.py                 # Pipeline stage runners
│   ├── stage2_features.py
│   ├── stage3_models.py
│   ├── stage4_backtest.py
│   ├── stage5_analysis.py
│   └── build_frontend.py              # Zero-leakage frontend builder
├── results/
│   ├── experiments/
│   │   ├── experiment_ladder.csv      # Institutional metrics for E0 through E6
│   │   ├── bootstrap_significance.csv # 1,000-sample bootstrap p-values and 95% CIs
│   │   ├── daily_returns_e0_e6.csv    # Daily backtest returns for all experiments
│   │   ├── weights_e0_e6.csv          # Rebalance weights history
│   │   └── equity_curves_e0_e6.png    # Cumulative out-of-sample equity growth
│   ├── feedback/
│   │   ├── prediction_feedback.csv    # 15,309 point-in-time prediction vs realization records
│   │   ├── rolling_ic_summary.csv     # 567 rolling IC cycles across models
│   │   └── portfolio_feedback.csv     # Ex-ante forecasted vs realized portfolio volatility
│   ├── adaptive/
│   │   ├── champion_challenger_history.csv # Governance audit trail
│   │   └── model_governance_report.json    # Final model lifecycle report
│   ├── allocation_explanations.csv    # Stock attribution table with SHAP factor drivers
│   ├── allocation_explanations.json   # Machine-readable attribution payload
│   ├── strategy_metrics.csv           # S0–S7 benchmark and strategy metrics
│   ├── predictions.csv                # 15,687 out-of-sample walk-forward predictions
│   ├── model_metrics.csv              # Machine learning IC and error metrics
│   ├── shap_importance.csv            # Permutation TreeSHAP factor importance
│   └── frontend.json                  # Data contract for frontends
├── frontend/
│   ├── FolioWise_baseline.html        # Untouched baseline HTML template
│   ├── index.html                     # 100% Preserved Investor UI (Plain language portfolio tool)
│   ├── dashboard.html                 # 5-Tab Developer Quantitative Research Dashboard
│   └── baseline_diff.txt              # Unified diff proving only 3 permitted data lines changed
├── tests/                             # Full automated test suite (43 tests, 100% passing)
│   ├── test_data.py                   # Data integrity, PIT universe, TRI benchmarks
│   ├── test_features.py               # Factor calculations and 0-leakage verification
│   ├── test_models.py                 # Ridge, XGBoost, PyTorch LSTM, Ensemble, Newey-West
│   ├── test_optimizer.py              # Covariance shrinkage, asset/sector caps, ERC
│   ├── test_backtest.py               # Next-day execution, weight drift, Indian cost schedule
│   ├── test_experiments.py            # Stationary block bootstrap hypothesis testing
│   ├── test_adaptive.py               # Feedback loops and Champion-Challenger evaluator
│   ├── test_explainability.py         # Marginal risk attribution and SHAP rationales
│   ├── test_e2e.py                    # End-to-end integration pipeline
│   └── test_core.py                   # Core invariant regression tests
└── docs/
    ├── EXISTING_SYSTEM_AUDIT.md       # Pre-upgrade system audit
    ├── DATA_SOURCES.md                # Data acquisition, universe selection & survivorship
    ├── DATA_QUALITY_REPORT.md         # Ingestion quality gates & clean price certification
    ├── FEATURE_DOCUMENTATION.md       # 21 factor mathematical definitions
    └── UPGRADE_VERIFICATION_REPORT.md # Comprehensive upgrade verification report
```

---

## 2. SmartFolio Experiment Ladder (E0 – E6)

All experiments evaluated over the 2020–2026 walk-forward test period under strict next-day execution timing and 20 bps statutory round-trip transaction costs:

| ID | Model Architecture | CAGR (%) | Annual Vol (%) | Sharpe Ratio | Sortino | Max DD (%) | Monthly Turnover (%) | vs E0 95% Bootstrap CI | p-value |
|---|---|---|---|---|---|---|---|---|---|
| **E0** | 1/N Equal Weight (Baseline) | **17.11%** | 16.32% | **0.68** | 0.93 | -33.24% | 3.62% | Benchmark | — |
| **E1** | Traditional Markowitz (Sample Cov) | 9.60% | 17.60% | 0.20 | 0.28 | -32.88% | 26.20% | [-0.890, -0.026] | 0.060 |
| **E2** | Markowitz + Ledoit-Wolf Shrinkage | 9.66% | 17.65% | 0.21 | 0.28 | -33.38% | 25.82% | [-0.878, -0.024] | 0.059 |
| **E3** | XGBoost + Markowitz (Caps + Reg) | **15.30%** | 16.12% | **0.58** | 0.81 | **-27.19%** | 30.81% | [-0.406, +0.185] | 0.536 |
| **E4** | PyTorch LSTM Sequence Model | 12.77% | 15.48% | 0.44 | 0.60 | -31.22% | 40.21% | [-0.501, +0.033] | 0.114 |
| **E5** | XGB-LSTM Ensemble + Markowitz | 12.61% | 15.92% | 0.41 | 0.57 | -29.61% | 49.23% | [-0.528, +0.040] | 0.110 |
| **E6** | Adaptive FolioWise (Champion-Challenger) | 11.66% | 15.55% | 0.36 | 0.52 | **-27.19%** | 26.55% | [-0.717, +0.061] | 0.165 |

### Key Empirical Findings
1. **DeMiguel et al. (2009) Estimation Risk Confirmed:** The simple 1/N heuristic (E0) delivers the highest raw Sharpe (0.68) because parameter estimation error in expected returns combined with trading friction degrades active Markowitz optimization.
2. **Machine Learning with Risk Budgeting:** E3 (XGBoost with 10% asset cap, 35% sector cap, and L1 turnover penalty) achieves the lowest maximum drawdown (-27.19%) and outperforms sample Markowitz by +38 bps in Sharpe.
3. **Sequence Model Reality:** PyTorch LSTM sequence model exhibits negative out-of-sample IC (-0.0166) on monthly equity ranks, confirming that deep architectures require multi-market pre-training to outperform regularized tree-based models in cross-sectional equity ranking.

---

## 3. Quick Start & Execution Guide

### Virtual Environment Setup
```powershell
# Create virtual environment with Python 3.10+
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### Reproducing the Complete Research Pipeline
Run the modular stages in sequence:
```powershell
# Stage 1: Ingest & validate market data, NIFTY 50 TRI, and India VIX
python src/stage1_data.py

# Stage 2: Compute 21 factors & execute multi-date leakage test
python src/stage2_features.py

# Stage 3: Train walk-forward ML models (Ridge, XGBoost, LSTM, Ensemble)
python src/stage3_models.py

# Stage 4: Run S0-S7 backtests with next-day execution & 20 bps cost schedule
python src/stage4_backtest.py

# Stage 5: Run regime analysis, ablation study, and compile frontends
python src/stage5_analysis.py

# Execute SmartFolio Experiment Ladder (E0-E6) & Stationary Bootstrap
python src/experiments/runner.py
```

### Running the Full Automated Test Suite
```powershell
.\.venv\Scripts\pytest tests/ -v
```
All **43 unit, integration, and regression tests** will execute and pass in ~40 seconds.

---

## 4. Frontend Access

The project features a two-tier frontend architecture:

1. **Investor View (`frontend/index.html`):**
   - Pure, zero-dependency plain language portfolio builder for end investors.
   - 100% preserved visual design, DOM structure, and user workflow.
   - Verified by `frontend/baseline_diff.txt` (only 3 data lines updated).
   - Open directly: `frontend/index.html`.

2. **Developer Research Dashboard (`frontend/dashboard.html`):**
   - 5-Tab quantitative research dashboard surfacing institutional empirical findings:
     - **Tab 1 (Overview):** S0–S7 performance summary and SmartFolio Experiment Ladder (E0–E6) table with 95% bootstrap confidence intervals.
     - **Tab 2 (Allocation & Explainability):** Stock-level attribution table with Marginal Risk Contribution (MRC), Percentage Risk Contribution (PRC), Diversification Benefits, and natural language rationales.
     - **Tab 3 (Performance vs NIFTY):** Interactive SVG equity growth chart and sub-period macroeconomic regime analysis.
     - **Tab 4 (Costs & Turnover):** Itemized Indian statutory trading cost schedule and cost sensitivity analysis (10, 20, 30 bps).
     - **Tab 5 (Model Comparison & Governance):** Walk-forward ML metrics (Rank IC, Newey-West t-stats, RMSE), 6-date leakage audit, and Champion-Challenger model governance lifecycle.
   - Open directly: `frontend/dashboard.html`.

---

## 5. Academic & Regulatory Disclaimer

*FolioWise is an academic research and engineering minor project developed for quantitative finance education. Historical backtests, simulated performances, and machine learning factor ranks do not guarantee future returns. Equity trading in Indian securities involves substantial risk of capital loss. All transaction costs model Indian statutory delivery rates (STT, Stamp Duty, SEBI turnover fees, NSE charges, GST, and large-cap slippage) under SEBI guidelines.*
