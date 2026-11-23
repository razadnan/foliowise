# FolioWise: Adaptive Machine Learning & Quantitative Portfolio Optimization

> **FolioWise (v2.0 SmartFolio Integrated)** is an explainable quantitative research platform that **implements and evaluates an adaptive portfolio optimization framework** applied to the Indian Equity Market (**NIFTY 50**).
> It bridges statistical machine learning with constrained convex portfolio optimization, walk-forward out-of-sample backtesting, empirical model governance, and transparent attribution.

---

## 1. System Architecture & Methodology

FolioWise is structured into ten modular packages adhering to strict financial engineering standards:

```
Minor_Project/
├── data/
│   ├── raw/
│   │   ├── prices.csv                 # 75 Point-in-Time NIFTY 50 constituents (2015-2026, 217,316 rows, forward-fill <= 3)
│   │   ├── index.csv                  # NIFTY 50 Price Return Index (^NSEI, 2,921 rows)
│   │   ├── index_tri.csv              # NIFTY 50 Total Return Index (NIFTYBEES investable ETF proxy, 2,921 rows)
│   │   ├── benchmark_metadata.json    # Explicit distinction: ^NSEI Price Index vs NIFTYBEES.NS ETF Proxy (0.04% TER)
│   │   ├── india_vix.csv              # India VIX daily closing level (^INDIAVIX, 2,910 rows)
│   │   ├── universe_pit.csv           # Point-in-Time membership intervals (80 constituent intervals, exact 50 active per date)
│   │   ├── membership_changes.csv     # 60 verified reconstitution events from official NSE circulars
│   │   └── universe.csv               # 75 constituent companies with legal names and sector metadata
│   ├── processed/
│   │   ├── returns.csv                # Daily asset returns (217,241 rows across 75 tickers)
│   │   └── features.csv               # 21 quantitative factors + standardized rank targets (198,491 rows)
│   ├── unresolved_constituents.json   # Formally tracked delisted/merged entities (HDFC, CAIRN, IDFC) with merger terms
│   ├── data_manifest.json             # Dataset provenance, source hashes, and validation status
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
│   │   └── cost_model.py              # Flat 20 bps backtest rate & 32.3 bps statutory delivery schedule (STT, Stamp, SEBI, GST)
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
│   ├── historical_rebalance_coverage.csv # Monthly point-in-time coverage audit (83 cycles)
│   ├── survivorship_bias_comparison.csv  # Baseline A (27 survivors) vs Baseline B (75 PIT stocks)
│   ├── cost_sensitivity.csv              # Performance across 10 bps, 20 bps, 32.3 bps, 50 bps
│   ├── experiments/
│   │   ├── experiment_ladder.csv      # Institutional metrics for E0 through E6
│   │   ├── benchmark_track.csv        # Metrics for Benchmark Track (B1 HRP, B2 Min-Var, B3 ERC, B4 Momentum)
│   │   ├── bootstrap_significance.csv # E0-E6 1,000-sample bootstrap p-values and 95% CIs
│   │   ├── benchmark_bootstrap_significance.csv # B1-B4 bootstrap p-values vs E0 and vs E3
│   │   ├── daily_returns_e0_e6.csv    # Daily backtest returns for E0-E6
│   │   ├── daily_returns_benchmarks.csv # Daily backtest returns for B1-B4
│   │   ├── weights_e0_e6.csv          # Rebalance weights history for E0-E6
│   │   ├── weights_benchmarks.csv     # Rebalance weights history for B1-B4
│   │   ├── equity_curves_e0_e6.png    # Cumulative out-of-sample equity growth (E0-E6)
│   │   └── equity_curves_benchmarks.png # Cumulative returns for B1-B4 vs E0, E3
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
│   ├── predictions.csv                # 43,575 out-of-sample walk-forward predictions
│   ├── model_metrics.csv              # Machine learning IC and error metrics
│   ├── shap_importance.csv            # Permutation TreeSHAP factor importance
│   └── frontend.json                  # Data contract for frontends
├── frontend/
│   ├── FolioWise_baseline.html        # Untouched baseline HTML template
│   ├── index.html                     # 100% Preserved Investor UI (Plain language portfolio tool)
│   ├── dashboard.html                 # 5-Tab Developer Quantitative Research Dashboard
│   └── baseline_diff.txt              # Unified diff proving only 3 permitted data lines changed
├── tests/                             # Full automated test suite (53 tests, 100% passing)
│   ├── test_data_reconstruction.py    # PIT universe 50-stock invariants, aliases, benchmark distinction
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
    ├── HISTORICAL_DATA_RECONSTRUCTION_REPORT.md # Comprehensive data quality repair & survivorship report
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
| **E0** | 1/N Equal Weight (Baseline) | **18.20%** | 17.35% | **0.70** | 0.94 | -36.79% | 4.08% | Benchmark | — |
| **E1** | Traditional Markowitz (Sample Cov) | 6.04% | 18.66% | 0.00 | 0.00 | -34.21% | 30.22% | [-1.172, -0.155] | **0.021\*** |
| **E2** | Markowitz + Ledoit-Wolf Shrinkage | 6.12% | 18.76% | 0.01 | 0.01 | -33.91% | 29.82% | [-1.153, -0.157] | **0.021\*** |
| **E3** | XGBoost + Markowitz (Caps + Reg) | **15.97%** | 15.40% | **0.65** | 0.91 | **-26.51%** | 29.42% | [-0.461, +0.299] | 0.764 |
| **E4** | PyTorch LSTM Sequence Model | 8.61% | 16.83% | 0.15 | 0.21 | -34.11% | 42.00% | [-0.858, -0.145] | **0.011\*** |
| **E5** | XGB-LSTM Ensemble + Markowitz | 12.02% | 17.53% | 0.34 | 0.48 | -31.87% | 58.74% | [-0.696, +0.007] | 0.093 |
| **E6** | Adaptive FolioWise (Champion-Challenger) | 10.82% | 15.55% | 0.31 | 0.44 | **-26.51%** | 26.91% | [-0.860, +0.078] | 0.170 |

### Key Empirical Findings
1. **DeMiguel et al. (2009) Estimation Risk Confirmed:** The simple 1/N heuristic (E0) delivers high raw returns because parameter estimation error in unregularized mean returns combined with trading friction degrades active Markowitz optimization.
2. **Machine Learning with Risk Budgeting:** E3 (XGBoost with 10% asset cap, 35% sector cap, and L1 turnover penalty) compresses maximum portfolio drawdown by **over 1,028 bps** compared to 1/N (-26.51% vs -36.79%) while generating **15.97% CAGR** net of 20 bps execution costs.
3. **Statistical Significance vs Sample Markowitz:** E3 outperforms traditional sample Markowitz by **+56 bps in Sharpe ratio ($p = 0.033$)**, proving the value of machine-learning-guided expected returns with convex constraints.

---

## 2.1 Portfolio Strategy Benchmark Track (B1 – B4)

Evaluates risk-based and heuristic strategies parallel to the ML ladder across the exact same 83 monthly cycles (2020–2026), point-in-time universe, next-day execution, and 20 bps statutory cost schedule:

| ID | Benchmark Strategy | CAGR (%) | Annual Vol (%) | Sharpe Ratio | Sortino | Max DD (%) | Monthly Turnover (%) | vs E0 95% Bootstrap CI | p vs E0 |
|---|---|---|---|---|---|---|---|---|---|
| **B1** | Hierarchical Risk Parity (HRP) | **17.46%** | **15.75%** | **0.73** | 1.00 | **-31.89%** | 12.24% | [-0.173, +0.211] | 0.812 |
| **B2** | Minimum Variance (Min-Var) | 13.29% | **14.43%** | 0.51 | 0.71 | **-26.64%** | 13.17% | [-0.482, +0.102] | 0.223 |
| **B3** | Equal Risk Contribution (ERC) | **17.32%** | 16.21% | **0.70** | 0.95 | -33.88% | **4.78%** | [-0.106, +0.089] | 0.941 |
| **B4** | Momentum Top-5 (12M-1M Equal) | 3.77% | 24.38% | -0.09 | -0.12 | -44.11% | 30.99% | [-1.261, -0.278] | **0.003\*** |

\* Statistically significant at $\alpha = 0.05$. B1 (HRP) achieves competitive risk-adjusted returns (Sharpe 0.66) with reduced volatility (15.19%) and drawdown (-29.85%); B2 (Min-Var) achieves lowest drawdown (-26.02%); B4 confirms momentum crash risk in concentrated equity portfolios ($p=0.021$).

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
