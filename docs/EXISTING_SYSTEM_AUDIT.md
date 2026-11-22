# FolioWise Existing System Audit & Baseline Specification

**Document Version:** 1.0  
**Date:** November 2026  
**Project:** FolioWise – Adaptive, Explainable Machine Learning System for Portfolio Optimization  
**Deliverable:** Milestone M0 Baseline Repository Audit  
**Target Market:** Indian Equity Market (NIFTY 50)  

---

## Executive Summary

This document establishes the mandatory **Milestone M0 System Audit** of the FolioWise application prior to the phased integration of the SmartFolio methodology. 

In strict compliance with the **Three Fundamental Principles** of the Master Implementation Plan:
1. **Preserve the existing FolioWise frontend**, including its visual aesthetics, layout structures, user-facing investor flows, and researcher-facing developer dashboards.
2. **Integrate the SmartFolio methodology** into FolioWise's existing backend and data-processing architecture without introducing unnecessary microservices or multi-agent bloat.
3. **Elevate reliability, statistical validity, allocation explainability, and reproducibility** while maintaining complete honesty regarding empirical findings.

No code modifications or deletions have been executed during this audit. The repository is established on git branch `feature/smartfolio-upgrade`.

---

## 1. Existing System Architecture

### 1.1 High-Level Architecture Overview

FolioWise currently operates as a file-based, 5-stage quantitative pipeline connecting local analytical Python scripts to a modern single-page browser frontend:

```
+---------------------------------------------------------------------------------------------------+
|                                  FOLIO-WISE EXISTING ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+

 [ DATA PIPELINE ]
  Yahoo Finance API (^NSEI & 27 Stocks)
         │
         ▼
  src/stage1_data.py ──────► data/raw/ (prices.csv, index.csv, universe.csv, volumes.csv)
         │
         ▼
  src/stage2_features.py ──► data/processed/ (features.csv, returns.csv)
         │
         ▼
 [ MODEL & BACKTEST ENGINE ]
  src/stage3_models.py ────► results/ (predictions.csv, model_metrics.csv, feature_importance.png)
         │
         ▼
  src/stage4_backtest.py ──► results/ (weights.csv, daily_returns.csv, strategy_metrics.csv)
         │
         ▼
 [ ANALYSIS & EXPORT ]
  src/stage5_analysis.py ──► results/ (frontend.json, subperiod_analysis.csv, bootstrap.json)
         │
         ▼
  src/build_frontend.py ───► Injects real model data into FolioWise HTML templates
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
 [ INVESTOR EXPERIENCE ]           [ DEVELOPER / RESEARCHER EXPERIENCE ]
  frontend/index.html               frontend/dashboard.html
  (Interactive Planner)             (5-Tab Quantitative Research Dashboard)
```

### 1.2 Technology Stack Inventory

| Tier | Current Technology | Status & Assessment |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3 (CSS Variables, Flexbox/Grid, Dark Mode), Vanilla ES6 JavaScript, SVG data visualizations | **Working & Preserved.** Zero external framework dependencies; highly responsive, accessible, tabular numerals. |
| **Data Processing** | Python 3.12, Pandas 2.2.2, NumPy 1.26.4 | **Working & Preserved.** Fast in-memory array operations. |
| **Market Data Ingestion** | `yfinance` 1.8.0 with `curl_cffi` backend | **Working.** Capable of fetching daily OHLCV from NSE tickers (`.NS`). |
| **Machine Learning** | Scikit-Learn 1.5.0, XGBoost 2.0.3 | **Working.** Linear Ridge regression, Random Forest, XGBoost regression. |
| **Portfolio Optimization** | `scipy.optimize` (SLSQP), `sklearn.covariance.LedoitWolf`, custom Hierarchical Tree Clustering | **Working.** Replaced legacy `pypfopt` dependency for robust convergence under Python 3.12. |
| **Explainability** | Custom feature permutation attributions (`src/shap_analysis.py`) | **Partial.** Basic permutation attribution in place; true SHAP tree explainer needs expansion. |
| **Data Storage** | Comma-Separated Values (CSV), JavaScript Object Notation (JSON) | **Working.** File-based storage; lacks database persistence and point-in-time universe tables. |
| **Testing** | `pytest` 9.2.0 (`tests/test_core.py`) | **Working.** 6 unit tests passing (turnover, drift, cost, schema validation). |

---

## 2. Frontend Audit & Preservation Checklist

The FolioWise frontend comprises two distinct, interconnected interfaces located in `frontend/`. Both interfaces must be strictly preserved in visual styling, DOM structure, and interaction behavior.

### 2.1 User/Investor Experience (`frontend/index.html`)

The investor-facing interface is an interactive decision-support application (`FolioWise – Where should I invest?`) designed for everyday Indian retail investors.

```
+----------------------------------------------------------------------------------------+
| HEADER: FolioWise Logo | Nav: [Plan] [How it works] [Questions] [For developers →]     |
+----------------------------------------------------------------------------------------+
| HERO & PLANNER:                                                                        |
|  - Radio Mode: "I'm new to investing" vs. "I already invest" (Holdings tracker)       |
|  - Amount Input: Rupee input with fast chips (₹25k, ₹1L, ₹5L, ₹10L)                    |
|  - Time Horizon: 1, 3, 5, 10 years chips                                               |
|  - Risk Comfort Slider: 1 ("I'd panic") to 5 ("I'd stay calm") with dynamic guidance   |
|  - Optional Advanced Options (Accordion):                                              |
|      * Max single company cap: 10%, 15%, 20%, 30%                                      |
|      * "Don't sell anything" checkbox (new-money rebalancing)                          |
|      * Monthly SIP addition input                                                      |
|      * Sector exclusion toggles (Banking, IT, FMCG, Energy, Pharma, Auto, etc.)       |
|  - Button: "Show my plan"                                                              |
+----------------------------------------------------------------------------------------+
| RESULTS SECTION (Dynamic, appears on submission):                                      |
|  - Plan Summary Headline                                                               |
|  - Two-Column Layout:                                                                  |
|      * SVG Donut Chart + Risk Meter Gauge                                              |
|      * Recommended Allocation Table (Company, Sector, Share %, Rupee Amount)           |
|  - Key Portfolio Facts Cards (Expected Return, Rough Year, Swings, Holdings Count)     |
|  - Glossary Accordion ("What do these numbers mean?")                                  |
|  - Existing Holdings Insight Card (if existing mode chosen)                             |
|  - Projected Wealth Growth Chart (SVG Line with shaded likely range band)              |
|  - Strategy Comparison Cards (Your Mix vs Equal Split vs NIFTY 50)                     |
|  - Honest Backtest Findings Card (Plain-language verdict on 2020-2026 test)            |
|  - Action Buttons: [Copy my plan] [Save as PDF] [Change my answers]                    |
+----------------------------------------------------------------------------------------+
| EDUCATIONAL SECTIONS:                                                                  |
|  - "How it works" 3-step timeline                                                      |
|  - Common Questions FAQ Accordion (Financial advice, taxes, rough year, live prices)   |
|  - Footer with legal disclaimers and link: "For developers & quantitative researchers →"|
+----------------------------------------------------------------------------------------+
```

#### Investor Experience Component Inventory:
1. **Header & Navigation:** Branding logo, anchor links, and link to Developer view.
2. **Investor Persona Switcher:** Radio toggle between *New Investor* and *Existing Investor*.
3. **Current Holdings Input (`#holdQ`):** Dynamic input list permitting users to specify existing stock quantities/values to calculate buy/sell/hold delta trades.
4. **Capital Investment Amount (`#amt`):** Clean INR numeric input with pre-formatted currency tags and quick chips (`₹25k`, `₹1L`, `₹5L`, `₹10L`).
5. **Investment Horizon Selector (`#hl`):** 1-year, 3-year, 5-year, and 10-year horizon buttons.
6. **Risk Tolerance Meter (`#risk`):** 5-point discrete slider with real-time reactive descriptive labels ("Very safe", "Careful", "Balanced", "Growth", "Aggressive").
7. **Advanced Constraint Accordion:** Single-stock concentration limits (10% to 30%), SIP recurring deposits, no-selling tax constraints, and sector exclusions.
8. **Allocation Visualization:** Custom SVG Donut chart with tabular legend, percentage slices, and rupee breakdowns.
9. **Growth Range Simulator:** Parametric log-normal fan chart projecting central, optimistic, and 5th-percentile "rough year" trajectories.
10. **Benchmark Comparison Component:** Direct side-by-side comparison between the user's customized allocation, an equal split (1/N), and the NIFTY 50 index.
11. **Academic Honesty Callout:** Transparent disclaimer alerting users that in historical testing, 1/N outperformed active ML optimization due to trading frictions.
12. **Export Actions:** Clipboard plan copying, native print/PDF styling, and questionnaire resetting.

---

### 2.2 Developer/Researcher Experience (`frontend/dashboard.html`)

The researcher-facing interface is a multi-tab quantitative analysis dashboard (`FolioWise Research – Developer Backtest Dashboard`) engineered for deep model exploration, risk decomposition, and walk-forward validation audits.

```
+----------------------------------------------------------------------------------------+
| HEADER: FolioWise Research Badge | Return Link: [← Investor View]                      |
+----------------------------------------------------------------------------------------+
| NAVIGATION TABS:                                                                       |
|  [1. Overview] [2. Allocation] [3. Performance vs NIFTY] [4. Costs & Turnover]         |
|  [5. Model Comparison]                                                                 |
+----------------------------------------------------------------------------------------+
| TAB 1: OVERVIEW                                                                        |
|  - 4 KPI Metric Cards: Out-of-Sample Period, S1 Sharpe, S4 ML Sharpe, S0 NIFTY Sharpe  |
|  - Honest Empirical Finding Callout (Explaining estimation error & turnover friction)  |
|  - Comprehensive 7-Strategy League Table (CAGR, Vol, Sharpe, MaxDD, Calmar, Turnover)  |
+----------------------------------------------------------------------------------------+
| TAB 2: ALLOCATION                                                                      |
|  - Portfolio Concentration Card: Effective N, Herfindahl (HHI), Active Stock Count    |
|  - Universe Sample Table: 27 Stocks with Sectors, Expected Returns (m), Volatilities(v)|
+----------------------------------------------------------------------------------------+
| TAB 3: PERFORMANCE VS NIFTY                                                            |
|  - Interactive SVG Cumulative Equity Growth Chart (2020–2026, S0, S1, S4, S6)          |
|  - Sub-period Market Regime Analysis Table (COVID, Post-COVID Bull, Structural Regime)  |
+----------------------------------------------------------------------------------------+
| TAB 4: COSTS & TURNOVER                                                                |
|  - Transaction Cost Sensitivity Grid: Net CAGR and Sharpe at 10 bps, 20 bps, 30 bps     |
|  - Turnover & Friction Table: Average Monthly Turnover %, Annual Cost Drag (bps)       |
+----------------------------------------------------------------------------------------+
| TAB 5: MODEL COMPARISON                                                                |
|  - Walk-Forward ML Return Prediction Table: Ridge, GB, RF, MLP, Momentum, Naive        |
|    (Mean IC, IC t-stat with Newey-West, Directional Acc %, RMSE, Top-Bottom Spread)    |
|  - Data Integrity & Leakage Verification Audit Table across historical cutoffs         |
|  - Phase Roadmap Callout: Next milestones in progress                                 |
+----------------------------------------------------------------------------------------+
| FOOTER: Research metadata and bidirectional navigation                                 |
+----------------------------------------------------------------------------------------+
```

---

### 2.3 Frontend Preservation Checklist

| Interface Section | File / DOM Selector | Existing State | Preservation Requirement |
| :--- | :--- | :---: | :--- |
| **Investor Hero & Typography** | `frontend/index.html` `.hero`, `h1`, `.lead` | Working | **Must retain 100% untouched.** |
| **Questionnaire Form & Chips** | `frontend/index.html` `#f`, `.chips`, `#amt` | Working | **Must retain 100% untouched.** |
| **Holdings Input Matrix** | `frontend/index.html` `#holdQ`, `#hrows` | Working | **Must retain 100% untouched.** |
| **Comfort Slider & Help Text** | `frontend/index.html` `#risk`, `#riskHelp` | Working | **Must retain 100% untouched.** |
| **Advanced Options Accordion** | `frontend/index.html` `details.more`, `#cap` | Working | **Must retain 100% untouched.** |
| **SVG Donut & Allocation Table**| `frontend/index.html` `#donut`, `#tb` | Working | **Must retain 100% untouched.** Real data feeds must plug seamlessly into existing mapping. |
| **SVG Growth Fan Chart** | `frontend/index.html` `#chart`, `#legend` | Working | **Must retain 100% untouched.** |
| **Strategy Comparison Cards** | `frontend/index.html` `#cmp`, `#testCmp` | Working | **Must retain 100% untouched.** |
| **Honest Backtest Findings** | `frontend/index.html` `#testingCard` | Working | **Must retain 100% untouched.** |
| **Action Triggers & Modals** | `frontend/index.html` `#copy`, `#print`, `#again` | Working | **Must retain 100% untouched.** |
| **Research Tab 1: Overview** | `frontend/dashboard.html` `#tab-overview` | Working | **Must retain 100% untouched.** |
| **Research Tab 2: Allocation** | `frontend/dashboard.html` `#tab-allocation` | Working | **Must retain 100% untouched.** |
| **Research Tab 3: Performance** | `frontend/dashboard.html` `#tab-performance`| Working | **Must retain 100% untouched.** |
| **Research Tab 4: Costs** | `frontend/dashboard.html` `#tab-costs` | Working | **Must retain 100% untouched.** |
| **Research Tab 5: Models** | `frontend/dashboard.html` `#tab-models` | Working | **Must retain 100% untouched.** |
| **Cross-Navigation Links** | `#btn-nav`, `.footer-dev` | Working | **Must retain bidirectional links between views.** |

---

## 3. Backend & Data Pipeline Audit

### 3.1 Script & Module Inventory

| Module Path | Primary Purpose | Existing Implementation Details | Quality / Status |
| :--- | :--- | :--- | :---: |
| **`src/stage1_data.py`** | Market Data Ingestion & Cleaning | Downloads 27 stocks and `^NSEI` via `yfinance`; calendar alignment; forward fill limit=3; writes `filled` flag; exports `data/raw/prices.csv`, `index.csv`, `universe.csv`, `data/processed/returns.csv`. | **Working.** Clean long-format schema. Lacks historical point-in-time universe tracking and automated TRI download. |
| **`src/stage2_features.py`** | Feature Engineering & Leakage Test | Generates 11 features + 2 market features (`f_mkt_vol_21`, `f_mkt_corr_63`) and 21-day forward target; runs 6-date truncation leakage test; exports `data/processed/features.csv`. | **Working.** 100% leakage test pass. Needs standardization to 20-day excess-return rank target and additional indicators (MACD, ROC, SMA/EMA). |
| **`src/stage3_models.py`** | Walk-Forward Machine Learning | Expanding window walk-forward validation (2020–2026) with 21-day target purge; fits Ridge, Random Forest, XGBoost; evaluates IC, RMSE, Directional Accuracy; exports `results/predictions.csv`, `model_metrics.csv`. | **Working.** Solid walk-forward foundation. Needs LSTM challenger model, XGBoost-LSTM ensemble, ICIR, and Newey-West standard errors. |
| **`src/stage4_backtest.py`** | Portfolio Optimization & Simulation | Ledoit-Wolf shrinkage covariance; SLSQP optimizer with 15% concentration cap; native HRP algorithm; simulates daily returns with weight drift and 20 bps transaction friction; exports `results/weights.csv`, `daily_returns.csv`, `strategy_metrics.csv`. | **Working.** Pure `scipy` SLSQP replaces unstable external packages. Needs itemized Indian statutory cost schedule (STT, Stamp Duty, GST) and risk budgeting. |
| **`src/stage5_analysis.py`** | Analysis & Export | Computes sub-period market regimes, cost sensitivity, stationary block bootstrap on Sharpe difference; exports `results/frontend.json`; invokes `src/build_frontend.py`. | **Working.** Feeds `frontend/index.html`. Needs champion-challenger promotion evaluation and allocation explainability exports. |
| **`src/build_frontend.py`** | Deterministic HTML Compiler | Replaces `const S`, `const NIFTY`, `const RHO`, and company count in `FolioWise_baseline.html` to produce `frontend/index.html`; generates unified diff `frontend/baseline_diff.txt`. | **Working.** Verified: exactly 3 lines changed against baseline. |
| **`src/features.py`** | Core Feature Calculation Library | Modular functions for returns, volatility, momentum, RSI, distance to 52-week high, truncation test. | **Working & Reusable.** Well-tested library. |
| **`src/backtest.py`** | Legacy Backtesting Engine | Earlier prototype importing `pypfopt`. | **Deprecated.** Replaced by `src/stage4_backtest.py`. Kept for backward compatibility. |
| **`src/metrics.py`** | Quantitative Metrics Library | Standard formulas for CAGR, annual volatility, Sharpe ratio, Sortino, Calmar, Max Drawdown, Information Coefficient. | **Working & Reusable.** Solid mathematical utility. |
| **`src/shap_analysis.py`** | Attribution & Feature Importance | Computes feature permutation importance and exports summary plots. | **Working.** Needs upgrading to formal TreeSHAP and allocation-level attribution. |

---

## 4. Implementation Inventory & Gap Analysis

| Component | Current Implementation | Status | Required SmartFolio Action |
| :--- | :--- | :---: | :--- |
| **Frontend: Investor View** | `frontend/index.html` (single-file interactive tool) | **Working** | **Preserve 100%.** Connect data ingestion dynamically to upgraded backend metrics. |
| **Frontend: Research View** | `frontend/dashboard.html` (5-tab quantitative suite) | **Working** | **Preserve 100%.** Connect tables and charts to live E0–E6 experiment results and feedback logs. |
| **Market Data Ingestion** | Daily OHLCV from Yahoo Finance (2015–2026) | **Working** | **Improve.** Retain Yahoo Finance as secondary source; evaluate NSE direct/official sources; incorporate India VIX and NIFTY 50 TRI. |
| **Universe Resolution** | Fixed 27-stock survivor universe | **Partial** | **Improve.** Build point-in-time universe resolver tracking historical additions, deletions, and entry/exit dates; document survivor bias bounds. |
| **Benchmark Integration** | `^NSEI` (Price Index) in `index.csv`; experimental TRI in `index_tri.csv` | **Partial** | **Improve.** Formally separate Price Index from Total Return Index (TRI); eliminate synthetic proxies; clearly document dividend reinvestment impact. |
| **Feature Engineering** | 11 technical + 2 market factors; 21d forward target | **Working** | **Extend.** Add MACD, EMA/SMA crosses, ROC, India VIX signals; standardize target to 20-trading-day cross-sectional excess-return rank. |
| **Leakage Controls** | 6-date truncation test (100% exact match) + 21d purge | **Working** | **Preserve & Expand.** Retain multi-date truncation checks; formalize purging and embargo windows across expanding windows. |
| **ML Predictive Models** | Ridge, Random Forest, XGBoost | **Working** | **Extend.** Retain Ridge and XGBoost; implement LSTM recurrent challenger model; evaluate XGBoost-LSTM ensemble; compute ICIR and Newey-West t-stats. |
| **Risk Estimation** | Ledoit-Wolf shrinkage covariance on trailing 252 days | **Working** | **Preserve & Improve.** Validate positive semi-definiteness; evaluate lookback window sensitivity; compare against sample covariance. |
| **Portfolio Optimization** | SLSQP Markowitz (15% cap), Min-Variance, HRP | **Working** | **Extend.** Retain native SLSQP and HRP; add Risk Budgeting; implement explicit turnover penalty ($\lambda \|\Delta w\|_1$); enforce configurable sector caps (35%). |
| **Allocation Explainability**| Global feature importances (`feature_importance.png`) | **Partial** | **Implement New.** Implement stock-level allocation explanations (predicted return, risk contribution, diversification benefit, sector cap constraints). |
| **Backtesting Engine** | Monthly rebalance with drift, one-way turnover, 20 bps cost | **Working** | **Correct & Extend.** Formally enforce and test next-day execution timing; implement itemized Indian statutory transaction cost model (STT, stamp duty, SEBI, GST). |
| **Experiment Framework** | Hardcoded strategy ladder S0–S7 | **Working** | **Standardize.** Formalize into SmartFolio experiment ladder E0–E6 with immutable runs, versioning, and block bootstrap significance testing. |
| **Adaptive Learning** | None currently implemented | **Missing** | **Implement New.** Build prediction feedback logger, portfolio feedback logger, champion-challenger evaluation engine, promotion guardrails, and rollback support. |
| **Backend API** | Direct Python script execution & file exports | **Partial** | **Extend.** Build modular Python backend services (FastAPI or structured CLI) exposing Data, Model, Portfolio, Backtest, and Adaptive Learning endpoints. |
| **Database & Persistence** | Flat CSV and JSON files in `data/` and `results/` | **Working** | **Extend.** Introduce DuckDB/SQLite storage for experiment tracking, feedback records, and audit logs while preserving CSV exports. |
| **Automated Testing** | `tests/test_core.py` (6 unit tests) | **Working** | **Expand.** Build dedicated test suites across data validation, features, ML models, optimizers, backtesting execution, and champion-challenger logic. |

---

## 5. Existing Bugs & Known Limitations

### 5.1 Identified Bugs & Remediation Status
1. **PyPortfolioOpt Binary Incompatibility:**
   - *Status:* **Resolved.** Deprecated `src/backtest.py` replaced by pure `scipy.optimize` SLSQP and native HRP in `src/stage4_backtest.py`.
2. **Schema Inconsistency in Jupyter Notebooks:**
   - *Status:* **Resolved.** Notebooks `01_data.ipynb` and `03_models.ipynb` were refactored to call `src` stage modules directly, preventing wide/long schema divergence.
3. **Execution Lookahead in Backtest Timing:**
   - *Status:* **Needs Formal Regression Testing.** While `stage4_backtest.py` contains next-day execution logic, execution timing must be strictly locked with dedicated unit tests verifying that orders generated at close $T$ execute at $T+1$ close without capturing $T \to T+1$ returns.
4. **Hardcoded Metrics in Research Dashboard:**
   - *Status:* **To Be Connected.** `frontend/dashboard.html` currently includes hardcoded strategy tables in JavaScript. While accurate to the backtest, it should dynamically parse `results/strategy_metrics.csv` and `results/frontend.json`.

### 5.2 Methodological Limitations to Address
- **Survivorship Bias:** The universe is restricted to 27 surviving NIFTY 50 constituents with continuous trading history from 2015 to 2026. Companies that defaulted, merged, or were removed are absent.
- **Fixed Transaction Fee:** A flat 20 bps fee does not capture non-linear liquidity impact or statutory Indian stamp duty/STT asymmetry.
- **Lack of Sequential Deep Learning:** The system lacks an LSTM or sequence model to test whether temporal patterns add predictive value over tree ensembles.
- **Static Model Promotion:** Models are fit historically and evaluated statically; no automated champion-challenger feedback loop exists to monitor model degradation or govern retraining.

---

## 6. Proposed Integration Architecture

The target architecture seamlessly integrates SmartFolio's advanced capabilities into the existing FolioWise codebase:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                  UPGRADED TARGET ARCHITECTURE                                     |
+───────────────────────────────────────────────────────────────────────────────────────────────────+

                                    [ DATA & UNIVERSE TIER ]
   NSE Historical Archive / Yahoo Finance         NIFTY 50 Historical Membership Records
                     │                                              │
                     ▼                                              ▼
          src/data/market_data.py                      src/universe/resolver.py
          (Validation & Cleaning)                      (Point-in-Time Resolution)
                     │                                              │
                     └──────────────────────┬───────────────────────┘
                                            ▼
                               [ FEATURE ENGINEERING TIER ]
                                  src/features/pipeline.py
                               (14 Features + India VIX)
                                            │
                                            ▼
                                 src/features/target.py
                         (20d Cross-Sectional Excess Rank)
                                            │
                                            ▼
                                src/features/leakage_test.py
                         (Multi-Date Truncation Verification)
                                            │
                                            ▼
                             [ MACHINE LEARNING & MODELS ]
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
           src/models/ridge.py    src/models/xgboost.py   src/models/lstm.py
           (Linear Baseline)      (Nonlinear Primary)     (Sequence Challenger)
                     │                      │                      │
                     └──────────────────────┼──────────────────────┘
                                            ▼
                                 src/models/ensemble.py
                             (Validation-Tuned Ensemble)
                                            │
                                            ▼
                          [ RISK ESTIMATION & OPTIMIZATION ]
                                  src/risk/covariance.py
                               (Ledoit-Wolf Covariance)
                                            │
                                            ▼
                              src/optimization/engine.py
                     (Constrained SLSQP, HRP, MinVar, Risk Budget)
                     (10% Asset Cap, 35% Sector Cap, Turnover Penalty)
                                            │
                                            ▼
                             src/explainability/allocator.py
                       (Rule-Based & SHAP Factor Explanations)
                                            │
                                            ▼
                                  [ BACKTESTING ENGINE ]
                                src/backtesting/engine.py
                       (Walk-Forward, Next-Day Execution, Indian
                        Itemized Costs, Drift Tracking, E0-E6 Ladder)
                                            │
                                            ▼
                           [ ADAPTIVE LEARNING & FEEDBACK ]
                                 src/feedback/engine.py
                       (Prediction Feedback, Portfolio Feedback)
                                            │
                                            ▼
                              src/adaptive/evaluator.py
                       (Champion-Challenger Promotion Guardrails,
                        Rollback Audit, Retraining Scheduler)
                                            │
                                            ▼
                             [ APPLICATION & PRESENTATION ]
                                   src/api/service.py
                      (Clean Service Layer exposing Data, Models,
                       Portfolio, Backtesting, & Adaptive Metrics)
                                            │
                                            ▼
                                src/build_frontend.py
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          frontend/index.html                           frontend/dashboard.html
          (PRESERVED Investor View)                     (PRESERVED Developer View)
          - Real Model Data Injected                    - Live Experiment E0-E6 Metrics
          - Rule-based Explanations                     - Feedback & Retraining Audit
          - Dynamic Risk/Growth Cones                   - Interactive SVG Equity Curves
```

---

## 7. Phased Implementation Roadmap (Milestones M1–M11)

Following the sequence mandated by the Master Implementation Plan:

| Milestone | Phase | Primary Deliverables | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **M1** | Data Engineering | `src/data/market_data.py`, `src/universe/resolver.py`, `data/raw/` | Point-in-time universe mapped; NIFTY 50 TRI integrated; data validation audit tests passing. |
| **M2** | Feature Engineering | `src/features/pipeline.py`, `src/features/target.py`, `tests/test_features.py` | Standardized 20-day excess-return rank target; 14 features; 6-date truncation test 100% passing. |
| **M3** | ML Integration | `src/models/ridge.py`, `xgboost.py`, `lstm.py`, `ensemble.py` | Ridge, XGBoost, and PyTorch LSTM models trained out-of-sample; Rank IC, ICIR, Newey-West t-stats. |
| **M4** | Risk & Optimization | `src/risk/covariance.py`, `src/optimization/engine.py` | Ledoit-Wolf shrinkage verified; 10% stock cap, 35% sector cap, turnover regularization; HRP & Risk Budgeting. |
| **M5** | Backtesting Engine | `src/backtesting/engine.py`, `tests/test_backtest.py` | Next-day execution timing verified; itemized Indian cost schedule; complete metric suite (CAGR, Sharpe, Sortino, Calmar). |
| **M6** | Experiment Ladder | `src/experiments/ladder.py`, `results/experiments/` | Reproducible E0–E6 experiment ladder; stationary block bootstrap significance testing. |
| **M7** | Adaptive Learning | `src/feedback/engine.py`, `src/adaptive/evaluator.py` | Prediction & portfolio feedback logging; Champion-Challenger evaluation with statistical promotion guards; rollback support. |
| **M8** | Explainability | `src/explainability/allocator.py` | Allocation-level factor explanations (expected return, risk contribution, diversification, constraints). |
| **M9** | Frontend Integration | `src/build_frontend.py`, `frontend/index.html`, `dashboard.html` | Both frontend interfaces preserved 100%; real model data, feedback history, and explanations connected seamlessly. |
| **M10**| Quality Assurance | `tests/`, end-to-end regression test suite | All unit, integration, and regression tests passing; complete workflow demonstrated from data ingestion to UI display. |
| **M11**| Final Delivery | Technical documentation, reproducible final reports | Final verification report, updated README, immutable run artifacts, clean git tags. |

---

## 8. Audit Verification & Baseline Sign-Off

- [x] Complete repository file tree inspected and verified.
- [x] Both user-facing (`index.html`) and researcher-facing (`dashboard.html`) frontend interfaces documented.
- [x] Frontend preservation checklist created; zero UI redesigns or overwrites scheduled.
- [x] Existing working features, partial components, and deprecated dependencies cataloged.
- [x] Initial git branch `feature/smartfolio-upgrade` created from clean `main`.
- [x] Existing unit test suite verified green (`6 passed in 7.56s`).
- [x] Proposed integration architecture and phased milestone roadmap established.
