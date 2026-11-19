# ML-Driven Portfolio Optimization for NIFTY 50 Stocks

> **FolioWise** – A college minor project that applies machine learning to build and backtest
> portfolios drawn from the NIFTY 50 universe, then surfaces the results through an
> interactive web frontend.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup](#setup)
4. [Running the Pipeline & Notebooks](#running-the-pipeline--notebooks)
5. [Module & Script Reference](#module--script-reference)
6. [Strategies](#strategies)
7. [Frontend Integration](#frontend-integration)
8. [Testing](#testing)
9. [Logs & Governance](#logs--governance)
10. [Disclaimer](#disclaimer)

---

## Project Overview

| Item | Details |
|------|---------|
| **Universe** | 27 large-cap NIFTY 50 stocks with complete 2015–2026 daily trading history |
| **Data** | Daily OHLCV from Yahoo Finance, 2015-01-02 → 2026-11-13 (2,921 trading days) |
| **Features** | 13 features: past returns (1/5/21/63d), rolling vol (21/63d), momentum (126-21d), RSI-14, volume change, 52-week high distance, relative-to-NIFTY return, 21d market volatility, 63d stock-market correlation |
| **Target** | 21-day forward return ($P_{t+21}/P_t - 1$) with 21-day purge window |
| **Models** | Ridge Regression (L2), Random Forest, Gradient Boosting, MLP Neural Net, Momentum & Naive baselines – expanding walk-forward CV |
| **Strategies** | S0 NIFTY 50 TRI · S1 Equal-weight · S2 Min-variance · S3 Max-Sharpe (hist) · S4 Max-Sharpe (ML blend) · S5 Top-5 ML equal-weight · S6 HRP (Ledoit-Wolf) · S7 Top-5 Momentum equal-weight |
| **Backtest** | Next-Day execution (signals at T close, traded at T+1 close), one-way turnover ($0.5\sum\|\Delta w\|$), itemized Indian cost model (0.20% base rate), Ledoit-Wolf covariance, 15% concentration cap (S5/S7 20%) |
| **Frontends** | Two-view architecture: Investor View (`index.html`) + 5-tab Quantitative Research Dashboard (`dashboard.html`), synchronized from `results/frontend.json` |

---

## Folder Structure

```
Minor_Project/
├── data/
│   ├── raw/
│   │   ├── prices.csv           # Long format: date, ticker, adj_close, volume, filled
│   │   ├── index.csv            # Benchmark: date, adj_close (^NSEI)
│   │   ├── index_tri.csv        # NIFTY 50 Total Return Index with 1.30% dividend yield
│   │   └── universe.csv         # Universe metadata: ticker, name, sector
│   └── processed/
│       ├── returns.csv          # Daily returns: date, ticker, ret_1d
│       └── features.csv         # 13 features + target_21d (72,090 rows)
├── notebooks/
│   ├── 01_data.ipynb            # Data engineering & quality audit
│   ├── 02_features.ipynb        # Feature engineering & leakage tests
│   ├── 03_models.ipynb          # Walk-forward ML training & IC evaluation
│   ├── 04_backtest.ipynb        # Portfolio optimization & daily backtesting
│   └── 05_analysis.ipynb        # Sub-period analysis, bootstrap & frontend export
├── src/
│   ├── stage1_data.py           # Stage 1: data acquisition & cleaning
│   ├── stage2_features.py       # Stage 2: feature engineering (13 factors, leakage test)
│   ├── stage3_models.py         # Stage 3: walk-forward ML (Ridge, RF, GB, MLP, baselines)
│   ├── stage4_backtest.py       # Stage 4: S0–S7 backtest (next-day exec, one-way turnover)
│   ├── stage5_analysis.py       # Stage 5: analysis, bootstrap, regimes, frontend export
│   ├── features.py              # Leak-free feature computation functions
│   ├── backtest.py              # Core backtesting utilities & optimizers
│   ├── metrics.py               # Portfolio & model metric calculations
│   ├── shap_analysis.py         # Permutation Shapley attribution (shap_summary.png)
│   └── build_frontend.py        # Deterministic frontend HTML builder
├── results/
│   ├── strategy_metrics.csv     # S0–S7 CAGR, Sharpe, Sortino, drawdown, turnover
│   ├── advanced_strategy_metrics.csv # IR, tracking error, max DD duration
│   ├── model_metrics.csv        # IC, ICIR, Newey-West t-stat per model
│   ├── model_config.json        # Frozen model hyperparameter registry (seed=42)
│   ├── cost_sensitivity.csv     # Sharpe at 10/20/30 bps per strategy
│   ├── regime_analysis.csv      # Calm/Volatile/Crisis regime performance
│   ├── ablation_study.csv       # Feature group ablation results
│   ├── parameter_sensitivity.csv# Cap & blend sensitivity sweep
│   ├── bootstrap_sharpe_difference.json # Block bootstrap S4 vs S1 Sharpe test
│   ├── extreme_returns_audit.csv# Returns > 25% with explanations
│   ├── leakage_test_report.csv  # 6-date truncation leakage audit (all PASS)
│   ├── fallback_log.csv         # Optimizer SLSQP fallback events
│   ├── shap_importance.csv      # Feature importance from permutation SHAP
│   ├── frontend.json            # Shared data source for both frontends
│   ├── equity_curves.png        # S0–S7 cumulative equity chart
│   ├── drawdown_chart.png       # S0–S7 rolling drawdown chart
│   ├── weights_over_time.png    # Monthly allocation heatmap per strategy
│   ├── efficient_frontier.png   # Efficient frontier with strategy overlays
│   ├── calibration_plot.png     # Predicted vs realized return quintile calibration
│   ├── feature_importance.png   # Feature importance bar chart
│   └── shap_summary.png         # SHAP beeswarm summary plot
├── report/
│   ├── SRS_v2.0.md              # Software Requirements Specification v2.0
│   ├── final_report.md          # Full 25–35 page academic research report
│   ├── presentation_slides.md   # 18 defense presentation slides
│   └── viva_preparation.md      # Comprehensive Q&A pack for oral defense
├── logs/
│   ├── defects.md               # Defect tracking log
│   ├── changelog.md             # Formal engineering changelog
│   └── handoff_s1.md … s4.md   # Stage handoff sign-off records
├── frontend/
│   ├── FolioWise_baseline.html  # Untouched baseline HTML template
│   ├── index.html               # Investor View: plain-language portfolio tool
│   ├── dashboard.html           # Developer Research Dashboard (5 tabs)
│   └── baseline_diff.txt        # Unified diff showing 3 lines changed vs baseline
├── tests/
│   └── test_core.py             # Pytest: 6 unit tests (turnover, drift, schema, costs)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/razadnan/foliowise.git
cd foliowise

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline & Notebooks

You can reproduce the complete project from end to end using the automated stage scripts:

```powershell
python src/stage1_data.py
python src/stage2_features.py
python src/stage3_models.py
python src/stage4_backtest.py
python src/stage5_analysis.py
```

Or open and run the Jupyter notebooks in order:
```powershell
jupyter notebook
```
1. `notebooks/01_data.ipynb`
2. `notebooks/02_features.ipynb`
3. `notebooks/03_models.ipynb`
4. `notebooks/04_backtest.ipynb`
5. `notebooks/05_analysis.ipynb`

---

## Module & Script Reference

### Stage Scripts
- **`src/stage1_data.py`:** Downloads 27 NIFTY 50 stocks and `^NSEI`, aligns to the market calendar, applies max 3-day forward fill, records `filled` flag, and exports `data/raw/prices.csv`, `index.csv`, `universe.csv`, and `data/processed/returns.csv`.
- **`src/stage2_features.py`:** Computes 13 backward-looking features and 21-day forward target, executes a 6-date truncation leakage test (100% exact match, Δ = 0.0000000000), and exports `data/processed/features.csv` (72,090 rows, 16 columns).
- **`src/stage3_models.py`:** Executes expanding-window walk-forward validation (2020–2026, 83 test cycles) with 21-day target purge, trains Ridge, Random Forest, Gradient Boosting, MLP Neural Net, Momentum and Naive baselines (seed 42, in-fold MinMax scaling). Exports `results/predictions.csv`, `results/model_metrics.csv`, `results/model_config.json`, `results/calibration_plot.png`.
- **`src/stage4_backtest.py`:** Next-day execution (signal at T close → trade at T+1 close → returns from T+1→T+2). Runs S0–S7 with Ledoit-Wolf covariance, SLSQP 15% cap, HRP, one-way turnover ($0.5\sum|\Delta w|$), Indian cost function. Exports `results/weights.csv`, `results/daily_returns.csv`, `results/strategy_metrics.csv`, `results/efficient_frontier.png`.
- **`src/stage5_analysis.py`:** Advanced evaluation: Sortino, IR, drawdown duration, 2000-resample stationary block bootstrap (21-day blocks), Calm/Volatile/Crisis regime analysis, feature ablations, parameter sensitivity. Exports `results/advanced_strategy_metrics.csv`, `results/bootstrap_sharpe_difference.json`, `results/regime_analysis.csv`, `results/ablation_study.csv`, `results/frontend.json`.

### Frontend Builder
- **`src/build_frontend.py`:** Reads `results/frontend.json` and injects real model data into `frontend/FolioWise_baseline.html` to generate `frontend/index.html`. Saves `frontend/baseline_diff.txt`. Strictly adheres to SRS §3.3 (only 3 lines modified).
- **`src/shap_analysis.py`:** Computes permutation-based Shapley values on the Gradient Boosting model. Saves `results/shap_summary.png` and `results/shap_importance.csv`. Pure Python/NumPy implementation (no SHAP binary dependency).

---

## Strategies

| ID | Strategy Name | Description | Net CAGR | Sharpe | Cap | Cost |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **S0** | NIFTY 50 TRI Benchmark | NIFTY 50 Total Return Index (dividends reinvested) | 13.27% | 0.41 | N/A | 0 bps |
| **S1** | Equal Weight (1/N) | Equal weighting (3.70% per stock), rebalanced monthly | 17.07% | 0.68 | 3.7% | 20 bps |
| **S2** | Minimum Variance | Ledoit-Wolf shrinkage covariance, no return forecasts, lowest drawdown | 13.21% | 0.50 | 15% | 20 bps |
| **S3** | Max Sharpe (Historical) | Markowitz optimization using 252-day trailing return mean | 9.24% | 0.18 | 15% | 20 bps |
| **S4** | Max Sharpe (ML Ridge) | Markowitz optimization using 50/50 blend of Ridge ML forecast + history | 13.89% | 0.46 | 15% | 20 bps |
| **S5** | Top-5 ML Equal Weight | Equal weight in top 5 stocks ranked by Ridge return forecast | 24.00% | 0.77 | 20% | 20 bps |
| **S6** | Hierarchical Risk Parity | HRP tree clustering & recursive bisection with Ledoit-Wolf covariance | 16.09% | 0.66 | 15% | 20 bps |
| **S7** | Top-5 Momentum Equal Weight | Equal weight in top 5 stocks by 12M−1M price momentum | 6.14% | 0.01 | 20% | 20 bps |

> **Primary Hypothesis Result:** S4 does **not** beat S1 on net Sharpe. Bootstrap test: P(S4 > S1) = 13.3%, mean Sharpe difference = −0.202 (95% CI: [−0.607, +0.145]). Equal-weighting dominates mean-variance ML optimization.

---

## Frontend Integration

FolioWise is delivered as a **two-view web application** synchronized via `results/frontend.json`:

- **Investor View (`frontend/index.html`):** Plain-language portfolio tool. Features a disclaimer banner, inline error messages, "What do these numbers mean?" glossary, honest testing card comparing strategies, and PDF print styles. Scoring uses a heuristic rule (not the SLSQP optimizer) — this is stated explicitly on the page.
- **Developer Research Dashboard (`frontend/dashboard.html`):** 5-tab quantitative research tool showing Overview metrics, Allocation weights, Performance vs NIFTY TRI, Costs & Turnover analysis, and Model Comparison. Cross-linked with the Investor View.
- **Single Source of Truth:** Both frontends read `results/frontend.json` to remain synchronized after every pipeline rerun.
- **Verification Diff:** Check `frontend/baseline_diff.txt` to confirm that only 3 lines changed against `FolioWise_baseline.html`:
  - FAQ company count (Line 173: "13 large companies" → "27 large companies")
  - Stock data array (Line 179: `const S=[...]`)
  - Market constants (Line 180: `const RHO=0.2200, NIFTY={m:0.1137, v:0.1774}`)

---

## Testing

Run unit tests covering portfolio turnover, weight drift, and transaction cost mechanics:

```powershell
pytest tests/
```

---

## Logs & Governance

- **`logs/defects.md`:** Comprehensive record of all bugs discovered, root causes, and resolutions.
- **`logs/changelog.md`:** Formal change log documenting major design and specification adjustments.
- **`logs/handoff_s1.md` through `s4.md`:** Complete handoff verification sign-offs between students.

---

## Disclaimer

*FolioWise is a student academic project developed strictly for educational and research purposes. It does not constitute financial, investment, or legal advice. Past performance does not guarantee future results. Investments in equity markets are subject to market risks, including the loss of principal capital.*
