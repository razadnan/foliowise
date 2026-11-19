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
| **Features** | Past returns (1/5/21/63d), rolling vol (21/63d), momentum (126-21d), RSI-14, volume change, 52-week high distance, relative-to-NIFTY return |
| **Target** | 21-day forward return ($P_{t+21}/P_t - 1$) with 21-day purge window |
| **Models** | Ridge Regression (L2), Random Forest, XGBoost – expanding walk-forward CV |
| **Strategies** | S0 NIFTY benchmark · S1 Equal-weight · S2 Min-variance · S3 Max-Sharpe (hist) · S4 Max-Sharpe (ML) · S5 Top-5 ML equal-weight · S6 HRP |
| **Backtest** | Monthly rebalance, 0.20% per-unit transaction cost, Ledoit-Wolf covariance, 15% concentration cap |
| **Frontend** | FolioWise single-file HTML page with real model outputs injected via `src/build_frontend.py` |

---

## Folder Structure

```
Minor_Project/
├── data/
│   ├── raw/
│   │   ├── prices.csv        # Long format: date, ticker, adj_close, volume, filled
│   │   ├── index.csv         # Benchmark: date, adj_close (^NSEI)
│   │   └── universe.csv      # Universe metadata: ticker, name, sector
│   └── processed/
│       ├── returns.csv       # Daily returns: date, ticker, ret_1d
│       └── features.csv      # 11 features + target_21d (72,090 rows)
├── notebooks/
│   ├── 01_data.ipynb         # Data engineering & quality audit
│   ├── 02_features.ipynb     # Feature engineering & leakage tests
│   ├── 03_models.ipynb       # Walk-forward ML training & IC evaluation
│   ├── 04_backtest.ipynb     # Portfolio optimization & daily backtesting
│   └── 05_analysis.ipynb     # Sub-period analysis, bootstrap & frontend export
├── src/
│   ├── stage1_data.py        # Automated stage 1 data acquisition & cleaning
│   ├── stage2_features.py    # Automated stage 2 feature engineering
│   ├── stage3_models.py      # Automated stage 3 walk-forward ML models
│   ├── stage4_backtest.py    # Automated stage 4 portfolio optimization & backtest
│   ├── stage5_analysis.py    # Automated stage 5 analysis & frontend generation
│   ├── features.py           # Leak-free feature computation functions
│   ├── backtest.py           # Core backtesting utilities & optimizers
│   ├── metrics.py            # Portfolio & model metric calculations
│   └── build_frontend.py     # Deterministic frontend HTML builder
├── results/                  # All code-generated tables (.csv), configs, and charts (.png)
├── report/
│   ├── final_report.md       # Full 25-35 page academic research report
│   ├── presentation_slides.md# 18 defense presentation slides
│   └── viva_preparation.md  # Comprehensive Q&A pack for oral defense
├── logs/
│   ├── defects.md            # Defect tracking log
│   ├── changelog.md          # Formal engineering changelog
│   └── handoff_s1.md … s4.md # Stage handoff sign-off records
├── frontend/
│   ├── FolioWise_baseline.html# Untouched baseline HTML template
│   ├── index.html            # Production frontend with real model data
│   └── baseline_diff.txt     # Unified diff showing exactly 3 lines changed
├── tests/
│   └── test_core.py          # Pytest unit tests for turnover, drift, and costs
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
- **`src/stage2_features.py`:** Computes 11 backward-looking features and 21-day forward target, executes a 6-date truncation leakage test (100% exact match), and exports `data/processed/features.csv`.
- **`src/stage3_models.py`:** Executes expanding-window walk-forward validation (2020–2026) with 21-day target purge, trains Ridge, Random Forest, and XGBoost, and exports `results/predictions.csv` and `results/model_metrics.csv`.
- **`src/stage4_backtest.py`:** Computes Ledoit-Wolf covariance, runs SLSQP 15% capped optimization, HRP, and simulating daily returns with weight drift and 20 bps turnover cost. Exports `results/weights.csv`, `results/daily_returns.csv`, and `results/strategy_metrics.csv`.
- **`src/stage5_analysis.py`:** Performs sub-period breakdown, runs stationary bootstrap on Sharpe differences, exports `results/frontend.json`, and triggers `src/build_frontend.py`.

### Frontend Builder
- **`src/build_frontend.py`:** Injects real model data into `frontend/FolioWise_baseline.html` to generate `frontend/index.html` and saves `frontend/baseline_diff.txt`. Strictly adheres to SRS §3.3 (only 3 lines modified).

---

## Strategies

| ID | Strategy Name | Description | Cap | Cost |
|:---|:---|:---|:---:|:---:|
| **S0** | NIFTY 50 Benchmark | Buy and hold `^NSEI` index | N/A | 0 bps |
| **S1** | Equal Weight ($1/N$) | Equal weighting (3.70% per stock), rebalanced monthly | 3.7% | 20 bps |
| **S2** | Minimum Variance | Ledoit-Wolf covariance optimization, no return forecasts | 15% | 20 bps |
| **S3** | Max Sharpe (Historical) | Markowitz optimization using 252-day trailing return mean | 15% | 20 bps |
| **S4** | Max Sharpe (ML Ridge) | Markowitz optimization using 50/50 blend of ML forecast and history | 15% | 20 bps |
| **S5** | Top-5 ML Equal Weight | Equal weight in top 5 stocks ranked by Ridge forecast | 20% | 20 bps |
| **S6** | Hierarchical Risk Parity | Tree clustering & recursive bisection (López de Prado) | 15% | 20 bps |

---

## Frontend Integration

FolioWise is delivered as a standalone single-file web application (`frontend/index.html`).

- **Verification Diff:** Check `frontend/baseline_diff.txt` to confirm that only 3 lines changed against `FolioWise_baseline.html`:
  - FAQ company count (Line 173: "13 large companies" $\to$ "27 large companies")
  - Stock data array (Line 179: `const S=[...]`)
  - Market constants (Line 180: `const RHO=0.2200, NIFTY={m:0.1137, v:0.1774}`)
- **Known Differences:** See SRS §3.3 and `report/final_report.md` Section 6.2 for details on scoring rule heuristics, 20% default UI cap, and educational demonstration scope.

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
