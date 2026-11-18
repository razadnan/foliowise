# ML-Driven Portfolio Optimization for NIFTY 50 Stocks

> **FolioWise** – A college minor project that applies machine learning to build and backtest
> portfolios drawn from the NIFTY 50 universe, then surfaces the results through an
> interactive web frontend.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Setup](#setup)
4. [Running the Notebooks](#running-the-notebooks)
5. [Module Reference](#module-reference)
6. [Strategies](#strategies)
7. [Frontend](#frontend)
8. [Logs & Governance](#logs--governance)
9. [Disclaimer](#disclaimer)

---

## Project Overview

| Item | Details |
|------|---------|
| **Universe** | ~25 large-cap NIFTY 50 stocks |
| **Data** | Daily OHLCV from Yahoo Finance, 2015-01-01 → 2026-01-01 |
| **Features** | Past returns (1/5/21/63d), rolling vol (21/63d), momentum (126-21d), RSI-14, volume change, 52-week high distance, relative-to-NIFTY return |
| **Target** | 21-day forward return (leak-free) |
| **Models** | Ridge Regression, Random Forest, XGBoost – walk-forward CV |
| **Strategies** | S0 NIFTY benchmark · S1 Equal-weight · S2 Min-variance · S3 Max-Sharpe (hist) · S4 Max-Sharpe (ML) · S5 Top-5 ML equal-weight · S6 HRP |
| **Backtest** | Monthly rebalance, 0.2% per-unit transaction cost, Ledoit-Wolf covariance |
| **Frontend** | FolioWise HTML page with real model data injected via `src/build_frontend.py` |

---

## Folder Structure

```
Minor_Project/
├── data/
│   ├── raw/          # yfinance downloads (prices, volumes)
│   └── processed/    # feature tables, ML predictions
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtest.ipynb
│   └── 05_frontend_export.ipynb
├── src/
│   ├── features.py         # leak-free feature builders
│   ├── backtest.py         # BacktestEngine class
│   ├── metrics.py          # performance metrics
│   └── build_frontend.py   # injects real data into HTML
├── results/                # CSV/JSON outputs from backtest
├── report/                 # charts, tables, final report PDF
├── logs/
│   ├── defects.md
│   ├── changelog.md
│   ├── handoff_s1.md … handoff_s4.md
├── frontend/
│   ├── FolioWise_baseline.html   # original template
│   └── index.html                # injected production page
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Git

### Install dependencies

```bash
# (Recommended) create a virtual environment first
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Register the kernel

```bash
python -m ipykernel install --user --name minor_project --display-name "Minor Project"
```

---

## Running the Notebooks

Run notebooks **in order**. Each notebook saves its outputs to `data/processed/` or `results/` for the next one to consume.

| # | Notebook | What it does | Key outputs |
|---|----------|--------------|-------------|
| 01 | `01_data_download.ipynb` | Downloads OHLCV for all tickers + NIFTY 50 index | `data/raw/prices.csv`, `data/raw/volumes.csv` |
| 02 | `02_feature_engineering.ipynb` | Builds leak-free features, runs leakage test | `data/processed/features.parquet` |
| 03 | `03_model_training.ipynb` | Walk-forward CV, trains Ridge/RF/XGB, computes IC | `data/processed/predictions.parquet`, `results/model_metrics.csv` |
| 04 | `04_backtest.ipynb` | Runs S0–S6 strategies, computes performance metrics | `results/strategy_returns.csv`, `results/strategy_metrics.csv`, `results/frontend.json` |
| 05 | `05_frontend_export.ipynb` | Calls `build_frontend.py` to inject data into HTML | `frontend/index.html`, `frontend/baseline_diff.txt` |

```bash
jupyter notebook
```

Then open each notebook from the `notebooks/` folder in order.

---

## Module Reference

### `src/features.py`
Leak-free feature computation. All features use **only data up to and including time t**.

| Function | Description |
|----------|-------------|
| `compute_past_returns(prices, windows)` | Past returns over 1/5/21/63 days |
| `compute_rolling_volatility(returns, windows)` | Rolling std of daily returns (21/63d) |
| `compute_momentum(prices, long_window, skip)` | 126d return skipping 21d (`prices[t-126]/prices[t-21]-1`) |
| `compute_rsi(prices, period)` | RSI-14 with Wilder smoothing |
| `compute_volume_change(volume, short_w, long_w)` | 21d avg vol / 63d avg vol − 1 |
| `compute_dist_52wh(prices)` | `price[t] / max(price[t-252:t]) - 1` |
| `compute_rel_nifty(stock_returns, nifty_returns)` | Stock 1-day return − NIFTY 1-day return |
| `build_target(prices, horizon=21)` | 21-day forward return (NaN for last 21 rows) |
| `run_leakage_test(...)` | Verifies no future data leaks into features |
| `build_features_table(prices_df, volume_df, nifty_returns)` | Full long-format feature table |

### `src/backtest.py`
`BacktestEngine` class. Instantiate with prices, returns, ML predictions, universe, and config dict.

```python
engine = BacktestEngine(prices, returns, predictions, universe, config)
results = engine.run_all_strategies()
engine.save_outputs("results/")
```

### `src/metrics.py`
Portfolio performance metrics.

| Function | Returns |
|----------|---------|
| `compute_cagr(returns)` | Annualised compound growth rate |
| `compute_volatility(returns)` | Annualised std deviation |
| `compute_sharpe(returns, risk_free=0.06)` | Sharpe ratio |
| `compute_max_drawdown(returns)` | Maximum drawdown (negative float) |
| `compute_calmar(returns)` | CAGR / abs(max drawdown) |
| `compute_ic(predictions_df)` | Information coefficient (rank corr vs realized returns) |
| `compute_strategy_metrics(daily_returns_df)` | Summary table for all strategies |

### `src/build_frontend.py`
Injects real backtest data from `results/frontend.json` into the FolioWise HTML template.

```bash
python src/build_frontend.py
```

---

## Strategies

| ID | Name | Description |
|----|------|-------------|
| S0 | NIFTY benchmark | Market-cap weights (index returns) |
| S1 | Equal weight | 1/N across universe |
| S2 | Min variance | Ledoit-Wolf covariance, long-only, 15% cap |
| S3 | Max Sharpe (hist) | Historical mean returns + Ledoit-Wolf cov |
| S4 | Max Sharpe (ML) | ML predicted returns + Ledoit-Wolf cov |
| S5 | Top-5 ML EW | Equal weight across top-5 ML predicted stocks |
| S6 | HRP | Hierarchical Risk Parity (PyPortfolioOpt) |

**Transaction cost:** 0.2% per unit of one-way turnover, applied at each monthly rebalance.

---

## Frontend

The `frontend/index.html` file is a self-contained single-page app.  
To regenerate it after a new backtest run:

```bash
python src/build_frontend.py
```

Open `frontend/index.html` in any modern browser — no server needed.

---

## Logs & Governance

| File | Purpose |
|------|---------|
| `logs/defects.md` | Bug tracker: symptom, evidence, impact, fix |
| `logs/changelog.md` | All changes with reason and stages to rerun |
| `logs/handoff_s1.md` … `handoff_s4.md` | Per-student handoff notes |

---

## Disclaimer

> FolioWise is a **student project for educational purposes only**.  
> Nothing here constitutes financial advice.  
> Past performance does not guarantee future results.  
> Always consult a SEBI-registered investment adviser before investing real money.
