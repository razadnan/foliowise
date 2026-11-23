# FolioWise — Deterministic Reproducibility Guide

**Project:** FolioWise: Adaptive Machine Learning & Quantitative Portfolio Optimization  
**Repository:** `https://github.com/razadnan/foliowise.git`  
**Target Environment:** Python 3.10 – 3.12 (Windows / Linux / macOS)  
**Standard Test Environment:** Windows 11 Enterprise (x64), Python 3.12.6  

---

## 1. System Prerequisites

Before executing the pipeline, ensure the host system satisfies the following baseline requirements:

| Component | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| **Operating System** | Windows 10/11, Ubuntu 20.04+, or macOS 12+ | Windows 11 / Ubuntu 22.04 LTS |
| **Python Version** | Python 3.10 to 3.12 (64-bit) | Python 3.12.6 |
| **Memory (RAM)** | 8 GB | 16 GB+ |
| **Storage** | 2 GB free disk space | 5 GB free SSD space |
| **Network Access** | Required only for initial data fetching | Offline capable using cached data |

---

## 2. Virtual Environment Setup & Installation

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/razadnan/foliowise.git
cd foliowise

# 2. Set execution policy for the current terminal session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Create clean virtual environment
python -m venv .venv

# 4. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade core package managers
python -m pip install --upgrade pip setuptools wheel

# 6. Install locked project dependencies
pip install -r requirements.txt
```

### Linux / macOS (Bash / Zsh)
```bash
# 1. Clone repository
git clone https://github.com/razadnan/foliowise.git
cd foliowise

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Dependency Verification

Verify that critical machine learning and optimization engines are installed correctly:
```powershell
python -c "import torch, xgboost, sklearn, scipy, pandas; print('PyTorch:', torch.__version__); print('XGBoost:', xgboost.__version__); print('Scikit-Learn:', sklearn.__version__)"
```
Expected output:
```text
PyTorch: 2.15.0+cpu (or >=2.0.0)
XGBoost: 2.1.4 (or >=2.1.0)
Scikit-Learn: 1.5.0
```

---

## 4. Full Pipeline Execution Order

FolioWise operates via a deterministic multi-stage pipeline. Run the stages in numerical order.

### Stage 1: Data Engineering & Universe Resolution
Reconstructs the authoritative point-in-time constituent database (80 constituent intervals, 75 historical stocks), downloads benchmark indices (^NSEI and NIFTYBEES.NS ETF TRI proxy), ingests India VIX, applies the 3-day forward-fill limit with filled boolean tracking, and generates data quality reports.
```powershell
python src/stage1_data.py
```
- **Execution Time:** ~15–30 seconds (using local cache; ~3 minutes if downloading fresh).
- **Key Artifacts Generated:**
  - `data/raw/prices.csv` (217,316 rows across 75 tickers)
  - `data/raw/index_tri.csv` (2,921 rows)
  - `data/raw/india_vix.csv` (2,910 rows)
  - `results/data_quality_summary.csv`
  - `results/normalized_prices.png`

### Stage 2: Feature Engineering & Multi-Date Leakage Verification
Calculates 21 quantitative factors and the standardized 20-day excess-return cross-sectional rank target. Executes multi-date truncation leakage tests across 6 historical cutoff dates (2018, 2019, 2021, 2022, 2023, 2024).
```powershell
python src/stage2_features.py
```
- **Execution Time:** ~10–15 seconds.
- **Key Artifacts Generated:**
  - `data/processed/features.csv` (198,491 rows)
  - `results/leakage_test_report.csv` (100% passed, 0.0 delta)
  - `results/feature_descriptive_stats.csv`

### Stage 3: Walk-Forward Machine Learning Models
Trains Ridge Regression (with chronological `TimeSeriesSplit`), shallow XGBoost Regressor, PyTorch LSTM Sequence Model, and the validation-tuned Ensemble across expanding historical walk-forward windows from 2020 through 2026. Enforces a 20-trading-day purge window.
```powershell
python src/stage3_models.py
```
- **Execution Time:** ~45–75 seconds.
- **Key Artifacts Generated:**
  - `results/predictions.csv` (43,575 walk-forward prediction rows)
  - `results/model_metrics.csv` (Rank IC, ICIR, Newey-West t-stats)
  - `results/shap_importance.csv`
  - `results/calibration_plot.png`

### Stage 4: Portfolio Optimization & Backtesting (S0 to S7)
Executes walk-forward daily backtesting for strategies S0 to S7 under strict next-day ($T+1$) execution, Ledoit-Wolf covariance shrinkage, intra-month weight drift, and 20 bps statutory delivery cost friction.
```powershell
python src/stage4_backtest.py
```
- **Execution Time:** ~30–45 seconds.
- **Key Artifacts Generated:**
  - `results/strategy_metrics.csv` (S0 to S7 metrics)
  - `results/weights.csv` (28,756 daily rebalance weights)
  - `results/daily_returns.csv` (13,408 daily return records)
  - `results/cost_sensitivity.csv` (10, 20, 32.3, 50 bps grid)
  - `results/fallback_log.csv` (0 fallbacks logged)
  - `results/equity_curves.png`
  - `results/drawdown_chart.png`

### Stage 5: Advanced Risk Analysis & Frontend Compilation
Calculates Sortino ratio, maximum drawdown duration, active tracking error, Information Ratio, stationary block bootstrap (S4 minus S1), and volatility regime terciles. Assembles `results/frontend.json` and compiles `frontend/index.html`.
```powershell
python src/stage5_analysis.py
```
- **Execution Time:** ~5–10 seconds.
- **Key Artifacts Generated:**
  - `results/advanced_strategy_metrics.csv`
  - `results/bootstrap_sharpe_difference.json`
  - `results/regime_analysis.csv`
  - `results/ablation_study.csv`
  - `results/frontend.json`
  - `frontend/index.html`

### Track B: SmartFolio Experiment Ladder (E0 to E6) & Benchmarks (B1 to B4)
Executes the institutional SmartFolio Experiment Ladder (E0–E6) and Benchmark Track (B1–B4) with 1,000 stationary block bootstrap resamples (Politis & Romano, 1994).
```powershell
python src/experiments/runner.py
```
- **Execution Time:** ~40–60 seconds.
- **Key Artifacts Generated:**
  - `results/experiments/experiment_ladder.csv`
  - `results/experiments/benchmark_track.csv`
  - `results/experiments/bootstrap_significance.csv`
  - `results/experiments/benchmark_bootstrap_significance.csv`
  - `results/experiments/equity_curves_e0_e6.png`
  - `results/experiments/equity_curves_benchmarks.png`

---

## 5. Automated Testing Verification

Execute the complete automated test suite to certify system integrity:
```powershell
python -m pytest tests/ -v
```

Expected output:
```text
======================= 62 passed, 2 warnings in 20.52s =======================
```

Run standalone internal component unit tests:
```powershell
# Verify metrics engine
python src/metrics.py

# Verify frontend compiler
python src/build_frontend.py
```

---

## 6. Verification Checkpoints & Expected Numerical Bounds

To confirm that the reproduced results match the audited baseline exactly, verify the following numerical checkpoints:

### 1. Data Dimensions Checkpoint
- `data/raw/prices.csv`: Exactly 75 unique tickers, 217,316 rows.
- `data/processed/features.csv`: Exactly 21 feature columns + targets, 198,491 rows.
- `results/weights.csv`: All date/strategy groups sum to $1.000 \pm 0.001$.

### 2. Backtest Performance Checkpoint (`results/strategy_metrics.csv`)
- **S0 (NIFTY 50 TRI):** CAGR $\approx 13.27\%$, Sharpe $\approx 0.41$.
- **S1 (Equal Weight 1/N):** CAGR $\approx 16.57\%$, Sharpe $\approx 0.66$.
- **S4 (Max Sharpe ML Blend):** CAGR $\approx 2.65\%$, Sharpe $\approx -0.40$ (due to $r_f = 6.0\%$).
- **S6 (HRP):** CAGR $\approx 16.14\%$, Sharpe $\approx 0.70$.

### 3. SmartFolio Experiment Ladder Checkpoint (`results/experiments/experiment_ladder.csv`)
- **E0 (1/N Equal Weight):** CAGR $\approx 18.20\%$, Sharpe $\approx 0.70$, Max DD $\approx -36.79\%$.
- **E3 (XGBoost + Markowitz):** CAGR $\approx 15.97\%$, Sharpe $\approx 0.65$, Max DD $\approx -26.51\%$.
- **B1 (HRP Benchmark):** CAGR $\approx 17.46\%$, Sharpe $\approx 0.73$, Max DD $\approx -31.89\%$.
- **B4 (Momentum Top-5):** CAGR $\approx 3.77\%$, Sharpe $\approx -0.09$, Max DD $\approx -44.11\%$.

---

## 7. Troubleshooting & Common Issues

### Issue 1: Windows Fatal Access Violation in PyTorch
- **Symptom:** `Windows fatal exception: access violation` when running PyTorch or Optree on Windows.
- **Root Cause:** Address space collision when `optree._C` is loaded after PyTorch threadpools initialize.
- **Solution:** Ensure `tests/conftest.py` is present. It imports `optree` and `torch` early and sets `torch._dynamo.config.disable = True`.

### Issue 2: XGBoost Typing Import Error
- **Symptom:** `ImportError: cannot import name 'IterationRange' from 'xgboost.dask._typing'`.
- **Solution:** Upgrade XGBoost to version 2.1.0 or higher:
  ```powershell
  pip install "xgboost>=2.1.0"
  ```

### Issue 3: Missing PyTorch
- **Symptom:** `ModuleNotFoundError: No module named 'torch'`.
- **Solution:** Install the official PyTorch CPU wheel:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

### Issue 4: Offline Reproduction (No Internet Access)
- The repository includes pre-downloaded, verified raw market data files in `data/raw/` (`prices.csv`, `index.csv`, `index_tri.csv`, `india_vix.csv`).
- In all pipeline scripts, pass `use_cache=True` (the default) to run completely offline without querying external Yahoo Finance APIs.
