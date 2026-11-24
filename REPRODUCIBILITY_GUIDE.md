# FolioWise — Complete End-to-End Reproducibility Guide

This guide provides step-by-step instructions for establishing the computational environment, replicating the full quantitative finance and machine learning pipeline, running automated regression tests, and viewing interactive dashboards for **FolioWise: Adaptive Machine Learning & Quantitative Portfolio Optimization**.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+
- **Python**: Version 3.10, 3.11, or 3.12 (Python 3.12.2 tested and verified)
- **Memory (RAM)**: Minimum 8 GB (16 GB recommended for PyTorch LSTM training)
- **Disk Space**: ~2 GB free disk space

---

## 2. Environment Setup

### 2.1 Clone Repository and Enter Directory
```bash
git clone https://github.com/razadnan/foliowise.git
cd foliowise
```

### 2.2 Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS (bash/zsh):
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*Note*: Optional external portfolio library `PyPortfolioOpt` is supported if installed, but FolioWise provides native, mathematically robust internal implementations (`src.optimization.engine.PortfolioOptimizer`) and does not strictly require it.

---

## 3. Step-by-Step Pipeline Execution

FolioWise operates a modular 5-stage sequential architecture. Run each stage in order:

### Stage 1: Point-in-Time Universe Reconstruction & Market Data Processing
Resolves semi-annual NIFTY 50 reconstitutions across 2015–2026, accounts for ticker symbol alias transitions, downloads raw prices/volumes, validates against positive price invariants, and computes total return index (TRI) dividend separation.
```bash
python src/stage1_data.py
```
- **Inputs**: `data/raw/membership_changes.csv`, `data/raw/ticker_aliases.csv`
- **Outputs**: `data/raw/prices.csv`, `data/raw/index.csv`, `data/raw/index_tri.csv`, `data/raw/india_vix.csv`

---

### Stage 2: Feature Engineering & Standardized Target Generation
Computes 21 backward-looking quantitative features (momentum, moving average ratios, RSI, MACD, annualized volatility, rate-of-change, India VIX regime metrics) and computes the standardized 20-day cross-sectional rank target without temporal look-ahead leakage.
```bash
python src/stage2_features.py
```
- **Inputs**: `data/raw/prices.csv`, `data/raw/index.csv`, `data/raw/india_vix.csv`
- **Outputs**: `data/processed/features.csv`, `data/processed/targets.csv`

---

### Stage 3: Walk-Forward Model Training & In-Fold Validation
Trains all machine learning models across 81 rolling walk-forward monthly windows:
- Model A: Ridge Linear Regularization (`TimeSeriesSplit` alpha validation)
- Model B: Random Forest Regressor
- Model C: XGBoost Gradient Booster
- Model D: PyTorch LSTM Sequence Model (genuine temporal sliding windows)
- Model E: Validation-Tuned XGBoost-LSTM Ensemble
```bash
python src/stage3_models.py
```
- **Inputs**: `data/processed/features.csv`, `data/processed/targets.csv`
- **Outputs**: `results/predictions/`, `results/models/`

---

### Stage 4: Cost-Aware Daily Backtesting (S0 to S6)
Simulates realistic daily portfolio returns from January 2020 through February 2026 under next-day execution timing, dynamic weight drift, and a flat 20 bps statutory cost schedule (STT, stamp duty, exchange friction):
- S0: NIFTY 50 TRI Benchmark
- S1: Equal Weight (1/N)
- S2: Minimum Variance (Ledoit-Wolf Covariance)
- S3: Max Sharpe (Historical Returns)
- S4: Max Sharpe (ML Expected Returns + Active Alpha Tilt)
- S5: Top-5 ML Equal Weight
- S6: Hierarchical Risk Parity (HRP)
- S7: Top-5 Momentum Benchmark
```bash
python src/stage4_backtest.py
```
- **Inputs**: `data/raw/prices.csv`, `results/predictions/`
- **Outputs**: `results/daily_returns.csv`, `results/weights.csv`, `results/strategy_metrics.csv`

---

### Stage 5: SmartFolio Experiment Ladder (E0 to E6) & Benchmark Track (B1 to B4)
Executes the standardized experiment ladder with Stationary Block Bootstrap statistical hypothesis testing against the 1/N equal-weight baseline (E0):
```bash
python src/experiments/runner.py
```
- **Outputs**:
  - `results/experiments/experiment_ladder.csv`
  - `results/experiments/benchmark_track.csv`
  - `results/experiments/bootstrap_significance.csv`
  - `results/experiments/benchmark_bootstrap_significance.csv`
  - `results/experiments/equity_curves_e0_e6.png`
  - `results/experiments/equity_curves_benchmarks.png`

---

### Stage 6: Explainability Attribution & Model Governance
Generates factor attribution and risk contribution explanations for portfolio holdings and runs the autonomous Champion-Challenger model evaluation loop:
```bash
python -c "from src.explainability.allocator import AllocationExplainer; expl = AllocationExplainer(); print('Explainability engine ready.')"
python -c "from src.adaptive.evaluator import ChampionChallengerEvaluator; gov = ChampionChallengerEvaluator(); print('Governance engine ready.')"
```
- **Outputs**: `results/feedback/prediction_feedback.csv`, `results/adaptive/model_governance_report.json`

---

### Stage 7: Build Interactive Frontend Dashboards
Synchronizes the latest 75-stock universe metrics, NIFTY baseline benchmarks, and backtest results into both investor-facing and research dashboards:
```bash
python src/build_frontend.py
```
- **Outputs**: `frontend/index.html`, `frontend/dashboard.html`

To view the dashboards:
- Open `frontend/dashboard.html` directly in any web browser (works offline with embedded datasets).
- Or run a lightweight local server:
  ```bash
  python -m http.server 8000
  ```
  and navigate to `http://localhost:8000/frontend/dashboard.html`.

---

## 4. Verification and Automated Testing

### 4.1 Run Complete Pytest Suite
Run the 72-test automated verification suite across all 13 test modules:
```bash
pytest tests/ -v
```
Expected output:
```text
============================= 72 passed in 32.66s =============================
```

### 4.2 Run Temporal Leakage Audit
Execute the multi-date historical truncation leakage check across 6 cutoff dates:
```bash
pytest tests/test_features.py::test_multi_date_leakage_on_real_data -v
```
Expected output:
```text
tests/test_features.py::test_multi_date_leakage_on_real_data PASSED
```

### 4.3 Verify Bug Fixes
Run the dedicated verification suite specifically targeting BUG-01 through BUG-12:
```bash
pytest tests/test_fix_verifications.py -v
```
Expected output:
```text
============================= 10 passed in 19.34s =============================
```

---

## 5. Random Seeds and Numerical Determinism

To ensure exact reproducibility across machines:
- **Numpy seed**: Set to `42` in data splitting and bootstrap routines.
- **Python random**: Set to `42` in data samplers.
- **Scikit-learn models**: Initialized with `random_state=42`.
- **PyTorch (LSTM)**: Initialized with `torch.manual_seed(42)` and deterministic cudnn flags.
- **XGBoost**: Configured with `random_state=42`.
