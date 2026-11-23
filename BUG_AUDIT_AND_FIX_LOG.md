# FolioWise — Comprehensive Bug Audit & Fix Log

**Project:** FolioWise: Adaptive Machine Learning & Quantitative Portfolio Optimization  
**Repository:** `https://github.com/razadnan/foliowise.git`  
**Audit Date:** November 2026  
**Auditor:** Quantitative Finance & Senior Software Engineering Audit Team  
**Final Audit Status:** **ALL 12 BUGS IDENTIFIED, REMEDIATED, AND VERIFIED (100% PASS)**

---

## 1. Executive Summary

This document provides the exhaustive, forensic log of every bug, defect, methodological error, architectural inconsistency, and numerical discrepancy identified during the end-to-end technical audit of the **FolioWise** codebase.

Each issue has been classified according to severity, traced to its exact root cause, resolved through defensive and mathematically sound code, and verified using automated unit and integration tests.

### Bug Inventory Summary

| ID | Severity | Component | Issue Summary | Status |
|:---|:---:|:---|:---|:---:|
| **BUG-001** | **Critical** | Environment / ML | XGBoost 2.0.3 typing mismatch breaking tests in Python 3.12 | `RESOLVED & VERIFIED` |
| **BUG-002** | **Critical** | Dependencies | Missing PyTorch (`torch`) dependency for `LSTMModel` challenger architecture | `RESOLVED & VERIFIED` |
| **BUG-003** | **Critical** | ML / Runtime | Windows C-extension fatal access violation in PyTorch dynamic loading | `RESOLVED & VERIFIED` |
| **BUG-004** | **High** | ML Engineering | Artificial decay lookback sequence synthesis in `LSTMModel` | `RESOLVED & VERIFIED` |
| **BUG-005** | **High** | ML Methodology | Non-chronological K-Fold CV in `RidgeCV` causing future data leakage | `RESOLVED & VERIFIED` |
| **BUG-006** | **High** | ML Ensembling | Lack of explicit weight normalization verification in `ModelEnsemble` | `RESOLVED & VERIFIED` |
| **BUG-007** | **High** | Optimization | Dual-constraint violation (stock cap vs sector cap) in `cap_and_normalize` | `RESOLVED & VERIFIED` |
| **BUG-008** | **Medium** | Architecture | Unhandled `ModuleNotFoundError` when optional `pypfopt` package is missing | `RESOLVED & VERIFIED` |
| **BUG-009** | **High** | Frontend / Reporting | Impossible non-monotonic cost sensitivity curve in `frontend.json` | `RESOLVED & VERIFIED` |
| **BUG-010** | **Medium** | Reporting / Analysis | Stale hardcoded S4 Sharpe baseline in `stage5_analysis.py` feature ablation | `RESOLVED & VERIFIED` |
| **BUG-011** | **Medium** | Analysis Pipeline | `UnboundLocalError` in `stage5_analysis.py` due to variable scope inversion | `RESOLVED & VERIFIED` |
| **BUG-012** | **Low** | Test Coverage | Lack of dedicated pytest unit tests for `src/metrics.py` and frontend schema | `RESOLVED & VERIFIED` |

---

## 2. Detailed Bug Forensic Reports

---

### BUG-001: XGBoost 2.0.3 Typing Import Mismatch in Python 3.12
- **ID:** `BUG-001`
- **Severity:** Critical
- **Component:** Environment & ML Inference (`requirements.txt`, `src/models/xgboost_model.py`)
- **Affected Files:** `requirements.txt`, `tests/test_e2e.py`, `tests/test_models.py`
- **Description:**  
  Running `pytest` under Python 3.12 resulted in an unhandled import failure:
  ```text
  ImportError: cannot import name 'IterationRange' from 'xgboost.dask._typing'
  ```
  This immediately aborted test discovery and execution for `tests/test_models.py` and `tests/test_e2e.py`.
- **Root Cause:**  
  `xgboost==2.0.3` pinned in `requirements.txt` was released prior to Python 3.12 stabilization. Internal typing imports in `xgboost.dask._typing` relied on deprecated module structures that fail under Python 3.12.
- **Remediation:**  
  Upgraded `xgboost` specification in `requirements.txt` from `xgboost==2.0.3` to `xgboost>=2.1.0` (installed `xgboost==2.1.4`).
- **Verification:**  
  Ran `pytest tests/test_models.py tests/test_e2e.py -v`. Both modules imported cleanly and passed without typing warnings or errors.

---

### BUG-002: Missing PyTorch Dependency in `requirements.txt`
- **ID:** `BUG-002`
- **Severity:** Critical
- **Component:** Dependencies & Deep Learning Sequence Modeling
- **Affected Files:** `requirements.txt`, `src/models/lstm_model.py`
- **Description:**  
  Importing or executing `src/models/lstm_model.py` threw `ModuleNotFoundError: No module named 'torch'`. The project defines `LSTMModel` as the primary non-linear deep learning challenger architecture in the SmartFolio experiment hierarchy (Model C and Experiment E4), yet `torch` was completely omitted from `requirements.txt`.
- **Root Cause:**  
  The v1 baseline assumed a local environment where PyTorch happened to be pre-installed, omitting it from the formal packaging manifest.
- **Remediation:**  
  1. Installed PyTorch CPU wheel (`torch==2.15.0+cpu`) compatible with Python 3.12 on Windows.
  2. Added `torch>=2.0.0` directly into `requirements.txt`.
- **Verification:**  
  Executed `python -c "import torch; print(torch.__version__)"` and verified clean import in `src/models/lstm_model.py`.

---

### BUG-003: Windows Fatal Access Violation in PyTorch Dynamic DLL Loading
- **ID:** `BUG-003`
- **Severity:** Critical
- **Component:** Runtime / Deep Learning Engine (`tests/conftest.py`, `src/models/lstm_model.py`)
- **Affected Files:** `src/models/lstm_model.py`, `tests/test_models.py`
- **Description:**  
  When running `pytest tests/test_models.py` on Windows 11 with Python 3.12, the test runner crashed with:
  ```text
  Windows fatal exception: access violation
  Current thread 0x00003b5c (most recent call first):
    File "...\optree\_C.cp312-win_amd64.pyd"
  ```
- **Root Cause:**  
  Under Windows, when `optree`'s native C-extension (`_C.cp312-win_amd64.pyd`) is dynamically imported *after* PyTorch has initialized its global multi-threaded OpenMP/BLAS runtime pool, a memory address space collision occurs. Additionally, PyTorch's `DataLoader` with default worker threads triggers TorchDynamo JIT hooks which are unstable under Windows Python 3.12.
- **Remediation:**  
  1. Created `tests/conftest.py` with an early-load fixture:
     ```python
     import optree
     import torch
     torch._dynamo.config.disable = True
     ```
  2. Refactored `LSTMModel.fit` in `src/models/lstm_model.py` to use direct mini-batch tensor slicing with in-memory permutation (`torch.randperm`), bypassing `DataLoader` multiprocessing overhead entirely.
  3. Removed duplicate `def predict` definition present in `src/models/lstm_model.py`.
- **Verification:**  
  Executed `pytest tests/test_models.py::test_lstm_model -v`. Passed in 1.18s with zero access violations.

---

### BUG-004: Artificial Decay Lookback Sequence Synthesis in `LSTMModel`
- **ID:** `BUG-004`
- **Severity:** High
- **Component:** Machine Learning Architecture (`src/models/lstm_model.py`)
- **Affected Files:** `src/models/lstm_model.py`
- **Description:**  
  In the initial implementation of `LSTMModel._prepare_sequences`, input 3D sequences `(n_samples, seq_len, n_features)` were synthesized by taking the contemporaneous feature row $X_t$ and multiplying it by scalar geometric decay factors:
  ```python
  # Faulty code:
  decay = np.exp(-np.arange(seq_len)[::-1] * 0.1)
  seq = np.outer(decay, x_row).reshape(seq_len, n_features)
  ```
  This synthetic sequence did not represent real historical prices or factor movements; it was an artificial mathematical decay that defeated the purpose of sequence modeling.
- **Root Cause:**  
  Incomplete prototype placeholder left during initial deep learning scaffolding.
- **Remediation:**  
  Rewrote `_prepare_sequences` to implement authentic temporal sliding-window slicing:
  $$S_t = [X_{t - \text{seq\_len} + 1}, \dots, X_{t - 1}, X_t]$$
  For initial warmup periods ($t < \text{seq\_len}$), historical padding replicates the earliest available observation backwards, ensuring zero lookahead while capturing genuine temporal feature trajectory.
- **Verification:**  
  Verified via `tests/test_models.py::test_lstm_model` that input sequences exhibit genuine temporal variation across time slices ($S_{t, 0} \neq S_{t, -1}$).

---

### BUG-005: Non-Chronological K-Fold CV in `RidgeCV` Leaking Future Observations
- **ID:** `BUG-005`
- **Severity:** High
- **Component:** Machine Learning Methodology (`src/models/ridge.py`)
- **Affected Files:** `src/models/ridge.py`
- **Description:**  
  `RidgeModel` tuned its $L_2$ regularization penalty $\alpha$ via `RidgeCV(cv=3)`. In Scikit-Learn, passing an integer `cv=3` defaults to standard unstratified K-Fold cross-validation. This randomly shuffles or splits temporal data into 3 folds, allowing future observations ($t > k$) to train the model while past observations ($t < k$) are in the validation fold.
- **Root Cause:**  
  Default Scikit-Learn parameter convention used without financial time-series override.
- **Remediation:**  
  Replaced standard K-Fold with strict chronological `TimeSeriesSplit(n_splits=self.cv)`:
  ```python
  from sklearn.model_selection import TimeSeriesSplit
  tscv = TimeSeriesSplit(n_splits=self.cv)
  self.model = RidgeCV(alphas=self.alphas, cv=tscv)
  ```
- **Verification:**  
  Verified in `tests/test_models.py::test_ridge_model` that alpha selection occurs strictly chronologically without future fold leakage.

---

### BUG-006: Missing Ensemble Weight Normalization Validation
- **ID:** `BUG-006`
- **Severity:** High
- **Component:** ML Ensembling (`src/models/ensemble.py`)
- **Affected Files:** `src/models/ensemble.py`
- **Description:**  
  `ModelEnsemble.predict` combined predictions from XGBoost and LSTM using `(w_xgb * p_xgb) + (w_lstm * p_lstm)`. However, there was no defensive assertion or re-normalization ensuring $w_{\text{xgb}} + w_{\text{lstm}} = 1.0$. If misconfigured parameters were passed, predictions would drift in scale, distorting downstream cross-sectional ranking.
- **Root Cause:**  
  Lack of defensive parameter validation.
- **Remediation:**  
  Added normalization check in `ModelEnsemble`:
  ```python
  total_w = self.weight_xgb + self.weight_lstm
  if abs(total_w - 1.0) > 1e-6:
      self.weight_xgb /= total_w
      self.weight_lstm /= total_w
  ```
- **Verification:**  
  Tested in `tests/test_models.py::test_model_ensemble` with arbitrary initial weights `[0.8, 0.8]` verifying automatic normalization to `[0.5, 0.5]`.

---

### BUG-007: Dual-Constraint Violation in `cap_and_normalize`
- **ID:** `BUG-007`
- **Severity:** High
- **Component:** Portfolio Optimization Engine (`src/optimization/engine.py`)
- **Affected Files:** `src/optimization/engine.py`
- **Description:**  
  When enforcing a single-stock concentration cap (e.g., 10%) and a sector budget cap (e.g., 35%), iterative clipping of stock weights could redistribute excess weight into other stocks in the same sector, causing the sector concentration to exceed its 35% statutory maximum.
- **Root Cause:**  
  Sequential single-pass projection where single-stock clipping redistributed weight without re-checking sector aggregate boundaries.
- **Remediation:**  
  Implemented an alternating projection algorithm in `cap_and_normalize`:
  1. Enforce single-stock upper bound $w_i \le c_{\text{stock}}$.
  2. Re-scale within each sector if $\sum_{i \in S_k} w_i > c_{\text{sector}}$.
  3. Re-normalize to $\sum w_i = 1.0$ and iterate until both constraints converge within $\epsilon = 10^{-6}$.
- **Verification:**  
  Verified via `tests/test_optimizer.py::test_optimizer_sector_cap_enforcement`. Passes across 100 random trial portfolios with zero budget or sector cap violations.

---

### BUG-008: Unhandled `ModuleNotFoundError` for Optional `pypfopt`
- **ID:** `BUG-008`
- **Severity:** Medium
- **Component:** Architecture & Backward Compatibility (`src/backtest.py`)
- **Affected Files:** `src/backtest.py`
- **Description:**  
  `src/backtest.py` had an unqualified top-level import `from pypfopt import EfficientFrontier, HRPOpt`. In environments where `pypfopt` was not installed, any script importing `src/backtest.py` crashed immediately with `ModuleNotFoundError`.
- **Root Cause:**  
  External library dependency not wrapped with defensive fallback.
- **Remediation:**  
  Wrapped the import in `try...except ImportError` with an automatic fallback to FolioWise's built-in native optimization engine (`src.optimization.engine.PortfolioOptimizer`) and `sklearn.covariance.LedoitWolf`.
- **Verification:**  
  Executed test suite in an environment without `pypfopt`; backtesting and optimization functions proceeded seamlessly using native routines.

---

### BUG-009: Impossible Non-Monotonic Cost Sensitivity Curve in `frontend.json`
- **ID:** `BUG-009`
- **Severity:** High
- **Component:** Frontend Data Contract & Stage 5 Analysis (`src/stage5_analysis.py`, `results/frontend.json`)
- **Affected Files:** `src/stage5_analysis.py`, `results/frontend.json`
- **Description:**  
  In `results/frontend.json`, the `cost_sensitivity` payload reported:
  - At 0.1% cost: Sharpe = `+0.48`
  - At 0.2% cost: Sharpe = `-0.40`
  - At 0.3% cost: Sharpe = `+0.43`
  This curve was mathematically impossible: increasing transaction costs from 0.1% to 0.2% plunged Sharpe by 0.88, yet increasing costs further to 0.3% miraculously increased Sharpe by 0.83.
- **Root Cause:**  
  In `src/stage5_analysis.py`, `s4_sharpe_at_0.1pct` was hardcoded to `0.48` and `s4_sharpe_at_0.3pct` to `0.43` from an older v1 research artifact. Meanwhile, `s4_sharpe_at_0.2pct` was dynamically reading `float(s4_row["sharpe"])` (which in v2.0 is `-0.40`).
- **Remediation:**  
  Refactored `stage5_analysis.py` to dynamically parse the empirical backtest results generated in `results/cost_sensitivity.csv`:
  - 10 bps: Sharpe = `-0.40` (CAGR 2.69%, vol 8.37%)
  - 20 bps: Sharpe = `-0.40` (CAGR 2.65%, vol 8.37%)
  - 32.3 bps: Sharpe = `-0.41` (CAGR 2.59%, vol 8.38%)
  - 50 bps: Sharpe = `-0.42` (CAGR 2.52%, vol 8.38%)
- **Verification:**  
  Verified via new automated test `tests/test_metrics_and_frontend.py::test_frontend_json_schema_and_consistency` asserting strict monotonic non-increasing Sharpe as cost increases ($S_{10} \ge S_{20} \ge S_{30}$).

---

### BUG-010: Stale Hardcoded S4 Baseline in Feature Ablation Table
- **ID:** `BUG-010`
- **Severity:** Medium
- **Component:** Reporting & Evaluation Pipeline (`src/stage5_analysis.py`)
- **Affected Files:** `src/stage5_analysis.py`, `results/ablation_study.csv`
- **Description:**  
  `src/stage5_analysis.py` hardcoded `net_sharpe_s4 = 0.46` for the full baseline in `results/ablation_study.csv`, while the actual audited S4 backtest strategy achieved Sharpe `-0.40`. This created an unexplained discrepancy between the metrics table and the ablation table.
- **Root Cause:**  
  Static dictionary definitions created prior to the Phase 1 methodology update (next-day execution, TRI benchmark, one-way turnover) were never updated to link to runtime backtest objects.
- **Remediation:**  
  Anchored the ablation table dynamically to `s4_base_sharpe = float(s4_row["sharpe"])`, with relative delta drops computed against the true empirical baseline.
- **Verification:**  
  Regenerated `results/ablation_study.csv` and confirmed baseline Sharpe matches `results/strategy_metrics.csv` exactly.

---

### BUG-011: Variable Scope Inversion (`UnboundLocalError`) in `stage5_analysis.py`
- **ID:** `BUG-011`
- **Severity:** Medium
- **Component:** Pipeline Execution (`src/stage5_analysis.py`)
- **Affected Files:** `src/stage5_analysis.py`
- **Description:**  
  During pipeline execution of `python src/stage5_analysis.py`, execution failed with:
  ```text
  UnboundLocalError: cannot access local variable 's4_row' where it is not associated with a value
  ```
- **Root Cause:**  
  `s4_row` was referenced in section 4 (ablation study) at line 214, but its assignment was located further down in section 6 (frontend JSON generation) at line 271.
- **Remediation:**  
  Reordered variable extractions (`s0_row`, `s1_row`, `s4_row`) to the top of `run_stage5` immediately after `strat_metrics_df` is loaded from disk.
- **Verification:**  
  Executed `python src/stage5_analysis.py`. Pipeline ran cleanly to completion with exit code 0.

---

### BUG-012: Incomplete Automated Test Coverage for Metrics & Frontend Contract
- **ID:** `BUG-012`
- **Severity:** Low
- **Component:** Test Suite & Quality Assurance (`tests/`)
- **Affected Files:** `tests/test_metrics_and_frontend.py`
- **Description:**  
  While `src/metrics.py` contained internal `__main__` unit tests, pytest never discovered or executed them during `pytest tests/`. Furthermore, there were no regression tests validating the structural integrity of `results/frontend.json` or verifying that `src/build_frontend.py` compiles valid HTML.
- **Root Cause:**  
  Standalone test harness not integrated into standard pytest discovery hierarchy.
- **Remediation:**  
  Created `tests/test_metrics_and_frontend.py` containing 9 comprehensive test cases:
  1. `test_compute_cagr_properties` (flat, compounding, empty)
  2. `test_compute_volatility_properties` (flat, standard normal, single observation)
  3. `test_compute_sharpe_properties` (risk-free equivalence, zero vol)
  4. `test_compute_max_drawdown_and_duration` (rising, drawdown + recovery)
  5. `test_compute_sortino_ratio` (upside-only, downside-heavy)
  6. `test_compute_tracking_error_and_ir` (active risk vs benchmark)
  7. `test_compute_ic` (perfect correlation, inverse correlation)
  8. `test_frontend_json_schema_and_consistency` (cross-file validation with `strategy_metrics.csv` and monotonicity of `cost_sensitivity`)
  9. `test_build_frontend_script` (successful compilation and template variable injection)
- **Verification:**  
  Ran `pytest tests/test_metrics_and_frontend.py -v`. All 9 tests passed.

---

## 3. Final Verification Status

Every bug logged above has been remediated, cross-verified, and integrated into the continuous regression suite:
- **Total Defect Count:** 12
- **Resolved & Verified:** 12 (100%)
- **Open Defects:** 0 (0%)
- **Automated Regression Suite:** 62 tests passing cleanly in ~20 seconds.
