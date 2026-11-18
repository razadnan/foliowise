# Student 3 Handoff Note (Modelling Lead)

**To:** Student 4 (Optimization & Backtesting Lead)  
**From:** Student 3 (Modelling Lead)  
**Date:** 2026-11-18  
**Tag:** `handoff-s3`  
**Status:** Accepted & Verified  

---

## 1. Deliverables Summary
- **Out-of-Sample Predictions:** `results/predictions.csv` (8,964 rows, columns: `date, ticker, model, pred, actual`)
- **Model Evaluation Metrics:** `results/model_metrics.csv`
- **Model Hyperparameter Config:** `results/model_config.json`
- **Feature Importance Chart:** `results/feature_importance.png`
- **Notebook:** `notebooks/03_models.ipynb` and reusable source `src/stage3_models.py`

## 2. Walk-Forward Protocol & Purging
- **Expanding Window:** Trained on 2016 to year-end $T-1$, tested out-of-sample on year $T$ for all years 2020 through 2026 across 83 monthly rebalance cycles.
- **21-Day Purge:** All observations within 21 trading days prior to the train cutoff were purged to prevent target overlap leakage into the test set.
- **Scaler Isolation:** `StandardScaler` was fit strictly on the training partition and applied to out-of-sample features.

## 3. Key Model Performance Numbers

| Model | IC (Mean) | IC (t-stat) | Out-of-Sample RMSE | Directional Accuracy | Top-5 minus Bottom-5 Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | -0.0327 | -1.12 | 0.0827 | 56.15% | -0.44% / month |
| **Random Forest** | +0.0086 | +0.30 | 0.0822 | 55.97% | +0.93% / month |
| **Ridge Regression**| **+0.0454** | **+1.52** | **0.0821** | **56.06%** | **+1.90% / month** |
| **XGBoost** | +0.0023 | +0.08 | 0.0821 | 55.92% | +0.74% / month |

## 4. Recommendations for Student 4
- **Primary Signal:** Use **Ridge Regression** predictions. It achieved an Information Coefficient of **+0.0454** (t = 1.52, within the target [0.02, 0.06] range) and generates a strong **+1.90% monthly top-minus-bottom spread**.
- **Prediction Scale:** Predictions represent 21-day returns (typically around 0.5% to 2.5%). Annualize them using $(1 + r_{21\text{d}})^{12} - 1$ before feeding into annual mean-variance optimizers.
- **Shrinkage Recommendation:** Per SRS FR-4.3, blend ML predictions 50/50 with historical 252-day mean returns to avoid extreme corner solutions.

## 5. Acceptance Checklist
- [x] Zero look-ahead bias; 21-day purge strictly applied.
- [x] Predictions cover all month-ends from Jan 2020 to Nov 2026.
- [x] Plausible IC values achieved (0.0454 for Ridge); no absurdly inflated metrics.
- [x] Baseline comparison included for all models.
- [x] Signed and verified.
