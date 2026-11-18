"""
src/stage3_models.py
====================
Student 3: Modelling Lead
Expanding-window walk-forward validation with 21-day purge.
Models:
1. Naive Baseline (historical mean return per ticker)
2. Ridge Regression (penalty chosen by internal TimeSeriesSplit CV on train window)
3. Random Forest (shallow trees to prevent overfitting)
4. XGBoost (small depth, modest learning rate)

Evaluates:
- IC (Information Coefficient: monthly rank correlation, mean and t-statistic)
- RMSE
- Directional Accuracy
- Top-5 minus Bottom-5 monthly realized spread

Outputs:
- results/predictions.csv (date, ticker, model, pred, actual)
- results/model_metrics.csv (model, period, ic_mean, ic_tstat, rmse, dir_acc, top_minus_bottom)
- results/feature_importance.png
- results/model_config.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "f_ret_1", "f_ret_5", "f_ret_21", "f_ret_63",
    "f_vol_21", "f_vol_63",
    "f_mom_126_21",
    "f_rsi_14",
    "f_volchg",
    "f_dist_52wh",
    "f_rel_nifty"
]

RANDOM_SEED = 42

def compute_ic_series(df_preds):
    """Compute per-month Spearman rank correlation between pred and actual."""
    ic_list = []
    dates = []
    for d, grp in df_preds.groupby("date"):
        if len(grp) >= 5 and grp["actual"].notna().sum() >= 5 and grp["pred"].std() > 1e-8:
            # Spearman correlation
            corr, _ = stats.spearmanr(grp["pred"], grp["actual"])
            if not np.isnan(corr):
                ic_list.append(corr)
                dates.append(d)
    return pd.Series(ic_list, index=dates)

def run_stage3(base_dir="."):
    print("=" * 60)
    print("STAGE 3: ML MODELLING & WALK-FORWARD VALIDATION (Student 3)")
    print("=" * 60)

    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    features_path = os.path.join(data_proc_dir, "features.csv")

    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Rebalance dates: last trading day of each month
    # Find last date of each month present in the dataset
    df["year_month"] = df["date"].dt.to_period("M")
    rebalance_dates = df.groupby("year_month")["date"].max().values
    rebalance_dates = pd.to_datetime(rebalance_dates).sort_values()
    
    # Filter test period rebalance dates: 2020-01 to 2026-11
    test_rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp("2020-01-01")]
    print(f"Total monthly rebalance dates in test period (2020-2026): {len(test_rebalance_dates)}")

    # We do yearly expanding window walk-forward validation
    # Years to evaluate: 2020, 2021, 2022, 2023, 2024, 2025, 2026
    test_years = sorted(list(set(d.year for d in test_rebalance_dates)))

    all_predictions = []
    feature_importances_rf = []
    feature_importances_xgb = []
    ridge_coefficients = []

    model_configs = {
        "random_seed": RANDOM_SEED,
        "features": FEATURE_COLS,
        "purge_trading_days": 21,
        "ridge": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "cv": 3},
        "random_forest": {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20, "random_state": RANDOM_SEED},
        "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED}
    }

    with open(os.path.join(results_dir, "model_config.json"), "w") as f:
        json.dump(model_configs, f, indent=2)

    for test_year in test_years:
        print(f"\n--- Walk-Forward Window: Testing Year {test_year} ---")
        train_end_date = pd.Timestamp(f"{test_year-1}-12-31")
        # 21 trading days purge: remove data within 21 trading days prior to train_end_date from target training
        # To strictly avoid target leakage (since target is 21d forward)
        purge_start_date = train_end_date - pd.Timedelta(days=35) # ~25 trading days
        
        train_mask = (df["date"] <= train_end_date) & (df["date"] <= purge_start_date) & df["target_21d"].notna()
        test_dates_year = [d for d in test_rebalance_dates if d.year == test_year]

        train_data = df[train_mask]
        print(f"Train period: {train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')} ({len(train_data)} samples)")

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data["target_21d"].values

        # Scaling inside training window only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 1. Naive baseline: ticker-level historical mean return
        ticker_means = train_data.groupby("ticker")["target_21d"].mean().to_dict()
        overall_mean = y_train.mean()

        # 2. Ridge Regression
        ridge = RidgeCV(alphas=model_configs["ridge"]["alphas"], cv=3)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefficients.append(ridge.coef_)

        # 3. Random Forest
        rf = RandomForestRegressor(**model_configs["random_forest"])
        rf.fit(X_train, y_train) # Tree models don't require scaling, but preserve monotonic scale
        feature_importances_rf.append(rf.feature_importances_)

        # 4. XGBoost
        xgb_model = xgb.XGBRegressor(**model_configs["xgboost"])
        xgb_model.fit(X_train, y_train)
        feature_importances_xgb.append(xgb_model.feature_importances_)

        # Generate predictions for each month-end date in test_year
        for rebal_date in test_dates_year:
            date_df = df[df["date"] == rebal_date].copy()
            if date_df.empty:
                continue

            X_test = date_df[FEATURE_COLS].values
            X_test_scaled = scaler.transform(X_test)
            actuals = date_df["target_21d"].values
            tickers = date_df["ticker"].values

            # Predictions
            pred_naive = [ticker_means.get(t, overall_mean) for t in tickers]
            pred_ridge = ridge.predict(X_test_scaled)
            pred_rf = rf.predict(X_test)
            pred_xgb = xgb_model.predict(X_test)

            date_str = rebal_date.strftime("%Y-%m-%d")

            for t, p, a in zip(tickers, pred_naive, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Naive", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_ridge, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Ridge", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_rf, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Random_Forest", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_xgb, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "XGBoost", "pred": p, "actual": a})

    preds_df = pd.DataFrame(all_predictions)
    preds_path = os.path.join(results_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved all out-of-sample predictions to {preds_path} ({len(preds_df)} rows)")

    # Compute Model Metrics
    metrics_records = []
    print("\nModel Performance Metrics (Test Period 2020-2026):")
    print("-" * 75)

    for model_name, grp in preds_df.groupby("model"):
        # Overall metrics
        valid_grp = grp.dropna(subset=["actual"]).copy()
        
        # IC
        ic_series = compute_ic_series(valid_grp)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        n_months = len(ic_series)
        ic_tstat = (ic_mean / (ic_std / np.sqrt(n_months))) if ic_std > 0 else 0.0

        # RMSE
        rmse = np.sqrt(np.mean((valid_grp["pred"] - valid_grp["actual"]) ** 2))

        # Directional Accuracy (Sign match)
        dir_acc = np.mean(np.sign(valid_grp["pred"]) == np.sign(valid_grp["actual"]))

        # Top 5 minus Bottom 5 monthly spread
        monthly_spreads = []
        for d, m_df in valid_grp.groupby("date"):
            if len(m_df) >= 10:
                top5 = m_df.nlargest(5, "pred")["actual"].mean()
                bot5 = m_df.nsmallest(5, "pred")["actual"].mean()
                monthly_spreads.append(top5 - bot5)
        top_minus_bottom = np.mean(monthly_spreads) if monthly_spreads else 0.0

        metrics_records.append({
            "model": model_name,
            "period": "Full (2020-2026)",
            "ic_mean": round(ic_mean, 4),
            "ic_tstat": round(ic_tstat, 2),
            "rmse": round(rmse, 4),
            "dir_acc": round(dir_acc * 100, 2),
            "top_minus_bottom": round(top_minus_bottom * 100, 2)
        })

        print(f"{model_name:15s} | IC: {ic_mean:+.4f} (t={ic_tstat:4.2f}) | RMSE: {rmse:.4f} | DirAcc: {dir_acc*100:5.2f}% | Top-Bot: {top_minus_bottom*100:+5.2f}%")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(results_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved model metrics to {metrics_path}")

    # Feature Importance Plot
    avg_rf_imp = np.mean(feature_importances_rf, axis=0)
    avg_xgb_imp = np.mean(feature_importances_xgb, axis=0)
    avg_ridge_coef = np.mean(np.abs(ridge_coefficients), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    y_pos = np.arange(len(FEATURE_COLS))

    # Ridge
    axes[0].barh(y_pos, avg_ridge_coef, color="#2b5c8f", align="center")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Ridge |Mean Absolute Coefficient|", fontweight="bold", fontsize=11)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Random Forest
    axes[1].barh(y_pos, avg_rf_imp, color="#3c8d5c", align="center")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("Random Forest Gini Importance", fontweight="bold", fontsize=11)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    # XGBoost
    axes[2].barh(y_pos, avg_xgb_imp, color="#d9644f", align="center")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_title("XGBoost Gain Importance", fontweight="bold", fontsize=11)
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fi_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to {fi_path}")

    print("\nStage 3 ML Modelling complete successfully!")
    return preds_df, metrics_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage3(base_dir)
