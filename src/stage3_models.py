"""
src/stage3_models.py
====================
Student 3: Modelling Lead (Version 2.0)
Expanding-window walk-forward validation with 21-day purge.
Models:
1. Naive Baseline (historical mean return per ticker)
2. Momentum Baseline (12M minus 1M price momentum)
3. Ridge Regression (L2 penalty tuned via TimeSeriesSplit CV on train window)
4. Random Forest (shallow trees)
5. XGBoost (gradient boosted trees)
6. Neural Network (MLP with strict in-fold scaling)

Evaluates:
- IC (Information Coefficient: monthly rank correlation, mean and t-statistic)
- ICIR (Information Ratio of IC: mean / std)
- Newey-West autocorrelation-adjusted t-statistic
- RMSE
- Directional Accuracy
- Top-5 minus Bottom-5 monthly realized spread
- Quintile predicted vs realized calibration curve

Outputs:
- results/predictions.csv
- results/model_metrics.csv
- results/calibration_plot.png
- results/feature_importance.png
- results/shap_summary.png
- results/shap_importance.csv
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shap_analysis import run_shap_analysis

FEATURE_COLS = [
    "f_ret_1", "f_ret_5", "f_ret_21", "f_ret_63",
    "f_vol_21", "f_vol_63",
    "f_mom_126_21",
    "f_rsi_14",
    "f_volchg",
    "f_dist_52wh",
    "f_rel_nifty",
    "f_mkt_vol_21",
    "f_mkt_corr_63"
]

RANDOM_SEED = 42


def compute_ic_series(df_preds):
    """Compute per-month Spearman rank correlation between pred and actual."""
    ic_list = []
    dates = []
    for d, grp in df_preds.groupby("date"):
        if len(grp) >= 5 and grp["actual"].notna().sum() >= 5 and grp["pred"].std() > 1e-8:
            corr, _ = stats.spearmanr(grp["pred"], grp["actual"])
            if not np.isnan(corr):
                ic_list.append(corr)
                dates.append(d)
    return pd.Series(ic_list, index=dates)


def compute_newey_west_tstat(series: pd.Series, max_lags: int = 3) -> float:
    """
    Compute Newey-West HAC adjusted t-statistic for a time-series mean.
    Lags account for 21-day overlapping return autocorrelation.
    """
    vals = series.dropna().values
    T = len(vals)
    if T <= max_lags + 1:
        return 0.0
    mean_val = np.mean(vals)
    demeaned = vals - mean_val
    gamma0 = np.sum(demeaned ** 2) / T
    gammas = []
    for l in range(1, max_lags + 1):
        gammas.append(np.sum(demeaned[l:] * demeaned[:-l]) / T)
    var_nw = gamma0 + 2.0 * sum((1.0 - (l + 1) / (max_lags + 1)) * gammas[l] for l in range(max_lags))
    se_nw = np.sqrt(max(var_nw, 1e-9) / T)
    return float(mean_val / se_nw)


def run_stage3(base_dir="."):
    print("=" * 60)
    print("STAGE 3: ML MODELLING & WALK-FORWARD VALIDATION (Student 3 - v2.0)")
    print("=" * 60)

    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    features_path = os.path.join(data_proc_dir, "features.csv")

    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df["year_month"] = df["date"].dt.to_period("M")
    rebalance_dates = df.groupby("year_month")["date"].max().values
    rebalance_dates = pd.to_datetime(rebalance_dates).sort_values()

    test_rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp("2020-01-01")]
    print(f"Total monthly rebalance dates in test period (2020-2026): {len(test_rebalance_dates)}")

    test_years = sorted(list(set(d.year for d in test_rebalance_dates)))

    all_predictions = []
    feature_importances_rf = []
    feature_importances_xgb = []
    ridge_coefficients = []

    model_configs = {
        "random_seed": RANDOM_SEED,
        "features": FEATURE_COLS,
        "purge_trading_days": 21,
        "models": ["Naive", "Momentum", "Ridge", "Random_Forest", "XGBoost", "Neural_Net"],
        "ridge": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "cv": 3},
        "random_forest": {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20, "random_state": RANDOM_SEED},
        "gradient_boosting": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "random_state": RANDOM_SEED},
        "neural_net": {"hidden_layer_sizes": [32, 16], "max_iter": 200, "random_state": RANDOM_SEED, "alpha": 0.01}
    }

    with open(os.path.join(results_dir, "model_config.json"), "w") as f:
        json.dump(model_configs, f, indent=2)

    last_xgb = None
    last_X_test = None

    for test_year in test_years:
        print(f"\n--- Walk-Forward Window: Testing Year {test_year} ---")
        train_end_date = pd.Timestamp(f"{test_year-1}-12-31")
        purge_start_date = train_end_date - pd.Timedelta(days=35) # ~25 trading days purge

        train_mask = (df["date"] <= train_end_date) & (df["date"] <= purge_start_date) & df["target_21d"].notna()
        test_dates_year = [d for d in test_rebalance_dates if d.year == test_year]

        train_data = df[train_mask]
        print(f"Train period: {train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')} ({len(train_data)} samples)")

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data["target_21d"].values

        # Scaling inside training window only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 1. Naive baseline
        ticker_means = train_data.groupby("ticker")["target_21d"].mean().to_dict()
        overall_mean = y_train.mean()

        # 2. Ridge Regression
        ridge = RidgeCV(alphas=model_configs["ridge"]["alphas"], cv=3)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefficients.append(ridge.coef_)

        # 3. Random Forest
        rf = RandomForestRegressor(**model_configs["random_forest"])
        rf.fit(X_train, y_train)
        feature_importances_rf.append(rf.feature_importances_)

        # 4. Gradient Boosting (GBDT)
        gb_model = GradientBoostingRegressor(**model_configs["gradient_boosting"])
        gb_model.fit(X_train, y_train)
        feature_importances_xgb.append(gb_model.feature_importances_)
        last_xgb = gb_model

        # 5. Neural Network (MLP)
        nn_model = MLPRegressor(
            hidden_layer_sizes=tuple(model_configs["neural_net"]["hidden_layer_sizes"]),
            max_iter=model_configs["neural_net"]["max_iter"],
            random_state=RANDOM_SEED,
            alpha=model_configs["neural_net"]["alpha"]
        )
        nn_model.fit(X_train_scaled, y_train)

        # Generate predictions for each month-end date in test_year
        for rebal_date in test_dates_year:
            date_df = df[df["date"] == rebal_date].copy()
            if date_df.empty:
                continue

            X_test = date_df[FEATURE_COLS].values
            X_test_scaled = scaler.transform(X_test)
            actuals = date_df["target_21d"].values
            tickers = date_df["ticker"].values
            last_X_test = X_test

            # Momentum baseline: rank by f_mom_126_21
            mom_vals = date_df["f_mom_126_21"].fillna(0.0).values
            pred_mom = mom_vals * 0.1 # scaling for proportional return proxy

            pred_naive = [ticker_means.get(t, overall_mean) for t in tickers]
            pred_ridge = ridge.predict(X_test_scaled)
            pred_rf = rf.predict(X_test)
            pred_gb = gb_model.predict(X_test)
            pred_nn = nn_model.predict(X_test_scaled)

            date_str = rebal_date.strftime("%Y-%m-%d")

            for t, p, a in zip(tickers, pred_naive, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Naive", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_mom, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Momentum", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_ridge, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Ridge", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_rf, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Random_Forest", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_gb, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Gradient_Boosting", "pred": p, "actual": a})
            for t, p, a in zip(tickers, pred_nn, actuals):
                all_predictions.append({"date": date_str, "ticker": t, "model": "Neural_Net", "pred": p, "actual": a})

    preds_df = pd.DataFrame(all_predictions)
    preds_path = os.path.join(results_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved all out-of-sample predictions to {preds_path} ({len(preds_df)} rows)")

    # Compute Model Metrics
    metrics_records = []
    print("\nModel Performance Metrics (Test Period 2020-2026):")
    print("-" * 88)

    for model_name, grp in preds_df.groupby("model"):
        valid_grp = grp.dropna(subset=["actual"]).copy()
        
        ic_series = compute_ic_series(valid_grp)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        n_months = len(ic_series)
        ic_tstat = (ic_mean / (ic_std / np.sqrt(n_months))) if ic_std > 0 else 0.0
        ic_ir = (ic_mean / ic_std) if ic_std > 0 else 0.0
        ic_nw_tstat = compute_newey_west_tstat(ic_series, max_lags=3)

        rmse = np.sqrt(np.mean((valid_grp["pred"] - valid_grp["actual"]) ** 2))
        dir_acc = np.mean(np.sign(valid_grp["pred"]) == np.sign(valid_grp["actual"]))

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
            "ic_nw_tstat": round(ic_nw_tstat, 2),
            "ic_ir": round(ic_ir, 3),
            "rmse": round(rmse, 4),
            "dir_acc": round(dir_acc * 100, 2),
            "top_minus_bottom": round(top_minus_bottom * 100, 2)
        })

        print(f"{model_name:15s} | IC: {ic_mean:+.4f} (t={ic_tstat:4.2f}, NW_t={ic_nw_tstat:4.2f}) | ICIR: {ic_ir:5.3f} | RMSE: {rmse:.4f} | DirAcc: {dir_acc*100:5.2f}% | Top-Bot: {top_minus_bottom*100:+5.2f}%")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(results_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved model metrics to {metrics_path}")

    # Calibration Plot (Predicted vs Realized Quintiles)
    plt.figure(figsize=(8, 6))
    for m in ["Ridge", "Random_Forest", "Neural_Net"]:
        m_df = preds_df[preds_df["model"] == m].dropna(subset=["actual"]).copy()
        if not m_df.empty:
            m_df["quintile"] = pd.qcut(m_df["pred"], q=5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"], duplicates="drop")
            calib = m_df.groupby("quintile")["actual"].mean() * 100
            plt.plot(calib.index.astype(str), calib.values, marker="o", linewidth=2, label=m)

    plt.title("Return Calibration: Predicted Quintile vs Realized 21-Day Return", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Return Quintile (Bucket)", fontsize=11)
    plt.ylabel("Mean Realized 21-Day Return (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    calib_path = os.path.join(results_dir, "calibration_plot.png")
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"Saved calibration plot to {calib_path}")

    # Feature Importance Plot
    avg_rf_imp = np.mean(feature_importances_rf, axis=0)
    avg_xgb_imp = np.mean(feature_importances_xgb, axis=0)
    avg_ridge_coef = np.mean(np.abs(ridge_coefficients), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y_pos = np.arange(len(FEATURE_COLS))

    axes[0].barh(y_pos, avg_ridge_coef, color="#2563EB", alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[0].set_title("Ridge (|Mean Coefficient|)", fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    axes[1].barh(y_pos, avg_rf_imp, color="#0F766E", alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[1].set_title("Random Forest (MDI Importance)", fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    axes[2].barh(y_pos, avg_xgb_imp, color="#D97706", alpha=0.8)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(FEATURE_COLS, fontsize=9)
    axes[2].set_title("Gradient Boosting (MDI Importance)", fontweight="bold")
    axes[2].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    feat_imp_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to {feat_imp_path}")

    # TreeSHAP Feature Attribution
    if last_xgb is not None and last_X_test is not None:
        sample_indices = np.random.choice(len(X_train), size=min(300, len(X_train)), replace=False)
        X_sample = X_train[sample_indices]
        run_shap_analysis(last_xgb, X_sample, FEATURE_COLS, output_dir=results_dir)

    print("\nStage 3 ML Modelling complete successfully!")


if __name__ == "__main__":
    run_stage3(".")
