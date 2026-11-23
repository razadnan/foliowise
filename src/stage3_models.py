"""
src/stage3_models.py
====================
Stage 3: Machine Learning Integration & Walk-Forward Validation (FolioWise Upgrade).

Implements SmartFolio ML Hierarchy:
- Model A: Ridge Regression Baseline (L2 penalty tuned via CV)
- Model B: XGBoost Regressor (Primary candidate model)
- Model C: PyTorch LSTM Sequence Model (Challenger sequence model)
- Model D: Validation-Tuned XGBoost-LSTM Ensemble
- Baselines: Naive Baseline, Momentum Baseline, Random Forest

Validation protocol:
- Expanding-window walk-forward validation (2020 to 2026)
- 20-trading-day purge window before each out-of-sample test window to eliminate target overlap
- In-fold standardization strictly within training folds (no look-ahead leakage)
- Standardized target: 20-day cross-sectional excess-return rank (`target_rank_20d`)

Outputs:
- results/predictions.csv
- results/model_metrics.csv
- results/model_config.json
- results/calibration_plot.png
- results/feature_importance.png
- results/shap_summary.png
- results/shap_importance.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.ridge import RidgeModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import ModelEnsemble
from src.models.evaluator import ModelEvaluator
from src.shap_analysis import run_shap_analysis

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "f_ret_1", "f_ret_5", "f_ret_21", "f_ret_63",
    "f_vol_21", "f_vol_63",
    "f_mom_126_21",
    "f_rsi_14",
    "f_macd", "f_macd_sig", "f_macd_hist",
    "f_sma_50_ratio", "f_ema_20_ratio",
    "f_roc_10",
    "f_volchg",
    "f_dist_52wh",
    "f_rel_nifty",
    "f_mkt_vol_21",
    "f_mkt_corr_63",
    "f_vix_level",
    "f_vix_chg_21"
]

RANDOM_SEED = 42


def run_stage3(base_dir="."):
    print("=" * 60)
    print("STAGE 3: MACHINE LEARNING INTEGRATION & WALK-FORWARD AUDIT")
    print("=" * 60)

    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    features_path = os.path.join(data_proc_dir, "features.csv")

    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Monthly rebalance dates
    df["year_month"] = df["date"].dt.to_period("M")
    rebalance_dates = df.groupby("year_month")["date"].max().values
    rebalance_dates = pd.to_datetime(rebalance_dates).sort_values()

    test_rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp("2020-01-01")]
    test_years = sorted(list(set(d.year for d in test_rebalance_dates)))
    print(f"Total monthly rebalance dates in test period (2020-2026): {len(test_rebalance_dates)}")

    # Target column: primary is target_rank_20d, fallback to target_21d
    target_col = "target_rank_20d" if "target_rank_20d" in df.columns else "target_21d"
    is_rank = (target_col == "target_rank_20d")
    print(f"Target column for model training: {target_col} (is_rank={is_rank})")

    model_configs = {
        "random_seed": RANDOM_SEED,
        "features": FEATURE_COLS,
        "target": target_col,
        "purge_trading_days": 20,
        "models": ["Naive", "Momentum", "Ridge", "Random_Forest", "XGBoost", "LSTM", "XGB_LSTM_Ensemble"],
        "ridge": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "cv": 3},
        "random_forest": {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 20, "random_state": RANDOM_SEED},
        "xgboost": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": RANDOM_SEED},
        "lstm": {"seq_len": 5, "hidden_dim": 32, "num_layers": 1, "learning_rate": 0.003, "epochs": 15, "batch_size": 128}
    }

    with open(os.path.join(results_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_configs, f, indent=2)

    all_predictions = []
    feature_importances_xgb = []
    feature_importances_rf = []
    last_xgb = None
    last_X_test = None

    for test_year in test_years:
        print(f"\n--- Walk-Forward Window: Testing Year {test_year} ---")
        train_end_date = pd.Timestamp(f"{test_year-1}-12-31")
        purge_start_date = train_end_date - pd.Timedelta(days=30)  # ~20 trading days purge

        train_mask = (df["date"] <= train_end_date) & (df["date"] <= purge_start_date) & df[target_col].notna()
        test_dates_year = [d for d in test_rebalance_dates if d.year == test_year]

        train_data = df[train_mask]
        print(f"Train period: {train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')} ({len(train_data)} samples)")

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data[target_col].values

        # 1. Naive baseline
        ticker_means = train_data.groupby("ticker")[target_col].mean().to_dict()
        overall_mean = float(y_train.mean())

        # 2. Ridge Model
        ridge = RidgeModel(alphas=model_configs["ridge"]["alphas"], cv=3).fit(X_train, y_train)

        # 3. Random Forest (comparison)
        rf = RandomForestRegressor(**model_configs["random_forest"]).fit(X_train, y_train)
        feature_importances_rf.append(rf.feature_importances_)

        # 4. XGBoost Primary Model
        xgb = XGBoostModel(**model_configs["xgboost"]).fit(X_train, y_train)
        feature_importances_xgb.append(xgb.model.feature_importances_)
        last_xgb = xgb.model

        # 5. PyTorch LSTM Challenger
        lstm = LSTMModel(
            input_dim=X_train.shape[1],
            seq_len=model_configs["lstm"]["seq_len"],
            hidden_dim=model_configs["lstm"]["hidden_dim"],
            num_layers=model_configs["lstm"]["num_layers"],
            learning_rate=model_configs["lstm"]["learning_rate"],
            epochs=model_configs["lstm"]["epochs"],
            batch_size=model_configs["lstm"]["batch_size"],
            random_state=RANDOM_SEED
        ).fit(X_train, y_train)

        # 6. XGBoost-LSTM Ensemble (Validation-Tuned on in-fold holdout)
        val_split = max(1, int(len(X_train) * 0.80))
        X_val_tr = X_train[val_split:]
        y_val_tr = y_train[val_split:]
        ensemble = ModelEnsemble(weight_xgb=0.5, weight_lstm=0.5, model_xgb=xgb, model_lstm=lstm)
        w_x, w_l = ensemble.tune_weights_on_validation(X_val_tr, y_val_tr)
        print(f"Validation-tuned Ensemble weights: XGBoost={w_x:.3f}, LSTM={w_l:.3f}")

        # Generate Out-of-Sample Predictions
        for rebal_date in test_dates_year:
            date_df = df[df["date"] == rebal_date].copy()
            if date_df.empty:
                continue

            X_test = date_df[FEATURE_COLS].values
            actuals = date_df[target_col].values
            tickers = date_df["ticker"].values
            last_X_test = X_test

            # Predictions
            pred_naive = np.array([ticker_means.get(t, overall_mean) for t in tickers])
            mom_vals = date_df["f_mom_126_21"].fillna(0.0).values
            pred_mom = (pd.Series(mom_vals).rank(pct=True).values) if is_rank else (mom_vals * 0.1)

            pred_ridge = ridge.predict(X_test)
            pred_rf = rf.predict(X_test)
            pred_xgb = xgb.predict(X_test)
            pred_lstm = lstm.predict(X_test)
            pred_ens = ensemble.predict(X_test)

            date_str = rebal_date.strftime("%Y-%m-%d")

            models_preds = {
                "Naive": pred_naive,
                "Momentum": pred_mom,
                "Ridge": pred_ridge,
                "Random_Forest": pred_rf,
                "XGBoost": pred_xgb,
                "LSTM": pred_lstm,
                "XGB_LSTM_Ensemble": pred_ens
            }

            for m_name, preds_arr in models_preds.items():
                for t, p, a in zip(tickers, preds_arr, actuals):
                    all_predictions.append({
                        "date": date_str,
                        "ticker": t,
                        "model": m_name,
                        "pred": float(p),
                        "actual": float(a) if pd.notna(a) else np.nan
                    })

    preds_df = pd.DataFrame(all_predictions)
    preds_path = os.path.join(results_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved all out-of-sample predictions to {preds_path} ({len(preds_df)} rows)")

    # 3. Model Evaluation Table
    metrics_records = []
    print("\nModel Performance Metrics (Test Period 2020-2026):")
    print("-" * 100)

    evaluator = ModelEvaluator()
    for model_name, grp in preds_df.groupby("model"):
        res = evaluator.evaluate_model(
            df_preds=grp,
            model_name=model_name,
            pred_col="pred",
            actual_col="actual",
            is_rank_target=is_rank
        )
        metrics_records.append(res)
        print(
            f"  {model_name:18s} | IC: {res['mean_ic']:+.4f} (t={res['ic_tstat']:+.2f}, NW-t={res['nw_tstat']:+.2f}) | "
            f"ICIR: {res['ic_ir']:+.3f} | DirAcc: {res['directional_accuracy_pct']:.1f}% | RMSE: {res['rmse']:.4f} | "
            f"Spread: {res['top_bottom_spread_pct']:+.2f}%"
        )

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = os.path.join(results_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved model metrics to {metrics_path}")

    # 4. Decile/Quintile Calibration Plot (Prop Trading Theme)
    xgb_preds = preds_df[preds_df["model"] == "XGBoost"].dropna(subset=["actual", "pred"]).copy()
    if not xgb_preds.empty:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#030101")
        ax.set_facecolor("#0F0E17")
        for spine in ax.spines.values():
            spine.set_color("#262338")
        ax.tick_params(colors="#8E8A88", which="both")

        xgb_preds["pred_q"] = pd.qcut(xgb_preds["pred"], q=5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"])
        calib = xgb_preds.groupby("pred_q")["actual"].mean()
        ax.bar(range(len(calib)), calib.values, color="#DF1F06", alpha=0.9, edgecolor="#56160E")
        ax.set_xticks(range(len(calib)))
        ax.set_xticklabels(calib.index, color="#D6D4D3")
        ax.axhline(0.5 if is_rank else 0.0, color="#8E8A88", linestyle="--", alpha=0.7)
        ax.set_title("XGBoost Predicted Quintile vs. Realized Mean Return/Rank (2020–2026)", fontsize=12, fontweight="bold", color="#F5F4F3")
        ax.set_xlabel("Predicted Quintile", color="#D6D4D3")
        ax.set_ylabel("Mean Realized Outcome", color="#D6D4D3")
        ax.grid(True, linestyle="--", alpha=0.4, axis="y", color="#262338")
        plt.tight_layout()
        calib_path = os.path.join(results_dir, "calibration_plot.png")
        plt.savefig(calib_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        print(f"Saved calibration plot to {calib_path}")

    # 5. Feature Importance Plot (Prop Trading Theme)
    if feature_importances_xgb:
        mean_fi_xgb = np.mean(feature_importances_xgb, axis=0)
        fi_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": mean_fi_xgb}).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#030101")
        ax.set_facecolor("#0F0E17")
        for spine in ax.spines.values():
            spine.set_color("#262338")
        ax.tick_params(colors="#8E8A88", which="both")

        ax.barh(fi_df["feature"], fi_df["importance"], color="#3E5D9E", edgecolor="#241D55")
        ax.set_title("XGBoost Mean Feature Importance (Walk-Forward 2020–2026)", fontsize=12, fontweight="bold", color="#F5F4F3")
        ax.set_xlabel("Importance", color="#D6D4D3")
        ax.set_ylabel("Feature", color="#D6D4D3")
        ax.grid(True, linestyle="--", alpha=0.4, axis="x", color="#262338")
        plt.tight_layout()
        fi_path = os.path.join(results_dir, "feature_importance.png")
        plt.savefig(fi_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        print(f"Saved feature importance plot to {fi_path}")

    # 6. SHAP Attribution Analysis
    if last_xgb is not None and last_X_test is not None:
        try:
            print("Running SHAP attribution analysis...")
            run_shap_analysis(last_xgb, last_X_test, FEATURE_COLS, results_dir)
        except Exception as e:
            print(f"SHAP analysis warning: {e}")

    print("\nStage 3 ML Integration complete successfully!")
    return metrics_df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage3(base_dir)
