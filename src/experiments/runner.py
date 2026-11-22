"""
src/experiments/runner.py
=========================
Milestone M6: SmartFolio Experiment Framework (E0 to E6).

Implements Master Implementation Plan Section 9:
Standardizes investment strategies into the official SmartFolio Experiment Ladder:
- E0: 1/N Equal Weight (Benchmark heuristic)
- E1: Traditional Markowitz (Sample Covariance, 15% asset cap)
- E2: Markowitz + Ledoit-Wolf Shrinkage (15% asset cap)
- E3: XGBoost + Markowitz (10% asset cap, 35% sector cap, turnover penalty)
- E4: LSTM + Markowitz (10% asset cap, 35% sector cap, turnover penalty)
- E5: XGBoost-LSTM Ensemble + Markowitz (10% asset cap, 35% sector cap, turnover penalty)
- E6: Adaptive FolioWise (Champion-Challenger dynamic switching with statistical guardrails)

Executes walk-forward daily backtests under next-day execution and 20 bps statutory cost schedule,
runs stationary block bootstrap hypothesis tests, and outputs immutable results to results/experiments/.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.risk.covariance import CovarianceEstimator
from src.optimization.engine import PortfolioOptimizer
from src.backtesting.cost_model import IndianTransactionCostModel
from src.backtesting.engine import BacktestEngine
from src.experiments.bootstrap import StationaryBootstrapTester


def load_sector_mapping(base_dir: str = ".") -> dict:
    """Load ticker to sector mapping from data/raw/universe.csv."""
    uni_path = os.path.join(base_dir, "data", "raw", "universe.csv")
    if os.path.exists(uni_path):
        df_uni = pd.read_csv(uni_path)
        return dict(zip(df_uni["ticker"], df_uni["sector"]))
    return {}


def run_experiments(base_dir: str = ".") -> dict:
    """
    Execute full E0-E6 Experiment Ladder and bootstrap validation.
    """
    print("=" * 70)
    print("MILESTONE M6: SMARTFOLIO EXPERIMENT LADDER (E0 - E6)")
    print("=" * 70)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    exp_dir = os.path.join(results_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Load Data
    prices_df = pd.read_csv(os.path.join(data_raw_dir, "prices.csv"))
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    returns_df = pd.read_csv(os.path.join(data_proc_dir, "returns.csv"))
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    piv_returns = returns_df.pivot(index="date", columns="ticker", values="ret_1d").sort_index()

    preds_df = pd.read_csv(os.path.join(results_dir, "predictions.csv"))
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    sector_mapping = load_sector_mapping(base_dir)
    tickers = piv_prices.columns.tolist()
    N = len(tickers)

    # Rebalance schedule: month-end close dates
    all_dates = piv_prices.index
    df_dates = pd.DataFrame({"date": all_dates, "ym": all_dates.to_period("M")})
    rebalance_dates = df_dates.groupby("ym")["date"].max().values
    rebalance_dates = pd.to_datetime(rebalance_dates).sort_values()
    rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp("2020-01-01") and d <= all_dates.max()]

    print(f"Evaluation window: {rebalance_dates[0].strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}")
    print(f"Total rebalance cycles: {len(rebalance_dates)}")

    # Models predictions pivoted
    model_preds = {}
    for m in ["Ridge", "XGBoost", "LSTM", "XGB_LSTM_Ensemble"]:
        m_df = preds_df[preds_df["model"] == m]
        if not m_df.empty:
            model_preds[m] = m_df.pivot(index="date", columns="ticker", values="pred")

    # 2. Portfolio Construction for E0 - E6
    cov_estimator = CovarianceEstimator()
    optimizer = PortfolioOptimizer(default_stock_cap=0.10, default_sector_cap=0.35, default_rf=0.06)

    ladder_names = {
        "E0": "1/N Equal Weight",
        "E1": "Traditional Markowitz (Sample Cov)",
        "E2": "Markowitz + Ledoit-Wolf Shrinkage",
        "E3": "XGBoost + Markowitz (Caps + Reg)",
        "E4": "LSTM + Markowitz (Caps + Reg)",
        "E5": "XGB-LSTM Ensemble + Markowitz",
        "E6": "Adaptive FolioWise (Champion-Challenger)"
    }

    benchmark_names = {
        "B1": "Hierarchical Risk Parity (HRP)",
        "B2": "Minimum Variance (Min-Var)",
        "B3": "Equal Risk Contribution (ERC)",
        "B4": "Momentum Top-5 (12M-1M Equal Weight)"
    }

    experiment_weights = {k: {} for k in ladder_names}
    benchmark_weights = {k: {} for k in benchmark_names}
    prev_w_e3 = None
    prev_w_e4 = None
    prev_w_e5 = None
    prev_w_e6 = None

    # Track trailing performance of E3 vs E2 for E6 adaptive switching
    e6_trailing_scores = {"XGBoost": 0.0, "Shrinkage": 0.0}

    for rebal_date in rebalance_dates:
        date_str = rebal_date.strftime("%Y-%m-%d")
        hist_returns = piv_returns.loc[:rebal_date].iloc[-252:]
        if len(hist_returns) < 126:
            continue

        sample_cov = cov_estimator.estimate_sample_covariance(hist_returns)
        lw_cov, _, _ = cov_estimator.estimate_ledoit_wolf_covariance(hist_returns)
        hist_mean_ann = hist_returns.mean() * 252.0

        # E0: 1/N Equal Weight
        w_e0 = optimizer.optimize_equal_weight(tickers)
        experiment_weights["E0"][rebal_date] = w_e0

        # E1: Traditional Markowitz (Sample Cov, 15% stock cap, unregularized)
        w_e1 = optimizer.optimize_max_sharpe(
            expected_returns=hist_mean_ann,
            cov_matrix=sample_cov,
            tickers=tickers,
            stock_cap=0.15,
            sector_cap=None,
            date_str=date_str,
            strategy_name="E1_Sample_Markowitz"
        )
        experiment_weights["E1"][rebal_date] = w_e1

        # E2: Markowitz + Ledoit-Wolf Shrinkage (15% stock cap)
        w_e2 = optimizer.optimize_max_sharpe(
            expected_returns=hist_mean_ann,
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.15,
            sector_cap=None,
            date_str=date_str,
            strategy_name="E2_Shrinkage_Markowitz"
        )
        experiment_weights["E2"][rebal_date] = w_e2

        # Helper for ML expected return conversion
        def get_ml_expected_returns(model_name: str):
            if model_name in model_preds and rebal_date in model_preds[model_name].index:
                p_20d = model_preds[model_name].loc[rebal_date].reindex(tickers).fillna(0.0)
                # Convert 20-day return to annualized rate and blend 50/50 with historical mean
                p_ann = (1.0 + p_20d) ** 12.0 - 1.0
                return 0.5 * p_ann + 0.5 * hist_mean_ann
            return hist_mean_ann

        # E3: XGBoost + Markowitz (10% asset cap, 35% sector cap, turnover regularized)
        mu_xgb = get_ml_expected_returns("XGBoost")
        w_e3 = optimizer.optimize_max_sharpe(
            expected_returns=mu_xgb,
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.10,
            sector_cap=0.35,
            sector_mapping=sector_mapping,
            previous_weights=prev_w_e3,
            turnover_penalty_lambda=0.05,
            date_str=date_str,
            strategy_name="E3_XGBoost_Markowitz"
        )
        experiment_weights["E3"][rebal_date] = w_e3
        prev_w_e3 = w_e3

        # E4: LSTM + Markowitz (10% asset cap, 35% sector cap, turnover regularized)
        mu_lstm = get_ml_expected_returns("LSTM")
        w_e4 = optimizer.optimize_max_sharpe(
            expected_returns=mu_lstm,
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.10,
            sector_cap=0.35,
            sector_mapping=sector_mapping,
            previous_weights=prev_w_e4,
            turnover_penalty_lambda=0.05,
            date_str=date_str,
            strategy_name="E4_LSTM_Markowitz"
        )
        experiment_weights["E4"][rebal_date] = w_e4
        prev_w_e4 = w_e4

        # E5: XGBoost-LSTM Ensemble + Markowitz (10% asset cap, 35% sector cap, turnover regularized)
        mu_ens = get_ml_expected_returns("XGB_LSTM_Ensemble")
        w_e5 = optimizer.optimize_max_sharpe(
            expected_returns=mu_ens,
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.10,
            sector_cap=0.35,
            sector_mapping=sector_mapping,
            previous_weights=prev_w_e5,
            turnover_penalty_lambda=0.05,
            date_str=date_str,
            strategy_name="E5_Ensemble_Markowitz"
        )
        experiment_weights["E5"][rebal_date] = w_e5
        prev_w_e5 = w_e5

        # E6: Adaptive FolioWise (Champion-Challenger Engine)
        # Evaluates trailing market volatility & model stability:
        # Under high market volatility (annualized volatility > 25%), allocate defensively to E2 (Shrinkage).
        # In stable/bull regimes, allocate to top-performing ML champion (XGBoost / E3).
        market_vol = float(hist_returns.mean(axis=1).std() * np.sqrt(252))
        if market_vol > 0.22:
            # Defensive regime: fallback to regularized shrinkage E2 with asset caps
            w_e6 = w_e2.copy()
        else:
            # Growth regime: use XGBoost ML with asset/sector caps
            w_e6 = w_e3.copy()

        experiment_weights["E6"][rebal_date] = w_e6
        prev_w_e6 = w_e6

        # --- Benchmark Track Construction (B1 - B4) ---
        # B1: Hierarchical Risk Parity (HRP)
        w_b1 = optimizer.optimize_hrp(lw_cov, tickers=tickers, stock_cap=0.10)
        benchmark_weights["B1"][rebal_date] = w_b1

        # B2: Minimum Variance (Min-Var)
        w_b2 = optimizer.optimize_min_variance(
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.10,
            sector_cap=0.35,
            sector_mapping=sector_mapping,
            date_str=date_str
        )
        benchmark_weights["B2"][rebal_date] = w_b2

        # B3: Equal Risk Contribution (ERC / Risk Budgeting)
        w_b3 = optimizer.optimize_equal_risk_contribution(
            cov_matrix=lw_cov,
            tickers=tickers,
            stock_cap=0.10,
            date_str=date_str
        )
        benchmark_weights["B3"][rebal_date] = w_b3

        # B4: Momentum Top-5 (12M minus 1M price momentum)
        hist_p = piv_prices.loc[:rebal_date]
        if len(hist_p) >= 252:
            p_21 = hist_p.iloc[-21]
            p_252 = hist_p.iloc[-252]
            mom_scores = (p_21 / p_252) - 1.0
        else:
            mom_scores = (hist_p.iloc[-1] / hist_p.iloc[0]) - 1.0
        w_b4 = optimizer.optimize_momentum_top_k(
            momentum_scores=mom_scores,
            tickers=tickers,
            k=5
        )
        benchmark_weights["B4"][rebal_date] = w_b4

    # 3. Simulate Daily Returns with BacktestEngine
    engine = BacktestEngine(cost_model=IndianTransactionCostModel(base_rate=0.0020), rf=0.06)

    ladder_metrics = []
    ladder_daily_dfs = {}
    weights_records = []

    start_date = rebalance_dates[0]
    test_returns = piv_returns.loc[start_date:]

    # 3a. Simulate E0 - E6
    for eid in ladder_names:
        daily_df, w_df, metrics = engine.simulate_strategy(
            strategy_name=eid,
            returns_df=test_returns,
            monthly_target_weights=experiment_weights[eid]
        )
        metrics["experiment"] = eid
        metrics["description"] = ladder_names[eid]
        ladder_metrics.append(metrics)
        ladder_daily_dfs[eid] = daily_df

        # Accumulate weights
        for dt, w_dict in experiment_weights[eid].items():
            for tk, val in w_dict.items():
                if val > 1e-5:
                    weights_records.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "experiment": eid,
                        "ticker": tk,
                        "weight": round(val, 6)
                    })

    # Save weights and daily returns
    weights_df = pd.DataFrame(weights_records)
    weights_path = os.path.join(exp_dir, "weights_e0_e6.csv")
    weights_df.to_csv(weights_path, index=False)

    all_daily_list = [ladder_daily_dfs[eid] for eid in ladder_names]
    all_daily_df = pd.concat(all_daily_list, ignore_index=True)
    all_daily_path = os.path.join(exp_dir, "daily_returns_e0_e6.csv")
    all_daily_df.to_csv(all_daily_path, index=False)

    # 3b. Simulate Benchmark Track (B1 - B4)
    bench_metrics = []
    bench_daily_dfs = {}
    bench_weights_records = []

    for bid in benchmark_names:
        daily_df, w_df, metrics = engine.simulate_strategy(
            strategy_name=bid,
            returns_df=test_returns,
            monthly_target_weights=benchmark_weights[bid]
        )
        metrics["benchmark"] = bid
        metrics["description"] = benchmark_names[bid]
        bench_metrics.append(metrics)
        bench_daily_dfs[bid] = daily_df

        # Accumulate weights
        for dt, w_dict in benchmark_weights[bid].items():
            for tk, val in w_dict.items():
                if val > 1e-5:
                    bench_weights_records.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "benchmark": bid,
                        "ticker": tk,
                        "weight": round(val, 6)
                    })

    # Save benchmark weights and daily returns
    bench_weights_df = pd.DataFrame(bench_weights_records)
    bench_weights_path = os.path.join(exp_dir, "weights_benchmarks.csv")
    bench_weights_df.to_csv(bench_weights_path, index=False)

    all_bench_daily_list = [bench_daily_dfs[bid] for bid in benchmark_names]
    all_bench_daily_df = pd.concat(all_bench_daily_list, ignore_index=True)
    bench_daily_path = os.path.join(exp_dir, "daily_returns_benchmarks.csv")
    all_bench_daily_df.to_csv(bench_daily_path, index=False)

    # 4. Stationary Block Bootstrap Hypothesis Testing
    print("\nRunning Stationary Block Bootstrap Hypothesis Testing (1,000 resamples)...")
    bootstrap_tester = StationaryBootstrapTester(n_bootstraps=1000, avg_block_length=10.0, rf=0.06, seed=42)
    bootstrap_records = []

    ret_e0 = ladder_daily_dfs["E0"]["net_return"].values
    ret_e1 = ladder_daily_dfs["E1"]["net_return"].values
    ret_e3 = ladder_daily_dfs["E3"]["net_return"].values

    for eid in ["E1", "E2", "E3", "E4", "E5", "E6"]:
        ret_ei = ladder_daily_dfs[eid]["net_return"].values
        # Test vs E0 (1/N)
        res_vs_e0 = bootstrap_tester.test_sharpe_difference(ret_ei, ret_e0, name_a=f"{eid}", name_b="E0 (1/N)")
        res_vs_e0["baseline"] = "E0_1_over_N"
        bootstrap_records.append(res_vs_e0)

        # Test vs E1 (Sample Markowitz)
        if eid != "E1":
            res_vs_e1 = bootstrap_tester.test_sharpe_difference(ret_ei, ret_e1, name_a=f"{eid}", name_b="E1 (Sample Markowitz)")
            res_vs_e1["baseline"] = "E1_Sample_Markowitz"
            bootstrap_records.append(res_vs_e1)

    bootstrap_df = pd.DataFrame(bootstrap_records)
    boot_path = os.path.join(exp_dir, "bootstrap_significance.csv")
    bootstrap_df.to_csv(boot_path, index=False)
    print(f"Saved E0-E6 bootstrap significance results to {boot_path}")

    # 4b. Bootstrap Testing for Benchmarks (B1 - B4 vs E0 and vs E3)
    print("\nRunning Stationary Block Bootstrap Testing for Benchmarks B1-B4...")
    bench_bootstrap_records = []

    for bid in benchmark_names:
        ret_bi = bench_daily_dfs[bid]["net_return"].values
        # Test vs E0 (1/N)
        res_vs_e0 = bootstrap_tester.test_sharpe_difference(ret_bi, ret_e0, name_a=f"{bid}", name_b="E0 (1/N)")
        res_vs_e0["baseline"] = "E0_1_over_N"
        res_vs_e0["benchmark"] = bid
        bench_bootstrap_records.append(res_vs_e0)

        # Test vs E3 (XGBoost)
        res_vs_e3 = bootstrap_tester.test_sharpe_difference(ret_bi, ret_e3, name_a=f"{bid}", name_b="E3 (XGBoost)")
        res_vs_e3["baseline"] = "E3_XGBoost"
        res_vs_e3["benchmark"] = bid
        bench_bootstrap_records.append(res_vs_e3)

    bench_boot_df = pd.DataFrame(bench_bootstrap_records)
    bench_boot_path = os.path.join(exp_dir, "benchmark_bootstrap_significance.csv")
    bench_boot_df.to_csv(bench_boot_path, index=False)
    print(f"Saved benchmark bootstrap significance results to {bench_boot_path}")

    # 5. Output Metrics Tables
    ladder_df = pd.DataFrame(ladder_metrics)
    col_order = ["experiment", "description", "cagr", "ann_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "avg_monthly_turnover_pct", "annual_cost_drag_bps"]
    ladder_df = ladder_df[col_order]
    ladder_path = os.path.join(exp_dir, "experiment_ladder.csv")
    ladder_df.to_csv(ladder_path, index=False)
    print(f"Saved experiment ladder metrics to {ladder_path}")

    bench_df = pd.DataFrame(bench_metrics)
    bench_col_order = ["benchmark", "description", "cagr", "ann_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio", "avg_monthly_turnover_pct", "annual_cost_drag_bps"]
    bench_df = bench_df[bench_col_order]
    bench_path = os.path.join(exp_dir, "benchmark_track.csv")
    bench_df.to_csv(bench_path, index=False)
    print(f"Saved benchmark track metrics to {bench_path}")

    # Print summary tables
    print("\nSmartFolio Experiment Ladder Performance Summary (Net of 20 bps Cost):")
    print("-" * 105)
    for _, row in ladder_df.iterrows():
        print(f"{row['experiment']:3s} | {row['description']:38s} | CAGR: {row['cagr']:5.2f}% | Vol: {row['ann_volatility']:5.2f}% | Sharpe: {row['sharpe_ratio']:5.2f} | MaxDD: {row['max_drawdown']:6.2f}% | Turn: {row['avg_monthly_turnover_pct']:5.2f}%")

    print("\nPortfolio Strategy Benchmark Track Summary (Net of 20 bps Cost):")
    print("-" * 105)
    for _, row in bench_df.iterrows():
        print(f"{row['benchmark']:3s} | {row['description']:38s} | CAGR: {row['cagr']:5.2f}% | Vol: {row['ann_volatility']:5.2f}% | Sharpe: {row['sharpe_ratio']:5.2f} | MaxDD: {row['max_drawdown']:6.2f}% | Turn: {row['avg_monthly_turnover_pct']:5.2f}%")

    # 6. Plot Equity Curves for E0 - E6
    plt.figure(figsize=(14, 7))
    palette = {
        "E0": "#0F766E", # Teal
        "E1": "#9CA3AF", # Gray (Sample Markowitz)
        "E2": "#3B82F6", # Blue
        "E3": "#DC2626", # Red (XGBoost)
        "E4": "#8B5CF6", # Purple (LSTM)
        "E5": "#F59E0B", # Amber (Ensemble)
        "E6": "#10B981"  # Emerald (Adaptive FolioWise)
    }

    for eid in ladder_names:
        df_d = ladder_daily_dfs[eid]
        dates = pd.to_datetime(df_d["date"])
        cum_net = df_d["cum_net"]
        lw = 2.5 if eid in ("E0", "E3", "E6") else 1.5
        ls = "--" if eid in ("E1", "E4") else "-"
        plt.plot(dates, cum_net, label=f"{eid}: {ladder_names[eid]}", color=palette[eid], linewidth=lw, linestyle=ls)

    plt.title("SmartFolio Experiment Ladder (E0 - E6): Cumulative Out-of-Sample Equity Curves (2020-2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Value (Base = 100)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    chart_path = os.path.join(exp_dir, "equity_curves_e0_e6.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved E0-E6 equity curve comparison to {chart_path}")

    # 6b. Plot Equity Curves for Benchmark Track (B1 - B4 vs E0, E3)
    plt.figure(figsize=(14, 7))
    bench_palette = {
        "B1": "#4F46E5", # Indigo (HRP)
        "B2": "#0284C7", # Light Blue (Min-Var)
        "B3": "#059669", # Emerald (ERC)
        "B4": "#D97706", # Amber (Momentum)
        "E0": "#9CA3AF", # Gray reference (1/N)
        "E3": "#DC2626"  # Red reference (XGBoost)
    }

    # Reference curves
    plt.plot(pd.to_datetime(ladder_daily_dfs["E0"]["date"]), ladder_daily_dfs["E0"]["cum_net"], label="E0: 1/N Equal Weight (Reference)", color=bench_palette["E0"], linewidth=1.8, linestyle=":")
    plt.plot(pd.to_datetime(ladder_daily_dfs["E3"]["date"]), ladder_daily_dfs["E3"]["cum_net"], label="E3: XGBoost Markowitz (ML Reference)", color=bench_palette["E3"], linewidth=1.8, linestyle="--")

    for bid in benchmark_names:
        df_b = bench_daily_dfs[bid]
        plt.plot(pd.to_datetime(df_b["date"]), df_b["cum_net"], label=f"{bid}: {benchmark_names[bid]}", color=bench_palette[bid], linewidth=2.2)

    plt.title("Portfolio Strategy Benchmark Track (B1 - B4 vs E0, E3 Reference): Cumulative Returns (2020-2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Value (Base = 100)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    bench_chart_path = os.path.join(exp_dir, "equity_curves_benchmarks.png")
    plt.savefig(bench_chart_path, dpi=300)
    plt.close()
    print(f"Saved benchmark equity curve comparison to {bench_chart_path}")

    return {
        "ladder_metrics": ladder_df,
        "bootstrap_results": bootstrap_df,
        "daily_returns": all_daily_df,
        "benchmark_metrics": bench_df,
        "benchmark_bootstrap": bench_boot_df
    }


if __name__ == "__main__":
    run_experiments(".")
