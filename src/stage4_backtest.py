"""
src/stage4_backtest.py
======================
Student 4: Optimization and Backtest Lead
Implements cost-aware monthly backtest (2020-2026) for strategies S0 to S6.
- S0: NIFTY 50 index (^NSEI)
- S1: Equal Weight (1/N)
- S2: Minimum Variance (Ledoit-Wolf covariance, 15% cap)
- S3: Max Sharpe with Historical Mean Returns (Ledoit-Wolf cov, 15% cap)
- S4: Max Sharpe with ML Returns (Ridge predictions, 50/50 blend with historical, 15% cap)
- S5: Top-5 by ML Prediction, Equal Weight (20% each)
- S6: Hierarchical Risk Parity (HRP, Lopez de Prado algorithm)

Features:
- Pure scipy.optimize (SLSQP) & sklearn LedoitWolf optimization
- Accurate weight drift with daily stock prices between monthly rebalances
- Exact turnover calculation: sum(|w_new - w_drifted|)
- Transaction cost deduction on rebalance day (base 0.2%, sensitivity 0.1% and 0.3%)
- Performance metrics: CAGR, Volatility, Sharpe (rf=6%), Max Drawdown, Calmar, Avg Turnover
- Generates equity curves, drawdowns, weights over time plots
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.06
WEIGHT_CAP = 0.15
BASE_COST = 0.002 # 0.2%

def get_ledoit_wolf_cov(returns_window):
    """Compute annualised Ledoit-Wolf shrinkage covariance matrix."""
    lw = LedoitWolf()
    lw.fit(returns_window)
    cov_df = pd.DataFrame(lw.covariance_, index=returns_window.columns, columns=returns_window.columns)
    return cov_df * 252

def cap_and_normalize(weights_dict, cap=0.15):
    """Enforce long-only, max cap per asset, and sum to 1."""
    weights = pd.Series(weights_dict).clip(lower=0.0)
    if weights.sum() == 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    weights = weights / weights.sum()

    for _ in range(50):
        excess = weights[weights > cap] - cap
        if excess.empty or excess.sum() < 1e-7:
            break
        total_excess = excess.sum()
        weights = weights.clip(upper=cap)
        under_cap = weights[weights < cap]
        if under_cap.sum() > 0:
            weights[weights < cap] += total_excess * (under_cap / under_cap.sum())
        else:
            break

    return (weights / weights.sum()).to_dict()

def optimize_min_variance(cov_matrix, tickers, cap=0.15):
    """Compute Minimum Variance portfolio weights with cap via SLSQP."""
    cov = cov_matrix.loc[tickers, tickers].values
    n = len(tickers)
    w0 = np.ones(n) / n
    bounds = [(0.0, cap) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        return float(w @ cov @ w)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w_res = res.x
    else:
        # Fallback to analytical inverse-volatility
        diag = np.diag(cov)
        inv_vol = 1.0 / np.sqrt(np.maximum(diag, 1e-6))
        w_res = inv_vol / np.sum(inv_vol)

    return cap_and_normalize(dict(zip(tickers, w_res)), cap)

def optimize_max_sharpe(expected_returns, cov_matrix, tickers, cap=0.15, rf=0.06):
    """Compute Max Sharpe portfolio weights with cap via SLSQP."""
    cov = cov_matrix.loc[tickers, tickers].values
    mu = expected_returns.reindex(tickers).values
    n = len(tickers)
    w0 = np.ones(n) / n
    bounds = [(0.0, cap) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(np.maximum(w @ cov @ w, 1e-8)))
        return -(port_ret - rf) / port_vol

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w_res = res.x
    else:
        # Fallback to min variance
        return optimize_min_variance(cov_matrix, tickers, cap)

    return cap_and_normalize(dict(zip(tickers, w_res)), cap)

def get_quasi_diag(link):
    """Quasi-diagonalization of hierarchical clustering tree."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_rec_bipart(cov, sort_ix):
    """Recursive bisection for HRP weights."""
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            cov0 = cov.iloc[c_items0, c_items0]
            cov1 = cov.iloc[c_items1, c_items1]
            # Cluster variance via inverse variance weighting
            ivp0 = 1.0 / np.diag(cov0)
            ivp0 /= ivp0.sum()
            v0 = ivp0 @ cov0.values @ ivp0

            ivp1 = 1.0 / np.diag(cov1)
            ivp1 /= ivp1.sum()
            v1 = ivp1 @ cov1.values @ ivp1

            alpha = 1.0 - v0 / (v0 + v1)
            w[c_items0] *= alpha
            w[c_items1] *= 1.0 - alpha
    return w

def optimize_hrp(returns_window, cap=0.15):
    """Hierarchical Risk Parity (Lopez de Prado) with 15% cap."""
    cov = returns_window.cov() * 252
    corr = returns_window.corr()
    tickers = returns_window.columns.tolist()

    try:
        # Distance matrix
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0, 1))
        np.fill_diagonal(dist, 0.0)
        # Condensed distance
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method="single")
        sort_ix = get_quasi_diag(link)
        sorted_tickers = [tickers[i] for i in sort_ix]

        hrp_weights = get_rec_bipart(cov, sort_ix)
        hrp_dict = dict(zip(sorted_tickers, hrp_weights.values))
        return cap_and_normalize(hrp_dict, cap)
    except Exception:
        # Equal weight fallback
        return {t: 1.0 / len(tickers) for t in tickers}

def run_stage4(base_dir="."):
    print("=" * 60)
    print("STAGE 4: OPTIMIZATION & BACKTESTING (Student 4)")
    print("=" * 60)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")

    prices_df = pd.read_csv(os.path.join(data_raw_dir, "prices.csv"))
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    piv_prices = prices_df.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    returns_df = pd.read_csv(os.path.join(data_proc_dir, "returns.csv"))
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    piv_returns = returns_df.pivot(index="date", columns="ticker", values="ret_1d").sort_index()

    index_df = pd.read_csv(os.path.join(data_raw_dir, "index.csv"))
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date").set_index("date")
    index_returns = index_df["adj_close"].pct_change().dropna()

    preds_df = pd.read_csv(os.path.join(results_dir, "predictions.csv"))
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    # Best ML model based on IC is Ridge (+0.0454 IC, t=1.52)
    ml_preds = preds_df[preds_df["model"] == "Ridge"].copy()
    ml_pred_piv = ml_preds.pivot(index="date", columns="ticker", values="pred")

    # Monthly rebalance dates in test period (2020-01 to 2026-11)
    all_dates = piv_prices.index
    df_dates = pd.DataFrame({"date": all_dates, "ym": all_dates.to_period("M")})
    rebalance_dates = df_dates.groupby("ym")["date"].max().values
    rebalance_dates = pd.to_datetime(rebalance_dates).sort_values()
    rebalance_dates = [d for d in rebalance_dates if d >= pd.Timestamp("2020-01-01") and d <= all_dates.max()]

    print(f"Backtest period: {rebalance_dates[0].strftime('%Y-%m-%d')} to {all_dates.max().strftime('%Y-%m-%d')}")
    print(f"Total rebalance cycles: {len(rebalance_dates)}")

    tickers = piv_prices.columns.tolist()
    N = len(tickers)

    strategies = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    strategy_names = {
        "S0": "NIFTY 50 Index (Benchmark)",
        "S1": "Equal Weight (1/N)",
        "S2": "Minimum Variance",
        "S3": "Max Sharpe (Historical)",
        "S4": "Max Sharpe (ML Predicted + Blend)",
        "S5": "Top-5 ML Equal Weight",
        "S6": "Hierarchical Risk Parity (HRP)"
    }

    all_weights_records = []
    rebal_weights = {s: {} for s in strategies if s != "S0"}

    for rebal_date in rebalance_dates:
        # History up to rebalance date (lookback 252 days)
        hist_returns = piv_returns.loc[:rebal_date].iloc[-252:]
        if len(hist_returns) < 126:
            continue

        cov_matrix = get_ledoit_wolf_cov(hist_returns)
        hist_mean_ann = hist_returns.mean() * 252

        # S1: Equal Weight
        w_s1 = {t: 1.0 / N for t in tickers}
        rebal_weights["S1"][rebal_date] = w_s1

        # S2: Min Variance
        w_s2 = optimize_min_variance(cov_matrix, tickers, cap=WEIGHT_CAP)
        rebal_weights["S2"][rebal_date] = w_s2

        # S3: Max Sharpe Historical
        w_s3 = optimize_max_sharpe(hist_mean_ann, cov_matrix, tickers, cap=WEIGHT_CAP, rf=RISK_FREE_RATE)
        rebal_weights["S3"][rebal_date] = w_s3

        # S4: Max Sharpe ML (50/50 blend per SRS FR-4.3)
        if rebal_date in ml_pred_piv.index:
            m_preds_raw = ml_pred_piv.loc[rebal_date].reindex(tickers).fillna(0.0)
            ml_mean_ann = (1.0 + m_preds_raw).clip(lower=-0.5, upper=1.0) ** 12 - 1.0
            blended_returns = 0.5 * ml_mean_ann + 0.5 * hist_mean_ann
            w_s4 = optimize_max_sharpe(blended_returns, cov_matrix, tickers, cap=WEIGHT_CAP, rf=RISK_FREE_RATE)
        else:
            w_s4 = w_s3
        rebal_weights["S4"][rebal_date] = w_s4

        # S5: Top-5 ML Equal Weight (20% each in top 5 predicted stocks)
        if rebal_date in ml_pred_piv.index:
            top5_tickers = ml_pred_piv.loc[rebal_date].nlargest(5).index.tolist()
            w_s5 = {t: (0.20 if t in top5_tickers else 0.0) for t in tickers}
        else:
            top5_tickers = hist_mean_ann.nlargest(5).index.tolist()
            w_s5 = {t: (0.20 if t in top5_tickers else 0.0) for t in tickers}
        rebal_weights["S5"][rebal_date] = w_s5

        # S6: HRP
        w_s6 = optimize_hrp(hist_returns, cap=WEIGHT_CAP)
        rebal_weights["S6"][rebal_date] = w_s6

        for s in strategies:
            if s == "S0":
                continue
            for t, w in rebal_weights[s][rebal_date].items():
                all_weights_records.append({
                    "date": rebal_date.strftime("%Y-%m-%d"),
                    "strategy": s,
                    "ticker": t,
                    "weight": round(w, 6)
                })

    weights_df = pd.DataFrame(all_weights_records)
    weights_path = os.path.join(results_dir, "weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"Saved rebalance weights to {weights_path} ({len(weights_df)} rows)")

    # Simulate Daily Returns with Drift & Transaction Costs
    start_sim_date = rebalance_dates[0]
    test_calendar = all_dates[all_dates >= start_sim_date]

    daily_results = {s: [] for s in strategies}

    for s in strategies:
        if s == "S0":
            for d in test_calendar:
                idx_ret = index_returns.get(d, 0.0)
                daily_results["S0"].append({
                    "date": d.strftime("%Y-%m-%d"),
                    "strategy": "S0",
                    "gross_ret": idx_ret,
                    "net_ret": idx_ret,
                    "turnover": 0.0
                })
            continue

        current_weights = pd.Series(rebal_weights[s][start_sim_date])

        for i, d in enumerate(test_calendar):
            day_returns = piv_returns.loc[d].reindex(tickers).fillna(0.0)

            turnover = 0.0
            if d in rebalance_dates and i > 0:
                target_w = pd.Series(rebal_weights[s][d])
                turnover = np.sum(np.abs(target_w - current_weights))
                current_weights = target_w.copy()
            elif i == 0:
                turnover = 1.0

            gross_ret = np.sum(current_weights * day_returns)
            cost_penalty = BASE_COST * turnover if turnover > 0 and i > 0 else 0.0
            net_ret = gross_ret - cost_penalty

            daily_results[s].append({
                "date": d.strftime("%Y-%m-%d"),
                "strategy": s,
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "turnover": turnover
            })

            # Weight drift with price changes
            current_weights = current_weights * (1.0 + day_returns)
            port_growth = 1.0 + gross_ret
            if port_growth > 0:
                current_weights = current_weights / port_growth
            else:
                current_weights = pd.Series(1.0 / N, index=tickers)

    all_daily_records = []
    for s in strategies:
        all_daily_records.extend(daily_results[s])
    daily_returns_df = pd.DataFrame(all_daily_records)
    daily_ret_path = os.path.join(results_dir, "daily_returns.csv")
    daily_returns_df.to_csv(daily_ret_path, index=False)
    print(f"Saved daily returns to {daily_ret_path} ({len(daily_returns_df)} rows)")

    def calc_metrics(ret_series, turnover_series, cost=BASE_COST):
        cum_ret = (1.0 + ret_series).cumprod()
        total_days = len(ret_series)
        years = total_days / 252.0
        cagr = (cum_ret.iloc[-1] ** (1.0 / years)) - 1.0
        ann_vol = ret_series.std() * np.sqrt(252)
        excess_ret = cagr - RISK_FREE_RATE
        sharpe = excess_ret / ann_vol if ann_vol > 0 else 0.0

        peak = cum_ret.cummax()
        dd = (cum_ret - peak) / peak
        max_dd = dd.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
        avg_turnover = turnover_series[turnover_series > 0].mean() if (turnover_series > 0).any() else 0.0

        return {
            "cagr": cagr,
            "vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "calmar": calmar,
            "avg_turnover": avg_turnover
        }

    strat_metrics = []
    print("\nStrategy Performance Summary (2020-2026, Net of 0.2% Cost):")
    print("-" * 88)

    for s in strategies:
        s_df = daily_returns_df[daily_returns_df["strategy"] == s].set_index("date")
        m = calc_metrics(s_df["net_ret"], s_df["turnover"])
        strat_metrics.append({
            "strategy": s,
            "strategy_name": strategy_names[s],
            "cost": BASE_COST,
            "cagr": round(m["cagr"] * 100, 2),
            "vol": round(m["vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 2),
            "max_dd": round(m["max_dd"] * 100, 2),
            "calmar": round(m["calmar"], 2),
            "avg_turnover": round(m["avg_turnover"] * 100, 2)
        })
        print(f"{s:2s} ({strategy_names[s][:28]:28s}) | CAGR: {m['cagr']*100:5.2f}% | Vol: {m['vol']*100:5.2f}% | Sharpe: {m['sharpe']:4.2f} | MaxDD: {m['max_dd']*100:6.2f}% | Turnover: {m['avg_turnover']*100:5.1f}%")

    strategy_metrics_df = pd.DataFrame(strat_metrics)
    strat_metrics_path = os.path.join(results_dir, "strategy_metrics.csv")
    strategy_metrics_df.to_csv(strat_metrics_path, index=False)
    print(f"\nSaved strategy metrics to {strat_metrics_path}")

    # Cost Sensitivity Analysis
    cost_sensitivity_records = []
    for cost in [0.001, 0.002, 0.003]:
        for s in ["S1", "S4"]:
            s_df = daily_returns_df[daily_returns_df["strategy"] == s].copy()
            rebal_cost = cost * s_df["turnover"]
            net_r = s_df["gross_ret"] - rebal_cost
            m = calc_metrics(net_r, s_df["turnover"], cost=cost)
            cost_sensitivity_records.append({
                "strategy": s,
                "strategy_name": strategy_names[s],
                "transaction_cost_pct": cost * 100,
                "cagr": round(m["cagr"] * 100, 2),
                "vol": round(m["vol"] * 100, 2),
                "sharpe": round(m["sharpe"], 2),
                "max_dd": round(m["max_dd"] * 100, 2)
            })

    cost_df = pd.DataFrame(cost_sensitivity_records)
    cost_path = os.path.join(results_dir, "cost_sensitivity.csv")
    cost_df.to_csv(cost_path, index=False)
    print(f"Saved cost sensitivity analysis to {cost_path}")

    # Visualizations
    palette = {
        "S0": "#111111",
        "S1": "#2b5c8f",
        "S2": "#3c8d5c",
        "S3": "#d9822b",
        "S4": "#0F6E63",
        "S5": "#b94331",
        "S6": "#7c5cbf"
    }

    # 1. Equity Curves
    plt.figure(figsize=(12, 7))
    for s in strategies:
        s_df = daily_returns_df[daily_returns_df["strategy"] == s]
        cum = (1.0 + s_df["net_ret"]).cumprod()
        dates_dt = pd.to_datetime(s_df["date"])
        lw = 2.5 if s in ("S0", "S1", "S4") else 1.5
        ls = "--" if s == "S0" else "-"
        plt.plot(dates_dt, cum, label=f"{s}: {strategy_names[s]}", color=palette[s], linewidth=lw, linestyle=ls)

    plt.title("Cumulative Portfolio Performance (Net of 0.2% Cost, 2020 - 2026)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Portfolio Value (Base = 1.0)", fontsize=11)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    eq_path = os.path.join(results_dir, "equity_curves.png")
    plt.savefig(eq_path, dpi=300)
    plt.close()
    print(f"Saved equity curves plot to {eq_path}")

    # 2. Drawdowns
    plt.figure(figsize=(12, 6))
    for s in ["S0", "S1", "S2", "S4"]:
        s_df = daily_returns_df[daily_returns_df["strategy"] == s]
        cum = (1.0 + s_df["net_ret"]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100
        dates_dt = pd.to_datetime(s_df["date"])
        plt.plot(dates_dt, dd, label=f"{s}: {strategy_names[s]}", color=palette[s], linewidth=1.5)

    plt.title("Underwater Drawdown (%) Over Time", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Drawdown (%)", fontsize=11)
    plt.legend(loc="lower left", fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    dd_path = os.path.join(results_dir, "drawdown_chart.png")
    plt.savefig(dd_path, dpi=300)
    plt.close()
    print(f"Saved drawdown chart to {dd_path}")

    # 3. Weights Over Time (S4)
    plt.figure(figsize=(13, 6))
    w_s4_df = weights_df[weights_df["strategy"] == "S4"].pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    w_dates = pd.to_datetime(w_s4_df.index)
    plt.stackplot(w_dates, w_s4_df.T.values, labels=w_s4_df.columns, alpha=0.85)
    plt.title("S4: ML-Driven Asset Allocations Over Time (Capped at 15%)", fontsize=13, fontweight="bold")
    plt.xlabel("Rebalance Date", fontsize=11)
    plt.ylabel("Portfolio Weight", fontsize=11)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7.5, ncol=1)
    plt.tight_layout()
    wt_path = os.path.join(results_dir, "weights_over_time.png")
    plt.savefig(wt_path, dpi=300)
    plt.close()
    print(f"Saved weights over time plot to {wt_path}")

    print("\nStage 4 Optimization and Backtesting complete successfully!")
    return strategy_metrics_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage4(base_dir)
