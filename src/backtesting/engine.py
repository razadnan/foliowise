"""
src/backtesting/engine.py
=========================
Walk-Forward Daily Backtesting Engine with Strict Next-Day Execution.

Implements Master Implementation Plan Sections 8.1, 8.2, 8.3 & 8.5:
1. Signal Timing & Next-Day Execution:
   - Signals and target weights are determined at month-end close T.
   - Day T+1 uses previous drifted weights to earn returns from T to T+1.
   - Rebalance orders execute at T+1 close with one-way turnover and transaction friction deducted.
   - Target weights become effective exclusively for holding periods post T+1 close.
2. Weight Drift Modeling: Intra-month weights drift with daily individual asset returns.
3. Transaction Cost Schedule: Itemized Indian equity delivery fee model (default 20 bps).
4. Quantitative Performance Metrics: CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, Turnover.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable

from src.backtesting.cost_model import IndianTransactionCostModel


class BacktestEngine:
    """
    Simulates portfolio execution and daily returns under next-day execution timing.
    """

    def __init__(
        self,
        cost_model: Optional[IndianTransactionCostModel] = None,
        rf: float = 0.06
    ):
        self.cost_model = cost_model or IndianTransactionCostModel()
        self.rf = rf

    @staticmethod
    def compute_turnover(target_w: np.ndarray, current_w: np.ndarray) -> float:
        """
        One-way turnover: 0.5 * sum(|w_target - w_current|).
        """
        return float(0.5 * np.sum(np.abs(target_w - current_w)))

    @staticmethod
    def drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Post-return drifted weights at end of trading session.
        """
        unnorm = weights * (1.0 + returns)
        total = np.sum(unnorm)
        if total > 0:
            return unnorm / total
        return weights

    def simulate_strategy(
        self,
        strategy_name: str,
        returns_df: pd.DataFrame,
        monthly_target_weights: Dict[pd.Timestamp, Dict[str, float]],
        fee_rate: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Execute daily simulation with strict next-day execution.
        
        Parameters:
        -----------
        strategy_name: Identifier of the strategy
        returns_df: DataFrame of daily stock returns (date index, ticker columns)
        monthly_target_weights: Mapping of signal date T -> {ticker: target_weight}
        fee_rate: Optional override for one-way transaction fee
        
        Returns:
        --------
        (daily_results_df, weights_history_df, metrics_summary)
        """
        calendar = returns_df.index.sort_values()
        tickers = returns_df.columns.tolist()
        n_tickers = len(tickers)

        signal_dates = sorted(list(monthly_target_weights.keys()))
        signal_set = set(signal_dates)

        current_w = np.zeros(n_tickers)
        is_invested = False

        daily_records = []
        weights_records = []
        turnover_records = []

        pending_rebalance = None

        for t_date in calendar:
            day_returns = returns_df.loc[t_date, tickers].fillna(0.0).values

            # 1. Portfolio return during today's trading session (earned by existing weights before close)
            gross_ret = float(current_w @ day_returns) if is_invested else 0.0

            # 2. Check if a rebalance order executes today at close (T+1 execution)
            turnover_today = 0.0
            cost_today = 0.0

            if pending_rebalance is not None:
                target_dict = pending_rebalance["target_weights"]
                target_w = np.array([target_dict.get(tk, 0.0) for tk in tickers])
                if np.sum(target_w) > 0:
                    target_w = target_w / np.sum(target_w)

                if is_invested:
                    # Asset drift through today's session prior to rebalance
                    drifted_w = self.drift_weights(current_w, day_returns)
                    # One-way turnover against drifted weights
                    turnover_today = self.compute_turnover(target_w, drifted_w)
                else:
                    # Initial investment
                    turnover_today = float(np.sum(target_w))
                    is_invested = True

                cost_today = self.cost_model.compute_turnover_cost(turnover_today, fee_rate=fee_rate)
                turnover_records.append({
                    "date": t_date.strftime("%Y-%m-%d"),
                    "strategy": strategy_name,
                    "turnover": turnover_today,
                    "cost": cost_today
                })

                # Target weights execute at close of T+1 and become effective post-close
                current_w = target_w.copy()
                pending_rebalance = None
            elif is_invested:
                # Normal drift on days without rebalance
                current_w = self.drift_weights(current_w, day_returns)

            net_ret = gross_ret - cost_today

            daily_records.append({
                "date": t_date.strftime("%Y-%m-%d"),
                "strategy": strategy_name,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "turnover": turnover_today,
                "cost": cost_today
            })

            # 4. Check if today is a signal date T (signals formed at close of T)
            if t_date in signal_set:
                target_w_dict = monthly_target_weights[t_date]
                pending_rebalance = {
                    "signal_date": t_date,
                    "target_weights": target_w_dict
                }
                for tk, w_val in target_w_dict.items():
                    if w_val > 1e-5:
                        weights_records.append({
                            "date": t_date.strftime("%Y-%m-%d"),
                            "strategy": strategy_name,
                            "ticker": tk,
                            "weight": float(w_val)
                        })

        df_daily = pd.DataFrame(daily_records)
        df_weights = pd.DataFrame(weights_records)

        # Cumulative equity curves (Base 100)
        df_daily["cum_gross"] = (1.0 + df_daily["gross_return"]).cumprod() * 100.0
        df_daily["cum_net"] = (1.0 + df_daily["net_return"]).cumprod() * 100.0

        # Performance summary metrics
        metrics = self.calculate_metrics(df_daily, strategy_name)

        return df_daily, df_weights, metrics

    def calculate_metrics(self, df_daily: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """
        Compute standard institutional metrics for strategy.
        Incorporates Day 1 return into compounding, standard daily-excess Sharpe,
        downside-deviation Sortino, and initial capital drawdown.
        """
        net_rets = df_daily["net_return"].values
        gross_rets = df_daily["gross_return"].values
        n_days = len(net_rets)
        if n_days < 2:
            return {"strategy": strategy_name}

        cum_net = df_daily["cum_net"].values
        initial_capital = 100.0
        total_ret = (cum_net[-1] / initial_capital) - 1.0
        cagr = ((cum_net[-1] / initial_capital) ** (252.0 / n_days)) - 1.0
        ann_vol = float(np.std(net_rets, ddof=1) * np.sqrt(252))

        # Sharpe ratio: annualized daily excess returns over annualized volatility
        daily_rf = self.rf / 252.0
        excess_daily = net_rets - daily_rf
        ann_excess = float(np.mean(excess_daily) * 252.0)
        if ann_vol > 1e-7:
            sharpe = float(ann_excess / ann_vol)
        else:
            sharpe = 99.9 if ann_excess > 0 else 0.0

        # Sortino ratio: annualized daily excess returns over annualized downside deviation
        downside_diff = np.minimum(0.0, excess_daily)
        downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252))
        if downside_dev > 1e-7:
            sortino = float(ann_excess / downside_dev)
        else:
            sortino = 99.9 if ann_excess > 0 else 0.0

        # Drawdown: Include initial capital to capture day 1 drawdowns
        cum_with_start = np.insert(cum_net, 0, initial_capital)
        running_max = np.maximum.accumulate(cum_with_start)
        drawdowns = (cum_with_start - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        # Turnover & Cost drag
        turnovers = df_daily[df_daily["turnover"] > 0]["turnover"]
        avg_turnover = float(turnovers.mean() * 100.0) if len(turnovers) > 0 else 0.0
        cum_gross = df_daily["cum_gross"].iloc[-1]
        cagr_gross = ((cum_gross / initial_capital) ** (252.0 / n_days)) - 1.0
        annual_cost_drag_bps = float((cagr_gross - cagr) * 10000.0)

        return {
            "strategy": strategy_name,
            "cagr": round(cagr * 100.0, 2),
            "ann_volatility": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(max_dd * 100.0, 2),
            "calmar_ratio": round(calmar, 3),
            "avg_monthly_turnover_pct": round(avg_turnover, 2),
            "annual_cost_drag_bps": round(annual_cost_drag_bps, 1)
        }
