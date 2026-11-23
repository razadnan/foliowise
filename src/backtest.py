"""
src/backtest.py
===============
Cost-aware monthly-rebalancing backtest engine for the NIFTY 50 ML portfolio project.

Strategies
----------
S0  NIFTY 50 benchmark (market-cap weights → pass in index returns directly)
S1  Equal weight (1/N)
S2  Minimum variance (Ledoit-Wolf covariance, long-only, 15 % per-stock cap)
S3  Maximum Sharpe – historical mean returns + Ledoit-Wolf cov
S4  Maximum Sharpe – ML predicted returns + Ledoit-Wolf cov
S5  Top-5 ML stocks, equal weight
S6  Hierarchical Risk Parity (PyPortfolioOpt HRPOpt)

Design constraints
------------------
- Monthly rebalancing on the last trading day of each calendar month.
- Covariance estimated via Ledoit-Wolf shrinkage on the trailing 252 days.
- 15 % per-stock weight cap; long-only; weights sum to 1.
- Transaction cost = 0.2 % per unit of one-way turnover.
- If the optimiser fails for any reason, fall back to equal weight.
"""

import warnings
import logging
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

try:
    from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models
    from pypfopt.risk_models import CovarianceShrinkage
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

from src.optimization.engine import PortfolioOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Runs cost-aware portfolio backtests for strategies S0–S6.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format daily adjusted closing prices (rows=dates, cols=tickers).
    returns : pd.DataFrame
        Wide-format daily simple returns (pre-computed from prices).
    predictions : pd.DataFrame
        Long-format ML predictions with columns:
        ['date', 'ticker', 'predicted'] – one row per (date, ticker).
    universe : pd.DataFrame
        Long-format universe membership with columns:
        ['date', 'ticker'] – tickers active on each rebalance date.
    config : dict
        Backtest configuration. Keys:
          'weight_cap'      : float  (default 0.15)
          'transaction_cost': float  (default 0.002)
          'cov_lookback'    : int    (default 252)
          'risk_free'       : float  (default 0.06, annual)
          'top_n_ml'        : int    (default 5, for S5)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        predictions: pd.DataFrame,
        universe: pd.DataFrame,
        config: dict,
    ):
        self.prices = prices.sort_index()
        self.returns = returns.sort_index()
        self.predictions = predictions
        self.universe = universe
        self.config = {
            "weight_cap": 0.15,
            "transaction_cost": 0.002,
            "cov_lookback": 252,
            "risk_free": 0.06,
            "top_n_ml": 5,
            **config,
        }

        # Pre-build a dict: date → list of tickers in universe
        self._universe_map: Dict[pd.Timestamp, list] = {}
        for _, row in universe.iterrows():
            d = pd.Timestamp(row["date"])
            self._universe_map.setdefault(d, []).append(row["ticker"])

        # Pre-build ML predictions dict: date → {ticker: predicted_return}
        self._pred_map: Dict[pd.Timestamp, Dict[str, float]] = {}
        for _, row in predictions.iterrows():
            d = pd.Timestamp(row["date"])
            self._pred_map.setdefault(d, {})[row["ticker"]] = float(row["predicted"])

    # ------------------------------------------------------------------
    # Rebalance schedule
    # ------------------------------------------------------------------

    def get_rebalance_dates(self) -> pd.DatetimeIndex:
        """
        Return the last trading day of each calendar month in the price series.
        """
        idx = self.returns.index
        # Group by year-month and take the last index in each group
        gb = idx.to_series().groupby([idx.year, idx.month])
        last_days = gb.apply(lambda s: s.iloc[-1])
        return pd.DatetimeIndex(last_days.values)

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    def get_ledoit_wolf_cov(
        self,
        date: pd.Timestamp,
        universe_tickers: list,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """
        Estimate the covariance matrix using Ledoit-Wolf shrinkage on the
        trailing `lookback` trading days ending at `date`.

        Returns a DataFrame with tickers as both index and columns.
        Falls back to the sample covariance if LW fails.
        """
        end = date
        start_iloc = max(0, self.returns.index.get_loc(end) - lookback + 1)
        window = self.returns.iloc[start_iloc : self.returns.index.get_loc(end) + 1]

        # Keep only universe tickers that exist in returns
        tickers = [t for t in universe_tickers if t in window.columns]
        if len(tickers) < 2:
            # Not enough assets; return identity
            return pd.DataFrame(
                np.eye(len(universe_tickers)) * 0.04,
                index=universe_tickers,
                columns=universe_tickers,
            )

        sub = window[tickers].dropna(axis=1, how="all").dropna(axis=0, how="any")
        if sub.shape[0] < 30 or sub.shape[1] < 2:
            cov = sub.cov() if sub.shape[0] >= 2 else pd.DataFrame(
                np.eye(len(tickers)) * 0.04, index=tickers, columns=tickers
            )
            return cov

        try:
            if HAS_PYPFOPT:
                cs = CovarianceShrinkage(sub)
                cov = cs.ledoit_wolf()
            else:
                lw = LedoitWolf()
                lw.fit(sub.values)
                cov = pd.DataFrame(lw.covariance_ * 252.0, index=sub.columns, columns=sub.columns)
        except Exception as exc:
            logger.warning("Ledoit-Wolf failed (%s); using sample cov.", exc)
            cov = sub.cov() * 252.0

        return cov

    # ------------------------------------------------------------------
    # Weight calculators
    # ------------------------------------------------------------------

    def _equal_weight_dict(self, tickers: list) -> dict:
        n = len(tickers)
        return {t: 1.0 / n for t in tickers}

    def compute_weights_index(self, date: pd.Timestamp, universe_tickers: list) -> dict:
        """
        S0: Return equal weights as a proxy for NIFTY index (actual market-cap
        weights require a separate data feed; equal weight is a reasonable proxy
        when the index returns are used directly in run_strategy).
        """
        return self._equal_weight_dict(universe_tickers)

    def compute_weights_equal(self, universe_tickers: list) -> dict:
        """S1: 1/N equal weight."""
        return self._equal_weight_dict(universe_tickers)

    def compute_weights_min_var(
        self, date: pd.Timestamp, universe_tickers: list
    ) -> dict:
        """S2: Minimum variance portfolio (Ledoit-Wolf cov, long-only, weight cap)."""
        try:
            cov = self.get_ledoit_wolf_cov(date, universe_tickers)
            # Keep only tickers present in cov
            valid = [t for t in universe_tickers if t in cov.index]
            if len(valid) < 2:
                return self._equal_weight_dict(universe_tickers)

            if HAS_PYPFOPT:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ef = EfficientFrontier(
                        None,
                        cov.loc[valid, valid],
                        weight_bounds=(0, self.config["weight_cap"]),
                    )
                    ef.min_volatility()
                    cleaned = ef.clean_weights()
            else:
                opt = PortfolioOptimizer(default_stock_cap=self.config["weight_cap"])
                cleaned = opt.optimize_min_variance(cov.loc[valid, valid], valid, stock_cap=self.config["weight_cap"])

            # Map back to full universe (missing = 0)
            weights = {t: 0.0 for t in universe_tickers}
            weights.update({t: cleaned.get(t, 0.0) for t in valid})
            return self._normalise(weights, universe_tickers)

        except Exception as exc:
            logger.warning("Min-var failed on %s (%s); fallback EW.", date, exc)
            return self._equal_weight_dict(universe_tickers)

    def compute_weights_max_sharpe_hist(
        self, date: pd.Timestamp, universe_tickers: list
    ) -> dict:
        """
        S3: Maximum Sharpe using historical mean returns + Ledoit-Wolf cov.
        """
        try:
            cov = self.get_ledoit_wolf_cov(date, universe_tickers)
            valid = [t for t in universe_tickers if t in cov.index]
            if len(valid) < 2:
                return self._equal_weight_dict(universe_tickers)

            # Historical returns over lookback window
            end_iloc = self.returns.index.get_loc(date)
            start_iloc = max(0, end_iloc - self.config["cov_lookback"] + 1)
            window = self.returns.iloc[start_iloc: end_iloc + 1][valid].dropna(axis=0, how="any")
            
            if HAS_PYPFOPT:
                mu = expected_returns.mean_historical_return(
                    window, returns_data=True, frequency=252
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ef = EfficientFrontier(
                        mu,
                        cov.loc[valid, valid],
                        weight_bounds=(0, self.config["weight_cap"]),
                    )
                    ef.max_sharpe(risk_free_rate=self.config["risk_free"])
                    cleaned = ef.clean_weights()
            else:
                mu = window.mean() * 252.0
                opt = PortfolioOptimizer(default_stock_cap=self.config["weight_cap"], default_rf=self.config["risk_free"])
                cleaned = opt.optimize_max_sharpe(mu, cov.loc[valid, valid], valid, stock_cap=self.config["weight_cap"])

            weights = {t: 0.0 for t in universe_tickers}
            weights.update({t: cleaned.get(t, 0.0) for t in valid})
            return self._normalise(weights, universe_tickers)

        except Exception as exc:
            logger.warning("Max-Sharpe-hist failed on %s (%s); fallback EW.", date, exc)
            return self._equal_weight_dict(universe_tickers)

    def compute_weights_max_sharpe_ml(
        self, date: pd.Timestamp, universe_tickers: list
    ) -> dict:
        """
        S4: Maximum Sharpe using ML predicted returns + Ledoit-Wolf cov.
        """
        try:
            preds = self._pred_map.get(date, {})
            if not preds:
                logger.warning("No ML predictions for %s; fallback EW.", date)
                return self._equal_weight_dict(universe_tickers)

            cov = self.get_ledoit_wolf_cov(date, universe_tickers)
            valid = [t for t in universe_tickers if t in cov.index and t in preds]
            if len(valid) < 2:
                return self._equal_weight_dict(universe_tickers)

            # Build mu series from ML predictions (annualise return or alpha)
            mu_vals = {t: preds[t] * (252 / 21) if abs(preds[t]) < 1.0 and abs(preds[t]) > 0.05 else preds[t] for t in valid}
            mu = pd.Series(mu_vals)

            if HAS_PYPFOPT:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ef = EfficientFrontier(
                        mu,
                        cov.loc[valid, valid],
                        weight_bounds=(0, self.config["weight_cap"]),
                    )
                    ef.max_sharpe(risk_free_rate=self.config["risk_free"])
                    cleaned = ef.clean_weights()
            else:
                opt = PortfolioOptimizer(default_stock_cap=self.config["weight_cap"], default_rf=self.config["risk_free"])
                cleaned = opt.optimize_max_sharpe(mu, cov.loc[valid, valid], valid, stock_cap=self.config["weight_cap"])

            weights = {t: 0.0 for t in universe_tickers}
            weights.update({t: cleaned.get(t, 0.0) for t in valid})
            return self._normalise(weights, universe_tickers)

        except Exception as exc:
            logger.warning("Max-Sharpe-ML failed on %s (%s); fallback EW.", date, exc)
            return self._equal_weight_dict(universe_tickers)

    def compute_weights_top5_ml(
        self, date: pd.Timestamp, universe_tickers: list
    ) -> dict:
        """
        S5: Equal weight across top-N ML predicted tickers.
        """
        preds = self._pred_map.get(date, {})
        top_n = self.config["top_n_ml"]

        # Filter to universe, sort descending by prediction
        candidates = {t: preds[t] for t in universe_tickers if t in preds}
        if not candidates:
            logger.warning("No ML predictions for S5 on %s; fallback EW.", date)
            return self._equal_weight_dict(universe_tickers)

        top = sorted(candidates, key=candidates.__getitem__, reverse=True)[:top_n]
        weights = {t: 0.0 for t in universe_tickers}
        for t in top:
            weights[t] = 1.0 / len(top)
        return weights

    def compute_weights_hrp(
        self, date: pd.Timestamp, universe_tickers: list
    ) -> dict:
        """
        S6: Hierarchical Risk Parity.
        """
        try:
            end_iloc = self.returns.index.get_loc(date)
            start_iloc = max(0, end_iloc - self.config["cov_lookback"] + 1)
            window = self.returns.iloc[start_iloc: end_iloc + 1]

            valid = [t for t in universe_tickers if t in window.columns]
            if len(valid) < 2:
                return self._equal_weight_dict(universe_tickers)

            sub = window[valid].dropna(axis=0, how="any")
            if sub.shape[0] < 10:
                return self._equal_weight_dict(universe_tickers)

            if HAS_PYPFOPT:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hrp = HRPOpt(returns=sub)
                    hrp.optimize()
                    cleaned = hrp.clean_weights()
            else:
                opt = PortfolioOptimizer(default_stock_cap=self.config["weight_cap"])
                cov = sub.cov() * 252.0
                cleaned = opt.optimize_hrp(cov, tickers=valid, stock_cap=self.config["weight_cap"])

            weights = {t: 0.0 for t in universe_tickers}
            weights.update({t: cleaned.get(t, 0.0) for t in valid})

            # Enforce per-stock cap (HRP doesn't apply it natively)
            cap = self.config["weight_cap"]
            weights = self._apply_cap(weights, universe_tickers, cap)
            return self._normalise(weights, universe_tickers)

        except Exception as exc:
            logger.warning("HRP failed on %s (%s); fallback EW.", date, exc)
            return self._equal_weight_dict(universe_tickers)
            logger.warning("HRP failed on %s (%s); fallback EW.", date, exc)
            return self._equal_weight_dict(universe_tickers)

    # ------------------------------------------------------------------
    # Turnover & cost
    # ------------------------------------------------------------------

    def compute_turnover(self, old_weights: dict, new_weights: dict) -> float:
        """
        One-way portfolio turnover between old and new weights.

        Turnover = 0.5 * sum(|w_new_i - w_old_i|)
        Cost = transaction_cost * turnover  (applied externally)
        """
        all_tickers = set(old_weights) | set(new_weights)
        total = sum(
            abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0))
            for t in all_tickers
        )
        return 0.5 * total

    # ------------------------------------------------------------------
    # Strategy runner
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        strategy_id: str,
        weight_func: Callable,
    ) -> pd.DataFrame:
        """
        Run a single strategy and return a daily portfolio returns DataFrame.

        Parameters
        ----------
        strategy_id : str
            Identifier used for column naming and logging.
        weight_func : Callable
            Function (date, universe_tickers) -> dict of weights.
            For S1 (equal weight), pass a lambda that ignores date.

        Returns
        -------
        pd.DataFrame
            Single column `strategy_id`, indexed by date, containing
            daily net-of-cost simple portfolio returns.
        """
        rebal_dates = self.get_rebalance_dates()
        # Only rebalance on dates in our price series
        rebal_dates = rebal_dates[rebal_dates.isin(self.returns.index)]

        current_weights: Dict[str, float] = {}
        daily_rets = []

        tc = self.config["transaction_cost"]

        all_dates = self.returns.index
        rebal_set = set(rebal_dates)

        for date in all_dates:
            universe_tickers = self._universe_map.get(
                date,
                # Fall back to nearest available universe snapshot
                self._get_nearest_universe(date),
            )
            if not universe_tickers:
                daily_rets.append((date, 0.0))
                continue

            if date in rebal_set or not current_weights:
                # Compute new weights
                new_weights = weight_func(date, universe_tickers)

                # Apply transaction cost
                turnover = self.compute_turnover(current_weights, new_weights)
                cost = tc * turnover

                current_weights = new_weights
            else:
                cost = 0.0

            # Compute portfolio return for this day
            port_ret = sum(
                current_weights.get(t, 0.0) * self.returns.loc[date, t]
                for t in universe_tickers
                if t in self.returns.columns and not pd.isna(self.returns.loc[date, t])
            )
            port_ret -= cost
            daily_rets.append((date, port_ret))

            # Drift weights with daily returns (buy-and-hold between rebalances)
            if date not in rebal_set:
                new_w = {}
                total = 0.0
                for t, w in current_weights.items():
                    ret_t = (
                        self.returns.loc[date, t]
                        if t in self.returns.columns and not pd.isna(self.returns.loc[date, t])
                        else 0.0
                    )
                    new_w[t] = w * (1 + ret_t)
                    total += new_w[t]
                if total > 0:
                    current_weights = {t: v / total for t, v in new_w.items()}

        idx, vals = zip(*daily_rets) if daily_rets else ([], [])
        return pd.DataFrame({strategy_id: vals}, index=pd.DatetimeIndex(idx))

    # ------------------------------------------------------------------
    # Run all strategies
    # ------------------------------------------------------------------

    def run_all_strategies(self) -> Dict[str, pd.DataFrame]:
        """
        Run all 7 strategies (S0–S6) and return a dict of daily return DataFrames.

        S0 uses `returns` column '^NSEI' (or the first column if absent)
        directly as a benchmark rather than re-computing weights.

        Returns
        -------
        dict
            Keys = strategy IDs ('S0'..'S6'), values = daily return DataFrames.
        """
        results: Dict[str, pd.DataFrame] = {}

        # S0: NIFTY benchmark
        if "^NSEI" in self.returns.columns:
            s0_ret = self.returns[["^NSEI"]].rename(columns={"^NSEI": "S0"})
        else:
            # Fallback: equal weight across all universe tickers as proxy
            logger.warning("^NSEI not in returns; using EW proxy for S0.")
            s0_ret = self.run_strategy("S0", self.compute_weights_equal)
        results["S0"] = s0_ret

        # S1: Equal weight
        results["S1"] = self.run_strategy(
            "S1",
            lambda date, tickers: self.compute_weights_equal(tickers),
        )

        # S2: Min variance
        results["S2"] = self.run_strategy("S2", self.compute_weights_min_var)

        # S3: Max Sharpe (historical)
        results["S3"] = self.run_strategy(
            "S3", self.compute_weights_max_sharpe_hist
        )

        # S4: Max Sharpe (ML)
        results["S4"] = self.run_strategy(
            "S4", self.compute_weights_max_sharpe_ml
        )

        # S5: Top-5 ML equal weight
        results["S5"] = self.run_strategy("S5", self.compute_weights_top5_ml)

        # S6: HRP
        results["S6"] = self.run_strategy("S6", self.compute_weights_hrp)

        return results

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def save_outputs(self, results_dir: str):
        """
        Merge all strategy daily returns into one CSV and compute metrics.
        Also writes results/frontend.json for the HTML injector.

        Parameters
        ----------
        results_dir : str
            Path to the results directory (created if absent).
        """
        from src.metrics import compute_strategy_metrics

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Run all strategies
        all_results = self.run_all_strategies()

        # Merge into a single DataFrame
        merged = pd.concat(all_results.values(), axis=1)
        merged.to_csv(results_path / "strategy_returns.csv")
        logger.info("Saved strategy_returns.csv")

        # Performance metrics
        metrics_df = compute_strategy_metrics(merged, risk_free=self.config["risk_free"])
        metrics_df.to_csv(results_path / "strategy_metrics.csv")
        logger.info("Saved strategy_metrics.csv")
        logger.info("\n%s", metrics_df.to_string())

        # Build cumulative return series
        cum_returns = (1 + merged).cumprod()
        cum_returns.to_csv(results_path / "cumulative_returns.csv")

        # frontend.json – consumed by build_frontend.py
        universe_tickers = sorted(self.prices.columns.tolist())
        stocks_data = []
        for ticker in universe_tickers:
            if ticker == "^NSEI":
                continue
            ret_series = self.returns[ticker].dropna() if ticker in self.returns.columns else pd.Series(dtype=float)
            m = float(ret_series.mean() * 252) if len(ret_series) > 0 else 0.0
            v = float(ret_series.std() * np.sqrt(252)) if len(ret_series) > 0 else 0.0
            # Try to get a friendly name from ticker (strip .NS suffix)
            name = ticker.replace(".NS", "").replace(".BO", "")
            stocks_data.append({"ticker": ticker, "name": name, "m": round(m, 4), "v": round(v, 4)})

        nifty_col = "^NSEI" if "^NSEI" in self.returns.columns else merged.columns[0]
        nifty_ret = self.returns[nifty_col].dropna() if nifty_col in self.returns.columns else pd.Series(dtype=float)
        nifty_m = float(nifty_ret.mean() * 252) if len(nifty_ret) > 0 else 0.11
        nifty_v = float(nifty_ret.std() * np.sqrt(252)) if len(nifty_ret) > 0 else 0.15

        # Compute average pairwise correlation (rho) for the HTML model
        tickers_for_corr = [t for t in universe_tickers if t != "^NSEI" and t in self.returns.columns]
        if len(tickers_for_corr) > 1:
            corr_matrix = self.returns[tickers_for_corr].corr()
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            rho = float(np.nanmean(upper_triangle))
        else:
            rho = 0.45

        frontend_payload = {
            "stocks": stocks_data,
            "nifty": {"m": round(nifty_m, 4), "v": round(nifty_v, 4)},
            "rho": round(rho, 4),
            "strategy_metrics": metrics_df.round(4).to_dict(),
        }

        with open(results_path / "frontend.json", "w") as fh:
            json.dump(frontend_payload, fh, indent=2)
        logger.info("Saved frontend.json with %d stocks.", len(stocks_data))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self, weights: dict, tickers: list) -> dict:
        """Normalise weights to sum to 1."""
        total = sum(weights.get(t, 0.0) for t in tickers)
        if total <= 0:
            return self._equal_weight_dict(tickers)
        return {t: weights.get(t, 0.0) / total for t in tickers}

    def _apply_cap(self, weights: dict, tickers: list, cap: float) -> dict:
        """
        Iteratively clip weights to `cap` and redistribute excess uniformly.
        """
        w = {t: weights.get(t, 0.0) for t in tickers}
        for _ in range(50):
            over = {t: v for t, v in w.items() if v > cap}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            under = [t for t in tickers if w[t] < cap]
            if not under:
                for t in over:
                    w[t] = cap
                break
            for t in over:
                w[t] = cap
            per_under = excess / len(under)
            for t in under:
                w[t] += per_under
        return w

    def _get_nearest_universe(self, date: pd.Timestamp) -> list:
        """Return universe tickers for the most recent date <= `date`."""
        candidates = [d for d in self._universe_map if d <= date]
        if not candidates:
            return []
        return self._universe_map[max(candidates)]
