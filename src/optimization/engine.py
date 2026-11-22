"""
src/optimization/engine.py
==========================
Comprehensive Portfolio Optimization and Constraint Enforcement Engine.

Implements Master Implementation Plan Sections 7.2 & 7.3:
1. Traditional Markowitz (Sample Covariance)
2. Ledoit-Wolf Shrinkage Markowitz
3. ML-Driven Markowitz with Turnover Regularization: min -(mu^T w - rf) / sqrt(w^T Sigma w) + lambda * ||w - w_prev||_1
4. Minimum Variance Portfolio
5. Hierarchical Risk Parity (HRP)
6. Equal Risk Contribution (Risk Budgeting / ERC)
7. Configurable constraints:
   - Long-only: w_i >= 0
   - Fully invested: sum(w_i) == 1.0
   - Single-asset concentration cap: default 10% (configurable 10% to 30%)
   - Sector allocation cap: default 35% (configurable)
   - Turnover penalty (lambda * L1 norm)
8. Feasibility pre-check and documented graceful fallbacks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class PortfolioOptimizer:
    """
    Institutional portfolio optimization suite supporting SLSQP Markowitz,
    Minimum Variance, HRP, and Equal Risk Contribution with asset and sector caps.
    """

    def __init__(
        self,
        default_stock_cap: float = 0.10,
        default_sector_cap: float = 0.35,
        default_rf: float = 0.06,
        fallback_log: Optional[List[Dict[str, Any]]] = None
    ):
        self.default_stock_cap = default_stock_cap
        self.default_sector_cap = default_sector_cap
        self.rf = default_rf
        self.fallback_log = fallback_log if fallback_log is not None else []

    # ---------------------------------------------------------------------------
    # Constraint & Feasibility Helpers
    # ---------------------------------------------------------------------------

    def check_feasibility(
        self,
        n_assets: int,
        stock_cap: float,
        sector_cap: Optional[float] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        tickers: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Validate whether the given asset and sector caps can mathematically sum to 100%.
        """
        if n_assets * stock_cap < 0.999:
            return False, f"Infeasible: {n_assets} assets with max cap {stock_cap:.1%} can sum to at most {n_assets * stock_cap:.1%} < 100%."

        if sector_cap is not None and sector_mapping is not None and tickers is not None:
            # Check number of distinct sectors
            sectors = set(sector_mapping.get(t, "Unknown") for t in tickers)
            if len(sectors) * sector_cap < 0.999:
                return False, f"Infeasible: {len(sectors)} sectors with max sector cap {sector_cap:.1%} can sum to at most {len(sectors) * sector_cap:.1%} < 100%."

        return True, "Feasible"

    def _build_sector_constraints(
        self,
        tickers: List[str],
        sector_mapping: Dict[str, str],
        sector_cap: float
    ) -> List[Dict[str, Any]]:
        """
        Build linear inequality constraints for sector caps: sum_{i in sector} w_i <= sector_cap.
        """
        constraints = []
        sectors = set(sector_mapping.get(t, "Other") for t in tickers)

        for sec in sectors:
            indices = [i for i, t in enumerate(tickers) if sector_mapping.get(t, "Other") == sec]
            if len(indices) * (1.0 / len(tickers)) > sector_cap or True:
                # Add inequality: sector_cap - sum(w[indices]) >= 0
                def sec_constraint(w, idxs=indices, cap=sector_cap):
                    return cap - np.sum(w[idxs])
                constraints.append({"type": "ineq", "fun": sec_constraint})

        return constraints

    def cap_and_normalize(
        self,
        weights: np.ndarray,
        stock_cap: float = 0.10,
        tickers: Optional[List[str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        sector_cap: Optional[float] = None
    ) -> np.ndarray:
        """
        Iteratively project weights to satisfy non-negativity, stock cap, and sum-to-1.
        """
        w = np.clip(weights, 0.0, None)
        if np.sum(w) == 0:
            return np.ones(len(w)) / len(w)
        w = w / np.sum(w)

        # Iterative asset capping
        for _ in range(50):
            excess = np.sum(np.maximum(0.0, w - stock_cap))
            if excess < 1e-7:
                break
            w = np.minimum(w, stock_cap)
            under = w < stock_cap
            if np.sum(under) > 0:
                w[under] += excess * (w[under] / np.sum(w[under]) if np.sum(w[under]) > 0 else 1.0 / np.sum(under))
            else:
                break

        # Check and enforce sector caps if specified
        if sector_cap is not None and sector_mapping is not None and tickers is not None:
            for _ in range(20):
                violated = False
                for sec in set(sector_mapping.get(t, "Other") for t in tickers):
                    idxs = [i for i, t in enumerate(tickers) if sector_mapping.get(t, "Other") == sec]
                    sec_w = np.sum(w[idxs])
                    if sec_w > sector_cap + 1e-6:
                        violated = True
                        scale = sector_cap / sec_w
                        w[idxs] *= scale
                if violated:
                    w = np.minimum(w, stock_cap)
                    w = w / np.sum(w)
                else:
                    break

        return w / np.sum(w)

    # ---------------------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------------------

    def optimize_equal_weight(self, tickers: List[str]) -> Dict[str, float]:
        """Equal weight allocation (1/N)."""
        n = len(tickers)
        if n == 0:
            return {}
        w = 1.0 / n
        return {t: w for t in tickers}

    def optimize_momentum_top_k(
        self,
        momentum_scores: pd.Series,
        tickers: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Momentum Top-K Strategy (e.g. B4 Benchmark):
        Select top k assets by 12M minus 1M momentum scores and allocate equally (1/k each).
        """
        scores = momentum_scores.reindex(tickers).fillna(-999.0)
        top_k = scores.nlargest(k).index.tolist()
        w = {t: 0.0 for t in tickers}
        if len(top_k) > 0:
            weight_per_asset = 1.0 / len(top_k)
            for t in top_k:
                w[t] = weight_per_asset
        return w

    def optimize_min_variance(
        self,
        cov_matrix: pd.DataFrame,
        tickers: List[str],
        stock_cap: Optional[float] = None,
        sector_cap: Optional[float] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        date_str: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Minimum Variance Portfolio via SLSQP:
        min w^T Sigma w
        s.t. sum(w) == 1, 0 <= w_i <= stock_cap, sector_sum <= sector_cap
        """
        cap = stock_cap or self.default_stock_cap
        cov = cov_matrix.loc[tickers, tickers].values
        n = len(tickers)

        feasible, msg = self.check_feasibility(n, cap, sector_cap, sector_mapping, tickers)
        if not feasible:
            self._log_fallback(date_str, "Min_Variance", f"Infeasible constraints: {msg}", "Inverse-volatility fallback")
            return self._fallback_inverse_vol(cov, tickers, cap)

        w0 = np.ones(n) / n
        bounds = [(0.0, cap) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if sector_cap and sector_mapping:
            constraints.extend(self._build_sector_constraints(tickers, sector_mapping, sector_cap))

        def objective(w):
            return float(w @ cov @ w)

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 250})

        if res.success:
            w_opt = self.cap_and_normalize(res.x, cap, tickers, sector_mapping, sector_cap)
        else:
            self._log_fallback(date_str, "Min_Variance", f"SLSQP failure: {res.message}", "Inverse-volatility fallback")
            w_opt = self._fallback_inverse_vol(cov, tickers, cap, as_array=True)

        return dict(zip(tickers, [float(x) for x in w_opt]))

    def optimize_max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        tickers: List[str],
        stock_cap: Optional[float] = None,
        sector_cap: Optional[float] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        previous_weights: Optional[Dict[str, float]] = None,
        turnover_penalty_lambda: float = 0.0,
        date_str: Optional[str] = None,
        strategy_name: str = "Max_Sharpe"
    ) -> Dict[str, float]:
        """
        Max Sharpe / Expected Return Optimization via SLSQP with turnover penalty:
        min -(mu^T w - rf) / sqrt(w^T Sigma w) + lambda * ||w - w_prev||_1
        """
        cap = stock_cap or self.default_stock_cap
        cov = cov_matrix.loc[tickers, tickers].values
        mu = expected_returns.reindex(tickers).fillna(0.0).values
        n = len(tickers)

        feasible, msg = self.check_feasibility(n, cap, sector_cap, sector_mapping, tickers)
        if not feasible:
            self._log_fallback(date_str, strategy_name, f"Infeasible constraints: {msg}", "Min-variance fallback")
            return self.optimize_min_variance(cov_matrix, tickers, cap, sector_cap, sector_mapping, date_str)

        w0 = np.ones(n) / n
        bounds = [(0.0, cap) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if sector_cap and sector_mapping:
            constraints.extend(self._build_sector_constraints(tickers, sector_mapping, sector_cap))

        w_prev_arr = None
        if previous_weights and turnover_penalty_lambda > 0.0:
            w_prev_arr = np.array([previous_weights.get(t, 0.0) for t in tickers])

        def objective(w):
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(np.maximum(w @ cov @ w, 1e-8)))
            sharpe_term = -(port_ret - self.rf) / port_vol
            turnover_term = 0.0
            if w_prev_arr is not None:
                turnover_term = turnover_penalty_lambda * np.sum(np.abs(w - w_prev_arr))
            return sharpe_term + turnover_term

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 250})

        if res.success:
            w_opt = self.cap_and_normalize(res.x, cap, tickers, sector_mapping, sector_cap)
        else:
            self._log_fallback(date_str, strategy_name, f"SLSQP non-convergence: {res.message}", "Min-variance fallback")
            return self.optimize_min_variance(cov_matrix, tickers, cap, sector_cap, sector_mapping, date_str)

        return dict(zip(tickers, [float(x) for x in w_opt]))

    def optimize_equal_risk_contribution(
        self,
        cov_matrix: pd.DataFrame,
        tickers: List[str],
        stock_cap: Optional[float] = None,
        date_str: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Equal Risk Contribution (Risk Budgeting / ERC):
        Each asset contributes identically to total portfolio risk:
        w_i * (Sigma w)_i / (w^T Sigma w) == 1/N
        """
        cap = stock_cap or self.default_stock_cap
        cov = cov_matrix.loc[tickers, tickers].values
        n = len(tickers)
        target_rc = 1.0 / n

        w0 = np.ones(n) / n
        bounds = [(0.0, cap) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def erc_objective(w):
            port_vol = np.sqrt(np.maximum(w @ cov @ w, 1e-8))
            marginal_contrib = (cov @ w) / port_vol
            risk_contrib = w * marginal_contrib / port_vol
            # Minimize sum of squared deviations from target risk budget (1/N)
            return float(np.sum((risk_contrib - target_rc) ** 2))

        res = minimize(erc_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 250})
        if res.success:
            w_opt = self.cap_and_normalize(res.x, cap)
        else:
            self._log_fallback(date_str, "Risk_Budgeting_ERC", f"ERC non-convergence: {res.message}", "Inverse-volatility fallback")
            w_opt = self._fallback_inverse_vol(cov, tickers, cap, as_array=True)

        return dict(zip(tickers, [float(x) for x in w_opt]))

    def optimize_hrp(
        self,
        cov_matrix: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        stock_cap: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP) via Lopez de Prado algorithm.
        """
        t_list = tickers or cov_matrix.columns.tolist()
        cov = cov_matrix.loc[t_list, t_list]
        corr = cov.corr()

        # Distance matrix
        d_mat = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))
        condensed = squareform(d_mat, checks=False)
        link = linkage(condensed, method="single")

        sort_ix = self._get_quasi_diag(link)
        sorted_tickers = [t_list[i] for i in sort_ix]
        cov_sorted = cov.loc[sorted_tickers, sorted_tickers]

        # Recursive bisection
        hrp_weights = self._get_rec_bipart(cov_sorted, sorted_tickers)

        # Apply stock cap if requested
        cap = stock_cap or self.default_stock_cap
        w_dict = hrp_weights.to_dict()
        capped = self.cap_and_normalize(np.array([w_dict[t] for t in t_list]), cap)
        return dict(zip(t_list, [float(x) for x in capped]))

    # ---------------------------------------------------------------------------
    # HRP Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        link_int = link.astype(int)
        sort_ix = pd.Series([link_int[-1, 0], link_int[-1, 1]])
        num_items = link_int[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link_int[j, 0]
            df0 = pd.Series(link_int[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def _get_rec_bipart(cov: pd.DataFrame, sorted_tickers: List[str]) -> pd.Series:
        w = pd.Series(1.0, index=sorted_tickers)
        clusters = [sorted_tickers]

        while len(clusters) > 0:
            clusters = [
                c[start:end]
                for c in clusters
                for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                if len(c) > 1
            ]
            for i in range(0, len(clusters), 2):
                c1 = clusters[i]
                c2 = clusters[i + 1]
                v1 = PortfolioOptimizer._get_cluster_var(cov, c1)
                v2 = PortfolioOptimizer._get_cluster_var(cov, c2)
                alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
                w[c1] *= alpha
                w[c2] *= (1.0 - alpha)

        return w / w.sum()

    @staticmethod
    def _get_cluster_var(cov: pd.DataFrame, cluster: List[str]) -> float:
        cov_slice = cov.loc[cluster, cluster].values
        inv_diag = 1.0 / np.maximum(np.diag(cov_slice), 1e-8)
        w = inv_diag / np.sum(inv_diag)
        return float(w @ cov_slice @ w)

    # ---------------------------------------------------------------------------
    # Fallback and Logging
    # ---------------------------------------------------------------------------

    def _log_fallback(self, date_str: Optional[str], strategy: str, reason: str, action: str):
        self.fallback_log.append({
            "date": str(date_str)[:10] if date_str else "N/A",
            "strategy": strategy,
            "reason": reason,
            "fallback_action": action
        })

    def _fallback_inverse_vol(self, cov: np.ndarray, tickers: List[str], cap: float, as_array: bool = False):
        diag = np.diag(cov)
        inv_vol = 1.0 / np.sqrt(np.maximum(diag, 1e-6))
        w = inv_vol / np.sum(inv_vol)
        w_capped = self.cap_and_normalize(w, cap)
        if as_array:
            return w_capped
        return dict(zip(tickers, [float(x) for x in w_capped]))
