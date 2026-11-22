"""
src/experiments/bootstrap.py
============================
Stationary Block Bootstrap Significance Testing for Financial Time Series.

Implements Politis & Romano (1994) Stationary Block Bootstrap:
- Preserves serial correlation and heteroskedasticity in daily asset returns.
- Random block lengths distributed geometrically with mean block length L (default L=10 days).
- Evaluates statistical significance of Sharpe ratio differences between strategies:
  Delta Sharpe = Sharpe(Strategy A) - Sharpe(Strategy B).
- Computes two-sided p-values and 95% non-parametric confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any


def compute_annualized_sharpe(returns: np.ndarray, rf: float = 0.06, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio from daily net return series.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean_ret = np.mean(returns) * periods_per_year
    vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    if vol <= 1e-8:
        return 0.0
    return float((mean_ret - rf) / vol)


def stationary_block_bootstrap_indices(n_obs: int, avg_block_len: float = 10.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate resampled indices using Politis & Romano (1994) stationary bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    p = 1.0 / max(avg_block_len, 1.0)
    indices = np.empty(n_obs, dtype=int)
    
    current_idx = rng.integers(0, n_obs)
    for i in range(n_obs):
        indices[i] = current_idx
        # With probability p start a new block, else continue contiguous block (wrapping around)
        if rng.random() < p:
            current_idx = rng.integers(0, n_obs)
        else:
            current_idx = (current_idx + 1) % n_obs
            
    return indices


class StationaryBootstrapTester:
    """
    Hypothesis tester for portfolio strategies using stationary block bootstrap.
    """

    def __init__(
        self,
        n_bootstraps: int = 1000,
        avg_block_length: float = 10.0,
        rf: float = 0.06,
        seed: int = 42
    ):
        self.n_bootstraps = n_bootstraps
        self.avg_block_length = avg_block_length
        self.rf = rf
        self.seed = seed

    def test_sharpe_difference(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        name_a: str = "Strategy A",
        name_b: str = "Strategy B"
    ) -> Dict[str, Any]:
        """
        Test H0: Sharpe(A) == Sharpe(B) vs H1: Sharpe(A) != Sharpe(B).
        
        Returns:
            Dictionary containing observed Sharpe ratios, difference, standard error,
            95% confidence intervals, and two-sided p-value.
        """
        assert len(returns_a) == len(returns_b), "Return series must be of identical length"
        n_obs = len(returns_a)
        
        sharpe_a = compute_annualized_sharpe(returns_a, self.rf)
        sharpe_b = compute_annualized_sharpe(returns_b, self.rf)
        observed_diff = sharpe_a - sharpe_b
        
        rng = np.random.default_rng(self.seed)
        boot_diffs = np.empty(self.n_bootstraps)
        
        for b in range(self.n_bootstraps):
            boot_idx = stationary_block_bootstrap_indices(n_obs, self.avg_block_length, rng=rng)
            sample_a = returns_a[boot_idx]
            sample_b = returns_b[boot_idx]
            
            b_sharpe_a = compute_annualized_sharpe(sample_a, self.rf)
            b_sharpe_b = compute_annualized_sharpe(sample_b, self.rf)
            boot_diffs[b] = b_sharpe_a - b_sharpe_b
            
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        std_err = float(np.std(boot_diffs, ddof=1))
        
        # Dual percentile bootstrap p-value (Efron & Tibshirani 1993, Davison & Hinkley 1997)
        # Exactly dual to the 95% percentile confidence interval: p < 0.05 <=> 0 not in [ci_lower, ci_upper]
        p_percentile = 2.0 * min(float(np.mean(boot_diffs >= 0.0)), float(np.mean(boot_diffs <= 0.0)))
        p_val = min(max(p_percentile, 1.0 / self.n_bootstraps), 1.0)
        
        # Centered bootstrap p-value for methodological comparison
        centered_diffs = boot_diffs - observed_diff
        p_centered = float(np.mean(np.abs(centered_diffs) >= np.abs(observed_diff)))
        p_centered = min(max(p_centered, 1.0 / self.n_bootstraps), 1.0)
        
        is_significant = bool(p_val < 0.05 and (ci_lower > 0.0 or ci_upper < 0.0))
        
        return {
            "strategy_a": name_a,
            "strategy_b": name_b,
            "sharpe_a": round(sharpe_a, 4),
            "sharpe_b": round(sharpe_b, 4),
            "observed_diff": round(observed_diff, 4),
            "std_error": round(std_err, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "p_value": round(p_val, 4),
            "p_value_centered": round(p_centered, 4),
            "significant_5pct": is_significant
        }
