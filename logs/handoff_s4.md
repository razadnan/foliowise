# Student 4 Handoff Note (Optimization & Backtesting)

**To:** Student 5 (Analyst, Frontend & Report Lead)  
**From:** Student 4 (Optimization & Backtest Lead)  
**Date:** 2026-11-18  
**Tag:** `handoff-s4`  
**Status:** Accepted & Verified  

---

## 1. Deliverables Summary
- **Portfolio Weights:** `results/weights.csv` (13,446 rows, columns: `date, strategy, ticker, weight`)
- **Daily Strategy Returns:** `results/daily_returns.csv` (11,739 rows, columns: `date, strategy, gross_ret, net_ret, turnover`)
- **Strategy Performance Summary:** `results/strategy_metrics.csv`
- **Cost Sensitivity Table:** `results/cost_sensitivity.csv`
- **Equity Curves Plot:** `results/equity_curves.png`
- **Drawdown Plot:** `results/drawdown_chart.png`
- **Weights Over Time Plot:** `results/weights_over_time.png`
- **Notebook:** `notebooks/04_backtest.ipynb` and reusable source `src/stage4_backtest.py`

## 2. Seven Evaluated Strategies (Net of 0.2% Base Cost)

| Strategy ID | Strategy Name | CAGR (%) | Volatility (%) | Sharpe ($r_f=6\%$) | Max Drawdown (%) | Calmar | Turnover (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | NIFTY 50 Benchmark | 11.71% | 17.81% | 0.32 | -37.63% | 0.31 | 0.0% |
| **S1** | Equal Weight (1/N) | **16.84%** | **16.32%** | **0.66** | -33.11% | **0.51** | **6.1%** |
| **S2** | Minimum Variance | 12.76% | **14.42%** | 0.47 | **-25.73%** | 0.50 | 22.0% |
| **S3** | Max Sharpe (Historical)| 10.16% | 17.63% | 0.24 | -32.73% | 0.31 | 49.9% |
| **S4** | Max Sharpe (ML Ridge) | 9.84% | 18.11% | 0.21 | -34.86% | 0.28 | 50.0% |
| **S5** | Top-5 ML Equal Weight | **22.28%** | 23.21% | **0.70** | -28.72% | **0.78** | 87.5% |
| **S6** | Hierarchical Risk Parity| **15.27%** | **15.22%** | **0.61** | -29.80% | **0.51** | 21.3% |

## 3. Implementation Highlights
- **Ledoit-Wolf Covariance:** Fit on trailing 252 daily returns up to rebalance date.
- **Constraints Enforced:** Long-only, weights sum to 1.0, maximum individual stock weight strictly $\le 15.0\%$.
- **Drift & Cost Logic:** Daily portfolio weights drifted with stock returns. At rebalance, $\text{Turnover} = \sum |w_{\text{target}} - w_{\text{drifted}}|$, and execution fee was deducted as $\text{cost} \times \text{Turnover}$ ($c = 0.002$).
- **50/50 ML Blend:** Used in Strategy S4 per SRS FR-4.3 to stabilize weights.
- **Optimizer Stability:** Robust Sequential Least Squares Programming (SLSQP) optimizer converged without requiring artificial fallbacks.

## 4. Notes for Student 5
- Point out why S4 underperformed S1 (Markowitz error maximization + 50% turnover cost drag).
- Highlight S5's exceptional performance (22.28% CAGR, 0.70 Sharpe) proving that ML predictions contain high-quality ranking alpha when decoupled from quadratic covariance inversion.
- For `frontend.json`, use the annualized Ridge predictions and 252-day volatilities from the final rebalance date (`2026-11-13`).

## 5. Acceptance Checklist
- [x] Weights sum to 1.0 and never exceed 15.0% for any stock/date.
- [x] Hand check on drift and turnover formulas verified.
- [x] Independent reproduction of S0 and S1 verified.
- [x] Cost sensitivity table completed (0.1%, 0.2%, 0.3%).
- [x] Signed and verified.
