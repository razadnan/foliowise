# ML-Driven Portfolio Optimization for NIFTY 50 Stocks (Version 2.0)
## Comprehensive Viva & Oral Defense Preparation Pack

This document provides plain-language, rigorous academic answers to the viva examination questions for the Minor Project evaluation. All team members should review and understand these core concepts.

---

### Question 1: What is a Sharpe Ratio and why is it useful in portfolio management?
**Plain-Language Answer:**
The Sharpe ratio (William F. Sharpe, 1966) measures how much excess return an investment earns for every unit of total risk (annual volatility) it incurs:
$$\text{Sharpe Ratio} = \frac{\bar{r}_p - r_f}{\sigma_p}$$
where $\bar{r}_p$ is the annualized portfolio return, $r_f$ is the risk-free rate (fixed at 6.0% in our study, representing Indian 91-day Treasury bills), and $\sigma_p$ is the annualized standard deviation of daily portfolio returns.

**Why it is useful:**
- Comparing portfolios solely on raw return (CAGR) is misleading. An aggressive strategy might earn 20% return with 35% volatility and severe 50% drawdowns, whereas a defensive strategy might earn 16% return with only 15% volatility.
- The Sharpe ratio normalizes returns by their risk, allowing investors to evaluate whether extra return comes from genuine skill (alpha) or excessive risk exposure (beta).
- In our v2 study, Strategy S1 (Equal Weight) achieved a Sharpe ratio of **0.68** (CAGR 17.07%, Vol 16.34%), whereas the NIFTY 50 TRI benchmark achieved **0.41** (CAGR 13.27%, Vol 17.81%), proving that S1 delivered substantially higher risk-adjusted efficiency than the market index.

---

### Question 2: Why must we use walk-forward validation instead of a random train/test split?
**Plain-Language Answer:**
In standard machine learning (e.g., image classification or customer churn), data points are independent and identically distributed (i.i.d.), making random K-fold splits valid. In financial markets, data points are **time series** with strong temporal dependencies, autocorrelation, and non-stationarity.

**Why a random split is fatal:**
1. **Look-Ahead Contamination (Time Travel):** A random split puts data from Tuesday into the training set and Monday into the test set. The model memorizes future price shocks and uses them to "predict" the past.
2. **Autocorrelation & Overlapping Windows:** When target returns span 21 trading days, consecutive daily observations share 20 overlapping days. Randomly partitioning days between train and test leaks identical information across the boundary, yielding artificially inflated accuracy that collapses in real trading.
3. **Walk-Forward Emulates Real Life:** Our expanding-window walk-forward validation strictly mirrors reality: an investor trains models on historical data up to year-end $T-1$, freezes the model, and predicts the next year $T$ out-of-sample, before retraining with expanding history.

---

### Question 3: What is data leakage and how did our pipeline prove it was prevented?
**Plain-Language Answer:**
Data leakage occurs when any information from the future (or information that would not have been available at the decision timestamp) enters the feature engineering, model training, or portfolio optimization stage.

**Common forms of leakage we prevented:**
1. **Target Overlap Leakage:** Our target is the 21-day forward return ($P_{t+21}/P_t - 1$). If we train on data right up to December 31, the target for December 20 uses prices from January 20 of the next year! We enforced a **21-day target purge window** between the end of training data and the start of out-of-sample testing to ensure zero overlap.
2. **Global Preprocessing Leakage:** Standardizing features or calculating full-sample means/variances leaks future distribution parameters into past rows. In our pipeline, all `StandardScaler` transformations were fitted strictly inside the expanding training window.
3. **The Multi-Date Truncation Test:** In Stage 2, Student 2 proved zero leakage programmatically. We computed features on the full 2015–2026 dataset, and recomputed them on datasets physically truncated at 6 distinct historical dates (`2018-12-31`, `2019-12-31`, `2021-06-30`, `2022-12-30`, `2023-12-29`, `2024-12-31`). The feature values on the truncation date matched to 10 decimal places ($\Delta = 0.0000000000$), providing irrefutable mathematical proof of zero leakage.

---

### Question 4: Why does the covariance matrix matter, and why must it be shrunk (Ledoit-Wolf)?
**Plain-Language Answer:**
The covariance matrix ($\Sigma$) measures how each stock fluctuates relative to every other stock. It dictates how risks cancel out through diversification:
$$\sigma_p^2 = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{i,j} = w^T \Sigma w$$

**The Sample Covariance Problem:**
When calculating sample covariance from historical returns where the number of assets $N = 27$ is large relative to time observations $T = 252$, sample covariance contains immense estimation error. Extreme eigenvalues are pushed too high and too low. Inverting this matrix ($\Sigma^{-1}$) blows up noise, leading to extreme, unstable weights.

**Ledoit-Wolf Shrinkage Solution:**
Ledoit & Wolf (2004) proved that shrinking the sample covariance matrix towards a structured target (constant correlation matrix) minimizes quadratic Frobenius loss:
$$\hat{\Sigma}_{\text{LW}} = \delta F + (1 - \delta) S$$
This produces a well-conditioned, invertible covariance matrix, stabilizing portfolio weights across rebalancing cycles.

---

### Question 5: What is Next-Day Execution and why is it essential?
**Plain-Language Answer:**
In naive academic backtests, portfolio weights are calculated on the month-end closing price $T$, and trades are assumed to execute simultaneously at that exact same closing price. In reality, market participants cannot know the closing price until the session has already ended!

**Our Next-Day Execution Baseline (v2.0):**
1. **Signal Generation at $T$ Close:** On month-end date $T$, signals, ML forecasts, and optimal target weights $w^*_T$ are calculated after the market closes.
2. **Execution at $T+1$ Close:** Orders are executed at the closing price of the next trading day ($T+1$).
3. **Turnover & Frictions:** One-way turnover $0.5 \sum |w^*_T - w_{T+1-}|$ and transaction costs are assessed on $T+1$.
4. **Holding Returns:** The new weights earn returns from $T+1$ close to $T+2$ close. This completely eliminates execution look-ahead bias.

---

### Question 6: Why use the NIFTY 50 Total Return Index (TRI) instead of the standard price index?
**Plain-Language Answer:**
The standard NIFTY 50 price index measures only capital appreciation and drops cash dividend payments. In India, NIFTY 50 companies pay an average annual dividend yield of approximately 1.2% to 1.5%.
- Comparing equity portfolios (which naturally collect dividends) against a price-only index artificially inflates perceived alpha.
- Strategy S0 uses the **NIFTY 50 Total Return Index (TRI)**, which assumes all dividends are immediately reinvested into the index, establishing an honest, rigorous institutional benchmark.

---

### Question 7: Why did simple Equal Weight (S1) beat ML Max Sharpe (S4)?
**Plain-Language Answer:**
This is the classic DeMiguel, Garlappi, and Uppal (2009) empirical puzzle. 
1. **Estimation Error:** Machine learning models predict short-term return vectors with noisy out-of-sample $R^2$ (< 1%). Mean-variance optimizers act as "error maximizers," overweighting stocks with large positive forecast noise.
2. **Turnover Drag:** Mean-variance optimization requires active monthly portfolio adjustments (1.44% monthly one-way turnover in S4 vs 0.12% in S1). Cumulative transaction costs erode risk-adjusted returns.
3. **Structural Rebalancing Premia:** S1 automatically buys past losers (rebalancing after price drops) and trims past winners, capturing natural cross-sectional mean reversion with near-zero estimation risk.

---

### Question 8: Why did unconstrained Momentum (S7) fail while Ridge ML (S4/S5) survived?
**Plain-Language Answer:**
Strategy S7 (Top-5 Momentum based on 12M minus 1M price momentum) earned only **6.14% CAGR, Sharpe 0.01, and suffered a -40.48% drawdown**.
- Unregularized momentum suffers from **momentum crashes** during violent macroeconomic regime pivots (e.g. March 2020 COVID shock, late 2021 inflation shock). When market leadership rotates abruptly from tech/growth to value/defensive stocks, trailing 12-month winners plunge simultaneously.
- Ridge regression uses L2 shrinkage penalty and multi-factor features (volatility, technical indicators, liquidity, macro regime), mitigating concentration in overextended single-factor momentum bets.

---

### Question 9: What are the primary limitations of this study?
**Plain-Language Answer:**
1. **Constituent Survivorship Bias:** Our universe consists of 27 blue-chip equities present throughout 2015–2026. Companies removed from the index due to financial distress during this period (e.g., Yes Bank) were not included.
2. **Regime Specificity:** The 2020–2026 test period included extraordinary global liquidity, an intense bull run, and unprecedented retail market expansion in India. Results may differ during a multi-year secular bear market.
3. **Execution Capacity:** We model institutional fixed costs (20 bps round-trip), but do not simulate market impact or bid-ask bounce for large order sizes (AUM > ₹500 Crores).
4. **Short Test Horizon:** 6.8 years covers only one major market cycle (COVID shock → liquidity bull → tightening cycle). Multi-decade studies across diverse regimes are needed to generalize conclusions.
5. **Tax Omission:** Indian Short-Term Capital Gains tax (STCG at 20%) would further penalize high-turnover strategies (S5 at 2.14% monthly, S4 at 1.44%) relative to low-churn S1 and S6.

---

### Question 10: What do SHAP values and feature ablations tell us about model behavior?
**Plain-Language Answer:**
Feature importance is crucial for distinguishing *genuine* predictive signals from *coincidental* correlations in historical data.

**SHAP Attribution (Shapley Values):**
Using permutation-based Shapley attribution on our Gradient Boosting model, the ranked contributors to return predictions are:
1. `f_mom_126_21` — Medium-term price momentum (most critical)
2. `f_vol_21` — Short-term realized volatility
3. `f_mkt_vol_21` — 21-day NIFTY market volatility (macro regime)
4. `f_dist_52wh` — Proximity to 52-week high (overextension signal)
5. `f_rsi_14` — Technical RSI momentum oscillator

**Feature Group Ablation (Impact on Ridge IC and S4 Net Sharpe):**

| Dropped Group | Validation IC | S4 Net Sharpe | ΔSharpe | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| None (full 13 features) | +0.0382 | 0.46 | — | Baseline |
| Price Momentum | +0.0112 | 0.31 | −0.15 | Most vital group; removal severely impairs rank |
| Vol & Risk | +0.0321 | 0.40 | −0.06 | Impairs risk-adjusted position scaling |
| Volume Dynamics | +0.0375 | 0.45 | −0.01 | Minimal standalone contribution |
| Technical Overextension | +0.0298 | 0.38 | −0.08 | RSI + 52WH provide mean-reversion filter |
| Market Relative | +0.0340 | 0.42 | −0.04 | Beta adjustment stabilizes cross-sectional tilts |

**Key Insight:** Price momentum is the single dominant factor. The ablation result suggests that Ridge is essentially a regularized momentum model — the remaining 8 features provide incremental refinement but the core signal is medium-term price trend.
