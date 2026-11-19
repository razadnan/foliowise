# ML-Driven Portfolio Optimization for NIFTY 50 Stocks

**Final Academic Research Project Report**  
**Course:** Minor Project / Capstone Portfolio Optimization  
**Date:** November 2026  
**Target Benchmark:** NIFTY 50 (`^NSEI`)  
**Evaluation Horizon:** 2015 – 2026 (Out-of-sample Walk-Forward Test: 2020 – 2026)  

---

## Executive Summary

This project conducts an empirical investigation into whether machine learning return forecasts can generate superior, cost-adjusted risk-weighted performance when integrated into classic and modern portfolio optimization frameworks for Indian equities (NIFTY 50 universe). Using daily data from January 2015 to November 2026 across 27 blue-chip Indian stocks, we enforce strict methodological safeguards against look-ahead bias and target contamination via expanding-window walk-forward validation and a 21-day target purge.

We compare seven distinct investment strategies:
- **S0:** NIFTY 50 Index (Market Benchmark)
- **S1:** Equal Weight (1/N, 3.70% per stock)
- **S2:** Minimum Variance (Ledoit-Wolf shrinkage covariance, 15% concentration cap)
- **S3:** Max Sharpe with Historical Mean Returns (Classic Markowitz)
- **S4:** Max Sharpe with ML Returns (Ridge regression with 50/50 shrinkage blend)
- **S5:** Top-5 Stocks by ML Return Forecast (Equal-weighted 20% each)
- **S6:** Hierarchical Risk Parity (HRP, López de Prado algorithm)

### Key Empirical Findings
1. **Model Predictive Power:** Linear Ridge regression achieved a mean Information Coefficient (IC) of **+0.0454 (t = 1.52)** across 83 monthly out-of-sample cycles—positive, but not statistically significant at the 5% level ($t < 1.96$). Tree-based models demonstrated weak signal capture on short-horizon monthly returns: Random Forest achieved an IC of **+0.0086 (t = 0.30)**, while XGBoost achieved an IC of **+0.0112 (t = 0.41)**.
2. **The "Markowitz Curse" & Error Maximization:** In the mean-variance optimizer (S4), feeding ML predictions directly caused extreme sensitivity to noisy point estimates. S4 generated **9.84% CAGR** and a **Sharpe ratio of 0.21**, lagging both the passive benchmark S0 (**11.71% CAGR, Sharpe 0.32**) and simple Equal Weight S1 (**16.84% CAGR, Sharpe 0.66**). Monthly turnover of 50.0% created heavy cumulative drag from transaction costs (0.20% base fee).
3. **S5 (Top-5 ML Equal Weight): Promising but Unproven:** When ML predictions were used simply to rank and filter the cross-section into a concentrated top-5 portfolio without covariance optimization, S5 achieved **22.28% CAGR** and a **Sharpe ratio of 0.70** (compared to 0.66 for S1). However, S5 remains *promising but unproven*: it suffers from massive monthly turnover of **87.5%**, has no formal significance test against S1, holds 20% per stock (exceeding the standard 15% cap), and exhibits elevated annual volatility (23.21%).
4. **Equal-Weight and Risk-Parity Dominance:** The 1/N Equal Weight strategy (S1) compounded at **16.84% CAGR** with a **Sharpe ratio of 0.66** and minimal turnover (6.1%), confirming the classic DeMiguel et al. (2009) puzzle. Hierarchical Risk Parity (S6) achieved **15.27% CAGR** and **Sharpe 0.61** with an annual volatility of only **15.22%** (vs 17.81% for NIFTY 50). Minimum Variance (S2) restricted maximum drawdown to **-25.73%** during the March 2020 COVID shock, compared to **-37.63%** for the benchmark index.
5. **Statistical Significance of S4 vs. S1:** A stationary bootstrap hypothesis test ($N = 2,000$) on the Sharpe difference ($\text{Sharpe}_{\text{S4}} - \text{Sharpe}_{\text{S1}}$) yielded a 95% confidence interval of $[-0.812, +0.031]$. Because the 95% CI includes zero, we cannot conclude S1 beats S4 with 95% confidence; however, S4 is clearly not better than S1, and is probably worse ($P(\text{S4} > \text{S1}) = 3.4\%$).

---

## 1. Introduction & Research Problem

### 1.1 Context and Motivation
Portfolio optimization is a cornerstone of modern financial engineering. Since Harry Markowitz formulated Modern Portfolio Theory (MPT) in 1952, the mean-variance framework has served as the theoretical standard for asset allocation. Markowitz demonstrated that an investor can construct an efficient frontier of portfolios offering the highest expected return for a defined level of variance.

However, in practical asset management, MPT encounters severe empirical obstacles:
- **Estimation Risk:** Sample mean returns ($\hat{\mu}$) are notoriously noisy. Historical sample means calculated over trailing lookbacks rarely persist out of sample. As shown by Merton (1980), accurately estimating expected returns requires decades of stationary data, which rarely exist in dynamic markets.
- **Error Maximization (Michaud, 1989):** Mean-variance optimizers act as "estimation error maximizers." They overweight assets with large positive estimation errors and underweight assets with negative estimation errors, creating extreme, concentrated, and unstable portfolio allocations.
- **Transaction Cost Friction:** Optimizers that react aggressively to small perturbations in return forecasts churn holdings continuously. Trading commissions, bid-ask spreads, and securities transaction taxes quickly erode theoretical gains.

### 1.2 Core Research Question
Recent quantitative literature explores using regularized machine learning (Ridge regression, Random Forests, Gradient Boosting) to forecast cross-sectional returns. This project empirically answers:
> *Can machine learning models predict cross-sectional forward returns of NIFTY 50 stocks with sufficient precision to overcome estimation error, turnover drag, and transaction friction in real-world portfolio optimization?*

### 1.3 Scope & Operational Boundaries
- **Universe:** 27 highly liquid constituents of the NIFTY 50 index with continuous, unbroken daily trading histories from January 2015 to November 2026.
- **Benchmark:** NIFTY 50 Index (`^NSEI`).
- **Rebalance Frequency:** Monthly (last trading day of each calendar month; 83 rebalance cycles).
- **Investment Horizon:** 21 trading days (~1 calendar month) forward return prediction:
  $$r_{t+21} = \frac{P_{t+21}}{P_t} - 1$$
- **Risk-Free Rate:** Fixed at 6.0% annualized (reflecting Reserve Bank of India 91-day Treasury yields).
- **Out-of-Sample Test Period:** January 2020 through November 2026 (spanning the COVID-19 crash, 2020-2021 recovery, 2022 global inflation shock, and 2023–2026 structural market expansions).
- **Execution Cost:** Base transaction cost of 0.20% (20 bps) per one-way turnover, with sensitivity evaluated at 0.10% (10 bps) and 0.30% (30 bps).
- **Constraints:** Long-only ($w_i \ge 0$), fully invested ($\sum w_i = 1$), per-stock concentration cap $w_i \le 0.15$ (15%) for S2, S3, S4, and S6. S5 holds 5 stocks at 20% each.

---

## 2. Literature Review & Theoretical Foundations

### 2.1 Classic Mean-Variance Optimization (Markowitz, 1952)
Markowitz formulated portfolio selection as a quadratic optimization program:
$$\min_{w} \quad \frac{1}{2} w^T \Sigma w \quad \text{subject to} \quad w^T \mu \ge \mu_0, \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$
When maximizing the Sharpe ratio, the objective seeks weights along the capital allocation line:
$$\max_{w} \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$
Best and Grauer (1991) proved that portfolio weights are an order of magnitude more sensitive to estimation errors in expected returns $\mu$ than to errors in the covariance matrix $\Sigma$. A 1% change in expected return can swing optimal weights by 30% to 50%, producing extreme portfolio turnover.

### 2.2 Performance Measurement (Sharpe, 1966)
William F. Sharpe (1966) established the Sharpe ratio as the standard risk-adjusted performance measure for investment portfolios:
$$\text{SR} = \frac{R_p - R_f}{\sigma_p}$$
Sharpe demonstrated that superior nominal returns often merely reflect levered risk exposures. Evaluating portfolios requires adjusting returns by annual volatility $\sigma_p$. In our study, excess returns are measured above India's prevailing 6.0% risk-free rate, and statistical differences between portfolio Sharpe ratios are subjected to bootstrap hypothesis testing.

### 2.3 The 1/N Benchmark Puzzle (DeMiguel, Garlappi, & Uppal, 2009)
In a seminal empirical study across 14 portfolio optimization models, DeMiguel, Garlappi, and Uppal (2009) found that *no single optimizing model consistently beat the naive 1/N equal-weight portfolio out of sample*. They proved theoretically that for a 50-asset portfolio, an estimation window of over 3,000 months (250 years) would be required for sample-based mean-variance optimization to overcome estimation error. The 1/N strategy succeeds because it incurs zero estimation error, incurs low turnover (rebalancing only to restore weights after price drift), and systematically exploits size and contrarian mean-reversion premia.

### 2.4 Covariance Shrinkage (Ledoit & Wolf, 2004)
Sample covariance matrices $S = \frac{1}{T} \sum (r_t - \bar{r})(r_t - \bar{r})^T$ are ill-conditioned when the number of assets $N$ is of comparable order to observations $T$. Inverting ill-conditioned covariance matrices amplifies extreme eigenvalues. Ledoit and Wolf (2004) introduced an optimal linear shrinkage estimator:
$$\hat{\Sigma}_{\text{LW}} = \delta F + (1 - \delta) S$$
where $F$ is a structured shrinkage target (constant correlation matrix) and $\delta \in [0, 1]$ is estimated analytically to minimize expected quadratic Frobenius loss. This ensures well-conditioned risk estimates and invertible matrices.

### 2.5 Machine Learning in Asset Pricing (Gu, Kelly, & Xiu, 2020)
Gu, Kelly, and Xiu (2020) conducted an extensive study of machine learning for asset return prediction:
- In financial returns, the signal-to-noise ratio is extraordinarily low. Complex, unconstrained models readily overfit noise.
- Linear regularized models (Ridge, Lasso) and shallow trees consistently match or outperform deep neural architectures on cross-sectional equity returns.
- Out-of-sample $R^2$ values rarely exceed 1% to 2%, but an Information Coefficient (IC) of 0.02 to 0.05 is economically meaningful for cross-sectional ranking.

### 2.6 Hierarchical Risk Parity (López de Prado, 2016)
Marcos López de Prado (2016) introduced Hierarchical Risk Parity (HRP) to address the instability of quadratic programming without requiring covariance matrix inversion ($\Sigma^{-1}$). HRP combines unsupervised hierarchical clustering, matrix reordering (quasi-diagonalization), and recursive bisection:
1. Distance metric: $d_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}$.
2. Hierarchical tree clustering groups collinear assets together.
3. Inverse-variance allocation is performed top-down across branches, preventing collinear clusters from dominating portfolio weights.

---

## 3. Data & Feature Engineering

### 3.1 Universe Selection & Data Quality
The candidate universe was selected from the NIFTY 50 index. To avoid look-ahead calendar mismatches, all series were strictly aligned against the NIFTY 50 index calendar (`^NSEI`, 2,921 trading days from January 2, 2015 to November 13, 2026). 

Twenty-seven stocks met all inclusion criteria (continuous trading history, zero negative prices, no unfillable gaps > 3 days):
- **Banking & Financials (6):** HDFC Bank, ICICI Bank, Axis Bank, Kotak Mahindra Bank, Bajaj Finance, Bajaj Finserv.
- **Information Technology (5):** TCS, Infosys, Wipro, HCL Technologies, Tech Mahindra.
- **Energy & Utilities (5):** Reliance Industries, NTPC, Power Grid, ONGC, Coal India.
- **Consumer Goods & FMCG (5):** Hindustan Unilever, ITC, Nestle India, Titan, Asian Paints.
- **Industrials & Auto (3):** Maruti Suzuki, Larsen & Toubro, UltraTech Cement.
- **Pharma & Telecom (3):** Sun Pharma, Dr. Reddy's Laboratories, Bharti Airtel.

```
Total Downloaded Trading Days: 2,921 (Jan 2, 2015 to Nov 13, 2026)
Total Clean Stock Price Rows:   78,867 (27 stocks x 2,921 days)
Total Daily Return Rows:        78,840
Missing Data Handling:          Forward-fill max 3 days per FR-1.5; zero backward-fill
Filled Observations Flagged:    Boolean column `filled` preserved in data/raw/prices.csv
Price Verification:             Zero non-positive prices; all daily price moves <= 35%
```

### 3.2 Leak-Free Feature Definitions
Eleven backward-looking predictive features were engineered using information available strictly at market close on date $t$:

| Feature | Code Name | Lookback | Concept / Mathematical Formulation |
| :--- | :--- | :---: | :--- |
| **Short Return** | `f_ret_1` | 1 Day | $P_t / P_{t-1} - 1$ |
| **Weekly Return** | `f_ret_5` | 5 Days | $P_t / P_{t-5} - 1$ |
| **Monthly Return** | `f_ret_21` | 21 Days | $P_t / P_{t-21} - 1$ |
| **Quarterly Return**| `f_ret_63` | 63 Days | $P_t / P_{t-63} - 1$ |
| **Short Volatility**| `f_vol_21` | 21 Days | Sample standard deviation of daily returns over trailing 21 days |
| **Long Volatility** | `f_vol_63` | 63 Days | Sample standard deviation of daily returns over trailing 63 days |
| **Momentum** | `f_mom_126_21`| 126d (skip 21)| $P_{t-21} / P_{t-126} - 1$ (skips reversal month to isolate intermediate trend) |
| **RSI** | `f_rsi_14` | 14 Days | Wilder's smoothed Relative Strength Index $\in [0, 100]$ |
| **Volume Surge** | `f_volchg` | 21d vs 63d | $\frac{\text{MeanVol}_{21}}{\text{MeanVol}_{63}} - 1$ |
| **52-Week High Dist**| `f_dist_52wh`| 252 Days | $P_t / \max(P_{t-252:t}) - 1 \le 0$ (measures proximity to yearly peak) |
| **Relative NIFTY** | `f_rel_nifty` | 1 Day | $r_{\text{stock}, t} - r_{\text{NIFTY}, t}$ |

The forward prediction target is defined as the 21-trading-day future return:
$$\text{target\_21d}_t = \frac{P_{t+21}}{P_t} - 1$$
The dataset comprises 72,090 valid feature rows after discarding the initial 252-day warm-up period.

### 3.3 Leakage Verification (Truncation Invariance Test)
To guarantee absence of look-ahead bias, a programmatic truncation test evaluated features across six diverse historical dates:
$$\{2018\text{-}12\text{-}31, 2019\text{-}12\text{-}31, 2021\text{-}06\text{-}30, 2022\text{-}12\text{-}30, 2023\text{-}12\text{-}29, 2024\text{-}12\text{-}31\}$$
For each cutoff date $T_k$, features were computed on the full dataset and independently on the dataset physically truncated at $T_k$. Across all 1,782 tested asset-feature pairs, the maximum absolute difference was:
$$\max |f_{\text{full}}(T_k) - f_{\text{trunc}}(T_k)| = 0.0000000000$$
confirming 100% exact numerical identity and zero future data leakage.

---

## 4. Methodology & Walk-Forward Validation

### 4.1 Walk-Forward Protocol with 21-Day Purge Window
Standard K-fold cross-validation is mathematically invalid in time series forecasting due to serial correlation and target contamination. We implement an **expanding-window walk-forward validation** scheme:
- **Initial Training Set:** 2016-01-13 to 2019-11-26 (25,569 samples).
- **Out-of-Sample Test Years:** Re-trained annually at year-end to generate predictions for 2020, 2021, 2022, 2023, 2024, 2025, and 2026.
- **21-Day Target Purge:** Because the target spans 21 trading days forward, observations within 21 trading days prior to each training boundary are purged. This prevents any training target from overlapping with the test period.
- **Feature Scaling:** `StandardScaler` is fit exclusively on training data and applied out-of-sample to test data.

```
Walk-Forward Expanding Schedule:
[Train 2016-2019 (25,569 rows)] -> [Purge 21d] -> [Test Year 2020: 12 rebalance cycles]
[Train 2016-2020 (32,346 rows)] -> [Purge 21d] -> [Test Year 2021: 12 rebalance cycles]
[Train 2016-2021 (38,988 rows)] -> [Purge 21d] -> [Test Year 2022: 12 rebalance cycles]
[Train 2016-2022 (45,684 rows)] -> [Purge 21d] -> [Test Year 2023: 12 rebalance cycles]
[Train 2016-2023 (52,353 rows)] -> [Purge 21d] -> [Test Year 2024: 12 rebalance cycles]
[Train 2016-2024 (58,968 rows)] -> [Purge 21d] -> [Test Year 2025: 12 rebalance cycles]
[Train 2016-2025 (65,691 rows)] -> [Purge 21d] -> [Test Year 2026: 11 rebalance cycles]
Total Test Rebalances: 83 monthly cycles
```

### 4.2 Machine Learning Models
Four models of increasing complexity were trained and evaluated:
1. **Naive Baseline:** Cross-sectional ticker-level historical mean return over the training window.
2. **Ridge Regression (L2):** Linear shrinkage model with regularization penalty $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0, 1000.0\}$ chosen by internal 3-fold cross-validation on the training set.
3. **Random Forest:** Ensemble of 100 shallow trees (`max_depth = 3`, `min_samples_leaf = 20`, `random_state = 42`) to constrain variance.
4. **XGBoost:** Gradient boosted trees (`n_estimators = 100`, `max_depth = 3`, `learning_rate = 0.03`, `subsample = 0.8`, `colsample_bytree = 0.8`).

### 4.3 Portfolio Rebalancing & Cost Mechanics
At each month-end rebalance date $t$:
1. Covariance matrix $\hat{\Sigma}_{\text{LW}}$ is estimated using Ledoit-Wolf shrinkage on the trailing 252 daily returns.
2. Target weights $w_t^*$ are computed under constraints: $w_i \ge 0$, $\sum w_i = 1$, and $w_i \le 0.15$ (except S5 which holds 5 equal weights of 20%).
3. Between rebalances, weights drift naturally with asset returns:
   $$w_{i, \tau+1} = \frac{w_{i, \tau} (1 + r_{i, \tau})}{1 + r_{\text{portfolio}, \tau}}$$
4. On rebalance date $t$, turnover is computed against drifted weights:
   $$\text{Turnover}_t = \sum_{i=1}^N |w_{i, t}^* - w_{i, t}^{\text{drifted}}|$$
5. Net portfolio return on rebalance day reflects transaction friction:
   $$r_{\text{net}, t} = r_{\text{gross}, t} - c \cdot \text{Turnover}_t \quad (c = 0.0020)$$

---

## 5. Empirical Results & Findings

### 5.1 Machine Learning Predictive Performance
Evaluating predictions across 83 out-of-sample monthly rebalance dates (8,964 individual asset forecasts) yields the following performance metrics:

| Model | IC (Mean) | IC (t-stat) | Out-of-Sample RMSE | Directional Accuracy | Top-5 minus Bottom-5 Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | -0.0327 | -1.12 | 0.0827 | 56.15% | -0.44% / month |
| **Random Forest** | +0.0086 | +0.30 | 0.0822 | 55.97% | +0.93% / month |
| **Ridge Regression**| **+0.0454** | **+1.52** | **0.0821** | **56.06%** | **+1.90% / month** |
| **XGBoost** | +0.0112 | +0.41 | 0.0820 | 56.10% | +0.80% / month |

#### Critical Interpretation of Model Metrics
- **Ridge Regression:** Delivers an IC of **0.0454 (t = 1.52)**. While economically positive and consistent with empirical asset pricing literature (Gu et al., 2020), a t-statistic of 1.52 indicates that the rank correlation is **positive but not statistically significant at the 5% level** ($t_{\text{crit}} = 1.96$ for two-tailed, or $1.66$ for one-tailed). The top-quintile minus bottom-quintile spread of **+1.90% per month** demonstrates meaningful cross-sectional sorting ability.
- **Tree-Based Models (RF and XGBoost):** Random Forest (IC = +0.0086, t = 0.30) and XGBoost (IC = +0.0112, t = 0.41) achieved marginal predictive correlation. Non-linear tree splits overfit the high noise floor of monthly equity returns.

#### Feature Importance Analysis
Inspection of the model feature importance diagnostics (`results/feature_importance.png`) reveals clear patterns:
- **Ridge Coefficients:** Dominated positively by `f_mom_126_21` (intermediate momentum) and `f_dist_52wh` (distance from 52-week high), and negatively by short-term reversal (`f_ret_1`) and long volatility (`f_vol_63`).
- **Random Forest & XGBoost Importances:** Split heavily on `f_dist_52wh` (proximity to 52-week high) and `f_rsi_14`, followed by `f_mom_126_21`. However, tree algorithms partitioned these continuous features into discrete step functions that failed to generalize as effectively as linear shrinkage.

---

### 5.2 Strategy Performance Comparison (2020 – 2026)
All performance metrics are net of 0.20% (20 bps) transaction costs across the complete 83-cycle out-of-sample backtest:

| Strategy ID | Strategy Name | Net CAGR (%) | Annual Volatility (%) | Sharpe Ratio ($r_f=6\%$) | Max Drawdown (%) | Calmar Ratio | Avg Monthly Turnover (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | NIFTY 50 Benchmark | 11.71% | 17.81% | 0.32 | -37.63% | 0.31 | 0.0% |
| **S1** | Equal Weight (1/N) | **16.84%** | **16.32%** | **0.66** | -33.11% | **0.51** | **6.1%** |
| **S2** | Minimum Variance | 12.76% | **14.42%** | 0.47 | **-25.73%** | 0.50 | 22.0% |
| **S3** | Max Sharpe (Historical) | 10.16% | 17.63% | 0.24 | -32.73% | 0.31 | 49.9% |
| **S4** | Max Sharpe (ML Ridge) | 9.84% | 18.11% | 0.21 | -34.86% | 0.28 | 50.0% |
| **S5** | Top-5 ML Equal Weight | **22.28%** | 23.21% | **0.70** | -28.72% | **0.78** | **87.5%** |
| **S6** | Hierarchical Risk Parity| **15.27%** | **15.22%** | **0.61** | -29.80% | **0.51** | 21.3% |

---

### 5.3 Detailed Breakdown & Mechanism Analysis

#### Why Did S4 (Max Sharpe ML) Fail? (CAGR 9.84%, Sharpe 0.21)
Strategy S4 blended ML return predictions with historical returns ($50/50$) and passed them into a numerical SLSQP optimizer to maximize the Sharpe ratio subject to a 15% concentration cap. S4 lagged behind both the passive index S0 (Sharpe 0.32) and S1 (Sharpe 0.66):
1. **Estimation Error Magnification:** Even with L2 shrinkage and a 50/50 blend, small estimation errors in expected return inputs drastically distorted optimal portfolio weights. The optimizer pushed stocks with marginally higher return predictions to the 15% boundary while zeroing out others, concentrating risk into assets whose forecasts had the largest upward noise.
2. **Turnover Friction:** S4 experienced an average monthly turnover of **50.0%**. Churning half the portfolio each month incurred substantial drag from the 0.20% transaction cost, reducing annualized net performance by over 1.2%.
3. **The Strength of 1/N:** S1 achieved a turnover of only **6.1%**, capturing the rebalancing premium across 27 diversified large caps without any parameter estimation risk.

#### S5 (Top-5 ML Equal Weight): Promising but Unproven (Sharpe 0.70 vs 0.66)
Strategy S5 simply selected the top 5 stocks ranked by Ridge predicted return and allocated 20% to each. While S5 generated the highest nominal CAGR (**22.28%**) and Sharpe ratio (**0.70**), it must be classified as **promising but unproven**:
- **Massive Turnover:** S5 turned over an average of **87.5% per month**, requiring almost the entire portfolio to be liquidated and replaced every 30 days. In live trading, market impact, bid-ask spreads, and securities transaction taxes (STT) would severely punish such high churn.
- **Concentration Violation:** S5 holds 20% per stock, exceeding the 15% concentration cap enforced on S2, S3, S4, and S6. Five equal-weighted stocks mathematically cannot satisfy a 15% cap ($\frac{1}{5} = 0.20$).
- **No Formal Significance Test:** Unlike S4 vs. S1, S5 was not subjected to a paired bootstrap significance test. Its higher Sharpe (0.70 vs 0.66 for S1) comes with higher volatility (23.21% vs 16.32%) and may reflect sample luck in a concentrated subset of bull-market leaders.

#### Weight Stability Analysis (`results/weights_over_time.png`)
Inspection of the asset allocations over time reveals the root cause of S4's underperformance:
- **Allocation Volatility:** S4 exhibits violent weight oscillations between consecutive months. Individual stocks frequently transition from 0% to the 15% maximum cap and back to 0% within a single rebalance cycle.
- **Factor Concentration:** During 2020–2021, S4 concentrated heavily in high-beta IT and pharma exporters; by 2022–2023, it rotated abruptly into energy and state-owned enterprises (Coal India, NTPC). Because these shifts were driven by point return forecasts with an IC of only 0.045, the portfolio suffered repeated whip-saws.

---

### 5.4 Sub-Period Regime Analysis (FR-5.1)

| Market Regime | Period | NIFTY 50 (S0) | Equal Weight (S1) | Min Variance (S2) | Max Sharpe ML (S4) | Top-5 ML (S5) | HRP (S6) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID Crash & Recovery** (2020) | 11 mos | +14.9% (Sh 0.35) | **+23.4%** (Sh 0.60) | +19.8% (Sh 0.64) | +16.2% (Sh 0.38) | **+31.2%** (Sh 0.78) | +21.1% (Sh 0.61) |
| **Post-COVID Bull & Volatility** (2021–22)| 24 mos | +12.1% (Sh 0.37) | **+18.7%** (Sh 0.74) | +13.2% (Sh 0.52) | +10.4% (Sh 0.25) | **+24.1%** (Sh 0.72) | +16.4% (Sh 0.68) |
| **Recent Market Regime** (2023–26) | 46 mos | +10.8% (Sh 0.28) | **+14.5%** (Sh 0.61) | +10.4% (Sh 0.40) | +8.1% (Sh 0.17) | **+19.8%** (Sh 0.66) | +13.5% (Sh 0.56) |

Across all three regimes, S1 and S6 consistently outpaced the market benchmark (S0), while S4 consistently underperformed due to turnover drag and error magnification.

---

### 5.5 Transaction Cost Sensitivity Analysis (FR-4.9)

| Strategy | Fee Regime | Net CAGR (%) | Annual Volatility (%) | Sharpe Ratio ($r_f=6\%$) | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1 (Equal Weight)** | 0.10% (10 bps) | 16.91% | 16.32% | 0.67 | -33.09% |
| **S1 (Equal Weight)** | 0.20% (20 bps) | 16.84% | 16.32% | 0.66 | -33.11% |
| **S1 (Equal Weight)** | 0.30% (30 bps) | 16.76% | 16.32% | 0.66 | -33.13% |
| **S4 (Max Sharpe ML)** | 0.10% (10 bps) | 10.45% | 18.11% | 0.25 | -34.72% |
| **S4 (Max Sharpe ML)** | 0.20% (20 bps) | 9.84% | 18.11% | 0.21 | -34.86% |
| **S4 (Max Sharpe ML)** | 0.30% (30 bps) | 9.23% | 18.11% | 0.18 | -35.01% |

Because S1 incurs only 6.1% monthly turnover, its net performance is virtually invariant to transaction fees (Sharpe 0.67 $\to$ 0.66). Conversely, S4's 50.0% turnover causes rapid decay: each 10 bps increase in execution costs strips 61 bps of annualized compound return.

---

### 5.6 Bootstrap Significance Test (FR-5.5)
To rigorously test whether Strategy S1 genuinely outperforms Strategy S4, we conducted a stationary bootstrap hypothesis test ($N = 2,000$ resamples of identical duration):
- **Mean Sharpe Difference ($\text{Sharpe}_{\text{S4}} - \text{Sharpe}_{\text{S1}}$):** $-0.385$
- **95% Confidence Interval:** $[-0.812, +0.031]$
- **Empirical Probability $P(\text{Sharpe}_{\text{S4}} > \text{Sharpe}_{\text{S1}})$:** $3.4\%$

**Statistical Verdict:**
Because the 95% confidence interval $[-0.812, +0.031]$ includes zero, we cannot conclude at the strict 5% level that S1 beats S4. However, S4 is definitely not better than S1, and is probably worse ($P(\text{S4} > \text{S1}) = 3.4\%$). In practical asset management, there is no empirical justification for accepting S4's complexity and turnover when S1 provides higher returns with lower variance.

---

## 6. Frontend Integration (FolioWise)

### 6.1 Architecture & Verification Diff
The quantitative pipeline directly connects to the interactive client application **FolioWise** (`frontend/index.html`). The integration adheres strictly to SRS Section 3.3:
1. `src/stage5_analysis.py` exports `results/frontend.json` containing live model parameters:
   - 27 company profiles ($m$, $v$, sector).
   - NIFTY 50 empirical benchmark: $m = 11.37\%$, $v = 17.74\%$.
   - Average pairwise correlation: $\rho = 0.22$.
2. `src/build_frontend.py` compiles `frontend/index.html` from `frontend/FolioWise_baseline.html`.
3. **Verification Diff:** A unified diff (`frontend/baseline_diff.txt`) confirms that **only 3 lines were changed** in the original baseline:
   - Line 173: FAQ text updated from "13 large companies" to "27 large companies".
   - Line 179: `const S=[...]` array updated with 27 real stock parameters.
   - Line 180: `const RHO=0.2200, NIFTY={m:0.1137, v:0.1774}` updated with backtest constants.
   - The demo banner, styles, and calculation scripts remain completely untouched.

### 6.2 Known Differences Between Frontend and Backtest (SRS §3.3)
As mandated by SRS Section 3.3, the following known differences must be recognized:
1. **Allocation Engine:** The web page uses a simplified scoring rule that tilts toward higher expected return $m$ and lower volatility $v$ based on the user's selected comfort level, followed by iterative cap normalization. It does not run quadratic SLSQP optimization or Ledoit-Wolf shrinkage, and assumes a constant pairwise correlation ($\rho = 0.22$).
2. **Concentration Cap Default:** The web interface defaults to a 20% cap (with 10%, 15%, and 30% options available in the menu), whereas the backtest strictly enforced 15% (except S5).
3. **Growth Projection:** Growth cones displayed on the web page are simplified parametric log-normal projections ($m \pm z \cdot v$), not historical backtest paths.
4. **Purpose:** The web page serves as an educational illustration of model outputs for end users, while the backtest represents the formal empirical evidence. The demo banner and risk disclaimer remain visible on the live page.

---

## 7. Limitations & Threats to Validity

1. **Short Test Horizon (6 Years):** The out-of-sample backtest covers January 2020 to November 2026. While event-rich (including COVID-19 and post-pandemic inflation), a 6-year window represents approximately one full market cycle and cannot guarantee performance across multi-decade secular regimes.
2. **No Capital Gains Taxes:** The backtest accounts for trading transaction costs (0.20%) but does not model India's Short-Term Capital Gains (STCG, 20%) or Long-Term Capital Gains (LTCG, 12.5%) tax regimes. Strategies with extreme turnover (S5 at 87.5%, S4 at 50.0%) would suffer severe tax friction in taxable accounts compared to low-turnover S1 (6.1%).
3. **Survivorship Bias:** The stock universe was derived from current NIFTY 50 constituents with continuous trading history from 2015. Companies that were demoted, delisted, or restructured during this decade were excluded, imparting a positive survivorship bias to all strategies.
4. **Execution Slippage:** Orders are assumed to execute at official closing prices without market impact or bid-ask spread expansion during liquidity crises.

---

## 8. Conclusion & Future Work

### 8.1 Final Summary
This project built a reproducible, leak-free quantitative pipeline for NIFTY 50 equities. The empirical evidence provides clear answers:
1. **Regularized ML achieves positive but modest predictability:** Ridge regression produced an IC of +0.0454 (t = 1.52), useful for cross-sectional ranking but not significant at the 5% level.
2. **Mean-variance optimization fails in practice:** Feeding ML predictions into Markowitz optimizers (S4) amplifies estimation noise and generates excessive turnover drag, lagging both the market index and simple equal weighting.
3. **S5 is promising but unproven:** Ranking into a top-5 equal-weight portfolio captured strong nominal alpha (Sharpe 0.70) but incurred extreme turnover (87.5%) and lacks statistical validation.
4. **Heuristic simplicity wins:** 1/N Equal Weighting (S1, Sharpe 0.66) and Hierarchical Risk Parity (S6, Sharpe 0.61) remain the most robust methods for compounding capital in Indian equities.

### 8.2 Future Directions
- **Turnover Regularization:** Incorporating turnover penalty terms directly into the objective function ($\min w^T \Sigma w - \lambda_1 w^T \mu + \lambda_2 \|w - w_{\text{prev}}\|_1$).
- **Black-Litterman Shrinkage:** Blending ML factor views with the market equilibrium portfolio to avoid corner solutions.
- **Alternative Data Integration:** Incorporating quarterly corporate earnings momentum, institutional ownership changes, and order-flow imbalance.

---

## 9. References

1. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *The Review of Financial Studies*, 22(5), 1915–1953.
2. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223–2273.
3. Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
4. López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample. *The Journal of Portfolio Management*, 42(4), 59–69.
5. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77–91.
6. Merton, R. C. (1980). On Estimating the Expected Return on the Market: An Exploratory Investigation. *Journal of Financial Economics*, 8(4), 323–361.
7. Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized' Optimal? *Financial Analysts Journal*, 45(1), 31–42.
8. Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119–138.

---

## Appendix: Author Contributions & Team Sign-Off

| Role | Student Name | Main Output Deliverables | Sign-Off Status |
| :--- | :--- | :--- | :--- |
| **Student 1** | Aarav Patel (Data Engineer) | Data acquisition, calendar alignment, returns computation, `prices.csv` with `filled` flag, `index.csv`, `returns.csv`, `universe.csv` | Signed & Verified |
| **Student 2** | Ishaan Sharma (Feature Engineer) | 11 backward-looking features, 21d forward target, 6-date truncation leakage test, `features.csv`, `leakage_test_report.csv` | Signed & Verified |
| **Student 3** | Riya Sen (Modelling Lead) | Expanding-window walk-forward validation with 21d purge, Ridge, RF, XGBoost models, `predictions.csv`, `model_metrics.csv` | Signed & Verified |
| **Student 4** | Adnan Raza (Optimization & Backtest Lead) | Ledoit-Wolf shrinkage, SLSQP 15% capped optimizer, HRP algorithm, daily backtest with drift and turnover cost, `weights.csv`, `strategy_metrics.csv` | Signed & Verified |
| **Student 5** | Ananya Verma (Analyst & Report Lead) | Sub-period analysis, bootstrap hypothesis testing, `frontend.json`, FolioWise 3-line diff verification, project report, slides, and viva pack | Signed & Verified |
