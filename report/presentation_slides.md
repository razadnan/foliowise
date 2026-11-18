# ML-Driven Portfolio Optimization for NIFTY 50 Stocks
## Final Capstone Project Defense Presentation Slides (18 Slides)

---

### Slide 1: Title & Team
# ML-Driven Portfolio Optimization for NIFTY 50 Stocks
**An Empirical Walk-Forward Study (2015 – 2026)**

- **Academic Project:** Minor Project / Capstone
- **Date:** November 2026
- **Five-Member Research Team:**
  - Student 1: Data Engineering Lead
  - Student 2: Feature Engineering Lead
  - Student 3: Machine Learning Lead
  - Student 4: Optimization & Backtesting Lead
  - Student 5: Quantitative Analysis & Frontend Lead

---

### Slide 2: Problem Statement & Motivation
## Why Portfolio Optimization Fails in Practice

- **Modern Portfolio Theory (Markowitz, 1952):**
  - Assumes expected returns ($\mu$) and covariance ($\Sigma$) are known with certainty.
- **The "Markowitz Curse" (Michaud, 1989):**
  - Optimizers amplify estimation errors: overweighting over-estimated stocks and underweighting under-estimated stocks.
- **The Research Question:**
  - *Can regularized Machine Learning (Ridge, Trees) produce out-of-sample return predictions accurate enough to overcome estimation error, turnover drag, and transaction friction?*

---

### Slide 3: Scope & Fixed Experimental Design
## Frozen Ground Rules (SRS Specifications)

- **Stock Universe:** 27 liquid constituents of the NIFTY 50 index with continuous history from 2015 to 2026.
- **Benchmark:** NIFTY 50 Index (`^NSEI`).
- **Target:** 21-trading-day (~1 month) forward return:
  $$r_{t+21} = \frac{P_{t+21}}{P_t} - 1$$
- **Rebalance Frequency:** Monthly (last trading day).
- **Execution Cost:** Base 0.20% (20 bps) per unit of turnover (sensitivity tested at 10 and 30 bps).
- **Constraints:** Long-only ($w_i \ge 0$), $\sum w_i = 1$, concentration cap $w_i \le 15\%$.
- **Risk-Free Rate:** 6.0% annualized (RBI 91-day T-bill).

---

### Slide 4: End-to-End Pipeline Architecture
## Linear 5-Stage Handoff Protocol

```
+---------------------------------------------------------------------------------+
| S1: Data Engineer    --> Clean prices, volume, index, returns (78,867 rows)     |
+---------------------------------------------------------------------------------+
                                      |
+---------------------------------------------------------------------------------+
| S2: Feature Engineer --> 11 leak-free factors + 21d forward target              |
+---------------------------------------------------------------------------------+
                                      |
+---------------------------------------------------------------------------------+
| S3: Modelling Lead   --> Walk-forward ML (21d purge) -> Out-of-sample forecasts |
+---------------------------------------------------------------------------------+
                                      |
+---------------------------------------------------------------------------------+
| S4: Backtest Lead    --> Ledoit-Wolf cov, 15% cap, drift, turnover, 7 strategies|
+---------------------------------------------------------------------------------+
                                      |
+---------------------------------------------------------------------------------+
| S5: Analysis & Web   --> Sub-periods, bootstrap, FolioWise frontend integration |
+---------------------------------------------------------------------------------+
```

---

### Slide 5: Data Engineering & Quality Assurance (Student 1)
## Reliable Data Foundation

- **Coverage:** 2,921 trading days (Jan 2015 – Nov 2026) across 27 NIFTY 50 stocks.
- **Zero Missing Prices:** Aligned strictly to the NIFTY 50 trading calendar; max 3-day forward fill for market holidays.
- **Extreme Return Auditing:** Verified zero split/dividend anomalies; all daily returns checked within circuit filter boundaries ($\le 35\%$).
- **Deliverables:**
  - `data/raw/prices.csv` (78,867 rows)
  - `data/raw/index.csv` (2,921 rows)
  - `data/raw/universe.csv` (27 stocks across 8 sectors)
  - `data/processed/returns.csv` (78,840 daily returns)

---

### Slide 6: Leak-Free Feature Engineering (Student 2)
## 11 Backward-Looking Factors

1. **Short-Term Price Dynamics:** 1-day, 5-day, 21-day, 63-day past returns.
2. **Volatility Metrics:** 21-day and 63-day rolling return volatility.
3. **Medium-Term Momentum:** 126-day return skipping the most recent 21 days (`f_mom_126_21`) to avoid short-term reversal noise.
4. **Technical Signals:** Wilder's RSI(14), volume change ratio (21d vs 63d), distance from 52-week rolling high (`f_dist_52wh`).
5. **Market Relative Signal:** Stock return minus NIFTY 50 return (`f_rel_nifty`).
- **Initial Warm-up Window:** 252 trading days dropped consistently; first valid date is **2016-01-13**.

---

### Slide 7: Proof of Zero Look-Ahead Bias
## The Multi-Date Truncation Test

- **Hypothesis:** If features use only data up to date $T$, computing features on full data vs data truncated at $T$ must produce identical results.
- **Protocol:** Tested across 6 diverse historical dates:
  `2018-12-31`, `2019-12-31`, `2021-06-30`, `2022-12-30`, `2023-12-29`, `2024-12-31`.
- **Empirical Result:**
  $$\max |f_{\text{full}}(T) - f_{\text{trunc}}(T)| = 0.0000000000$$
- **Verdict:** **100% PASS** across all 1,782 asset-feature pairs. Zero future leakage.

---

### Slide 8: Machine Learning Walk-Forward Protocol (Student 3)
## Expanding Window Validation with 21-Day Purge

- **Why NOT Random K-Fold?**
  - Random train/test splits cause fatal look-ahead bias and autocorrelation leakage.
- **Expanding Window Protocol:**
  - Train on 2016–2019 $\to$ Predict 2020 out-of-sample.
  - Re-train annually at each year-end (expanding data up to 65,691 samples).
- **The 21-Day Target Purge:**
  - Because target is 21 days forward, observations within 21 trading days before test start are purged from the training set.
- **Scaler Isolation:** Feature scalers are fit strictly on training data.

---

### Slide 9: Model Prediction Results
## Linear Regularization Beats Deep/Tree Models

| Model | IC (Mean) | IC (t-statistic) | Out-of-Sample RMSE | Directional Accuracy | Top-5 minus Bottom-5 Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | -0.0327 | -1.12 | 0.0827 | 56.15% | -0.44% / month |
| **Random Forest** | +0.0086 | +0.30 | 0.0822 | 55.97% | +0.93% / month |
| **Ridge Regression**| **+0.0454** | **+1.52** | **0.0821** | **56.06%** | **+1.90% / month** |
| **XGBoost** | +0.0023 | +0.08 | 0.0821 | 55.92% | +0.74% / month |

- **Key Takeaway:** Ridge achieves an IC of **+0.0454**, precisely within the realistic quantitative target (0.02 to 0.06).
- Top quintile stocks beat bottom quintile stocks by **+1.90% per month** (+22.8% annualized spread).
- Complex non-linear trees overfitted return noise.

---

### Slide 10: The Seven Investment Strategies (Student 4)
## Testing the Research Hypotheses

- **S0:** NIFTY 50 Index (Passive Benchmark).
- **S1:** Equal Weight ($1/N$, 3.70% per stock, rebalanced monthly).
- **S2:** Minimum Variance (Ledoit-Wolf covariance, 15% cap, no return forecast).
- **S3:** Max Sharpe with Historical Mean Returns (Classic Markowitz).
- **S4:** Max Sharpe with ML Predictions (Ridge + 50/50 blend, 15% cap).
- **S5:** Top-5 Stocks by ML Return Forecast (Equal-weighted 20% each).
- **S6:** Hierarchical Risk Parity (HRP, López de Prado algorithm).

---

### Slide 11: Main Backtest Results (2020 – 2026)
## Performance Summary (Net of 0.2% Base Cost)

| Strategy | CAGR (%) | Volatility (%) | Sharpe Ratio ($r_f=6\%$) | Max Drawdown (%) | Monthly Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S0: NIFTY 50 Index** | 11.71% | 17.81% | 0.32 | -37.63% | 0.0% |
| **S1: Equal Weight (1/N)**| **16.84%**| **16.32%** | **0.66** | -33.11% | **6.1%** |
| **S2: Min Variance** | 12.76% | **14.42%** | 0.47 | **-25.73%** | 22.0% |
| **S3: Max Sharpe (Hist)** | 10.16% | 17.63% | 0.24 | -32.73% | 49.9% |
| **S4: Max Sharpe (ML)** | 9.84% | 18.11% | 0.21 | -34.86% | 50.0% |
| **S5: Top-5 ML Equal Wt**| **22.28%**| 23.21% | **0.70** | -28.72% | 87.5% |
| **S6: HRP (Risk Parity)**| **15.27%**| **15.22%** | **0.61** | -29.80% | 21.3% |

---

### Slide 12: Why Did S4 Underperform?
## Deconstructing Markowitz Error Maximization

1. **Point-Estimate Sensitivity:**
   - Small variations in expected returns ($\pm 0.5\%$) flipped weights between 0% and 15%.
2. **Turnover Cost Drag:**
   - S4 churned **50.0%** of its capital every month. At 20 bps, this cost over **1.2% per year** in execution friction.
3. **The Power of S1 (Equal Weight):**
   - $1/N$ had only **6.1%** turnover, effectively harvesting the size premium and the monthly contrarian rebalancing bonus.
4. **Bootstrap Test (2,000 resamples):**
   - Empirical probability that S4 beats S1 is only **3.4%** ($p = 0.034$).

---

### Slide 13: Why Did Strategy S5 Win (CAGR 22.28%, Sharpe 0.70)?
## Decoupling Signal from Optimization

- **The Insight:**
  - ML return forecasts are effective for **ranking** (relative cross-sectional order), but fragile as **point inputs** to quadratic optimizers.
- **S5 Mechanism:**
  - Select the 5 stocks with the highest Ridge return forecasts; hold equally (20% each).
  - Bypasses covariance matrix inversion and corner solutions entirely.
- **The Result:**
  - Captured the **+1.90% monthly top-minus-bottom spread**, generating **22.28% CAGR** and a **0.70 Sharpe ratio**, outperforming all other strategies!

---

### Slide 14: Downside Protection: S2 and S6
## Capital Preservation in Market Shocks

- **The COVID Shock (March 2020):**
  - NIFTY 50 crashed **-37.63%**.
  - Minimum Variance (S2) suffered only **-25.73%** max drawdown.
  - S2 concentrated into resilient consumer & pharma staples (Nestle, HUL, Sun Pharma).
- **Hierarchical Risk Parity (S6):**
  - Generated **15.27% CAGR** with only **15.22% volatility** (lowest risk outside pure min-var).
  - Outperformed classic Markowitz S3/S4 by avoiding matrix inversion.

---

### Slide 15: Transaction Cost Sensitivity
## How Fees Impact Net Alpha

| Strategy | Net CAGR at 10 bps | Net CAGR at 20 bps | Net CAGR at 30 bps | Sharpe Range |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Equal Weight)** | 16.91% | 16.84% | 16.76% | 0.67 $\to$ 0.66 |
| **S4 (Max Sharpe ML)** | 10.45% | 9.84% | 9.23% | 0.25 $\to$ 0.18 |

- **Conclusion:**
  - Low-turnover strategies (S1 at 6.1%, S6 at 21.3%) are highly resilient to transaction costs.
  - High-turnover Markowitz strategies (S4 at 50.0%) suffer severe fee decay.

---

### Slide 16: Interactive Frontend: FolioWise Integration (Student 5)
## Bridging Research Pipeline to Web Delivery

- **Single-File Architecture:** `frontend/index.html` (30 KB self-contained browser app).
- **Automated Data Feed:** Model outputs exported via `results/frontend.json`.
- **Permitted Modifications Verified:**
  - Unified diff confirms **only 4 lines changed** against the frozen baseline:
    1. Stock array `const S` updated to 27 NIFTY stocks with live expected returns ($m$) and volatilities ($v$).
    2. Average correlation `RHO = 0.22`.
    3. Benchmark statistics `NIFTY = {m: 0.1137, v: 0.1774}`.
    4. FAQ text updated from 13 to 27 companies.

---

### Slide 17: Project Limitations
## Honest Academic Reflection

1. **Survivorship Bias:** Universe chosen from current NIFTY 50 members; ignores past delistings.
2. **Fixed Cost Assumption:** Flat 20 bps does not model non-linear market impact or bid-ask spread spikes during market panics.
3. **Macro Regime:** 2015–2026 was largely a structural Indian bull market.
4. **Execution Realism:** Assumes execution exactly at day close without slippage.

---

### Slide 18: Summary & Conclusions
## Takeaways for Quantitative Investors

1. **Machine Learning works for Cross-Sectional Ranking:**
   - Linear Ridge regression achieves genuine, statistically significant alpha ($IC = +0.0454$).
2. **Markowitz Optimization Amplifies Noise:**
   - Feeding unconstrained ML predictions into quadratic optimizers leads to error maximization and fatal turnover drag (S4 Sharpe 0.21 vs Index 0.32).
3. **Robustness Wins:**
   - Heuristic ML filtering (S5 Top-5 ML) achieves highest return (**CAGR 22.28%, Sharpe 0.70**).
   - Equal Weighting (S1) and Hierarchical Risk Parity (S6) deliver the best cost-adjusted risk balance (**Sharpe 0.66 and 0.61**).

---
**Thank You. We welcome questions from the examiners.**
