# ML-Driven Portfolio Optimization for NIFTY 50 Stocks
## Final Capstone Project Defense Presentation Slides (18 Slides)

---

### Slide 1: Title & Team
# ML-Driven Portfolio Optimization for NIFTY 50 Stocks
**An Empirical Walk-Forward Study (2015 – 2026)**

- **Academic Project:** Minor Project / Capstone
- **Date:** November 2026
- **Five-Member Research Team:**
  - **Aarav Patel:** Student 1 (Data Engineering Lead)
  - **Ishaan Sharma:** Student 2 (Feature Engineering Lead)
  - **Riya Sen:** Student 3 (Machine Learning Lead)
  - **Adnan Raza:** Student 4 (Optimization & Backtesting Lead)
  - **Ananya Verma:** Student 5 (Quantitative Analysis & Frontend Lead)

---

### Slide 2: Problem Statement & Motivation
## Why Portfolio Optimization Fails in Practice

- **Modern Portfolio Theory (Markowitz, 1952):**
  - Assumes expected returns ($\mu$) and covariance ($\Sigma$) are known parameters.
- **The "Markowitz Curse" (Michaud, 1989):**
  - Mean-variance optimizers act as *estimation error maximizers*: overweighting assets with upward estimation noise and underweighting those with downward noise.
- **The 1/N Benchmark Paradox (DeMiguel et al., 2009):**
  - Naive equal-weighting beats optimizing models because it incurs zero estimation error and low turnover.
- **The Core Research Question:**
  - *Can regularized Machine Learning (Ridge, Trees) produce out-of-sample return predictions accurate enough to overcome estimation error, turnover drag, and transaction friction in real-world Indian markets?*

---

### Slide 3: Scope & Fixed Experimental Design
## Frozen Ground Rules (SRS Specifications)

- **Stock Universe:** 27 liquid constituents of the NIFTY 50 index with continuous history from 2015 to 2026.
- **Benchmark:** NIFTY 50 Index (`^NSEI`).
- **Target Variable:** 21-trading-day (~1 month) forward return:
  $$r_{t+21} = \frac{P_{t+21}}{P_t} - 1$$
- **Rebalance Frequency:** Monthly (last trading day; 83 test rebalance cycles).
- **Execution Cost:** Base 0.20% (20 bps) per one-way turnover (sensitivity tested at 10 and 30 bps).
- **Constraints:** Long-only ($w_i \ge 0$), $\sum w_i = 1$, concentration cap $w_i \le 15\%$ (S5 holds 5 stocks at 20%).
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
| S2: Feature Engineer --> 11 leak-free factors + 21d forward target (72,090 rows)|
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

- **Coverage:** 2,921 trading days (Jan 2, 2015 – Nov 13, 2026) across 27 NIFTY 50 stocks.
- **Calendar Alignment:** Reindexed strictly against NIFTY 50 (`^NSEI`) calendar to avoid date mismatches.
- **Gap Handling (FR-1.5):** Forward-fill max 3 consecutive days for trading halts; zero backward-fill to prevent lookahead bias.
- **Audit Verification:** Column `filled` explicitly records all filled observations. Zero non-positive prices; all price jumps checked within circuit bounds ($\le 35\%$).
- **Deliverables:** `prices.csv` (78,867 rows), `index.csv` (2,921 rows), `universe.csv` (27 stocks), `returns.csv` (78,840 rows).

---

### Slide 6: Leak-Free Feature Engineering (Student 2)
## 11 Backward-Looking Factors

1. **Short Returns:** 1-day, 5-day, 21-day, 63-day trailing returns.
2. **Volatility Metrics:** 21-day and 63-day rolling return standard deviation.
3. **Medium-Term Momentum:** 126-day return skipping the most recent 21 days (`f_mom_126_21`) to isolate intermediate trend from short-term reversal.
4. **Technical Signals:** Wilder's RSI(14), volume change ratio (21d vs 63d), distance from 52-week rolling high (`f_dist_52wh`).
5. **Market Relative Signal:** Stock daily return minus NIFTY 50 return (`f_rel_nifty`).
- **Warm-Up Window:** First 252 trading days dropped consistently; first valid date is **2016-01-13**.

---

### Slide 7: Proof of Zero Look-Ahead Bias
## Multi-Date Truncation Invariance Test

- **Hypothesis:** If features use only data up to date $T$, computing features on full data vs. data physically truncated at $T$ must produce identical numerical values.
- **Protocol:** Tested across 6 diverse historical cutoff dates:
  `2018-12-31`, `2019-12-31`, `2021-06-30`, `2022-12-30`, `2023-12-29`, `2024-12-31`.
- **Empirical Result:**
  $$\max |f_{\text{full}}(T) - f_{\text{trunc}}(T)| = 0.0000000000$$
- **Verdict:** **100% PASS** across all 1,782 tested asset-feature pairs. Zero future leakage.

---

### Slide 8: Machine Learning Walk-Forward Protocol (Student 3)
## Expanding Window Validation with 21-Day Purge

- **Why NOT Random K-Fold?**
  - Random train/test splits cause severe look-ahead bias and serial correlation leakage in financial time series.
- **Expanding Window Protocol:**
  - Initial train: 2016–2019 $\to$ Predict 2020 out-of-sample.
  - Re-train annually at each year-end (expanding dataset up to 65,691 samples by 2026).
- **The 21-Day Target Purge:**
  - Because target spans 21 trading days forward, observations within 21 trading days before test start are purged from the training set.
- **Scaler Isolation:** Feature scalers are fit strictly inside training windows.

---

### Slide 9: Model Prediction Results
## Realistic Machine Learning Signals

| Model | IC (Mean) | IC (t-statistic) | Out-of-Sample RMSE | Directional Accuracy | Top-5 minus Bottom-5 Spread |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline** | -0.0327 | -1.12 | 0.0827 | 56.15% | -0.44% / month |
| **Random Forest** | +0.0086 | +0.30 | 0.0822 | 55.97% | +0.93% / month |
| **Ridge Regression**| **+0.0454** | **+1.52** | **0.0821** | **56.06%** | **+1.90% / month** |
| **XGBoost** | +0.0112 | +0.41 | 0.0820 | 56.10% | +0.80% / month |

- **Statistical Interpretation:**
  - Ridge delivers an IC of **0.0454 (t = 1.52)**: positive, but **not statistically significant at the 5% level** ($t < 1.96$).
  - Meaningful cross-sectional sorting: Top quintile stocks beat bottom quintile stocks by **+1.90% per month** (+22.8% annualized).
  - Tree models (RF IC = +0.0086, XGB IC = +0.0112) overfit return noise.

---

### Slide 10: The Seven Investment Strategies (Student 4)
## Testing Alternative Allocation Frameworks

- **S0:** NIFTY 50 Index (Market Benchmark).
- **S1:** Equal Weight ($1/N$, 3.70% per stock, rebalanced monthly).
- **S2:** Minimum Variance (Ledoit-Wolf shrinkage covariance, 15% cap, no return forecast).
- **S3:** Max Sharpe with Historical Mean Returns (Classic Markowitz).
- **S4:** Max Sharpe with ML Predictions (Ridge + 50/50 blend, 15% cap).
- **S5:** Top-5 Stocks by ML Return Forecast (Equal-weighted 20% each).
- **S6:** Hierarchical Risk Parity (HRP, López de Prado algorithm).

---

### Slide 11: Main Backtest Results (2020 – 2026)
## Performance Summary (Net of 0.20% Base Cost)

| Strategy | Net CAGR (%) | Volatility (%) | Sharpe Ratio ($r_f=6\%$) | Max Drawdown (%) | Monthly Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S0: NIFTY 50 Index** | 11.71% | 17.81% | 0.32 | -37.63% | 0.0% |
| **S1: Equal Weight (1/N)**| **16.84%**| **16.32%** | **0.66** | -33.11% | **6.1%** |
| **S2: Min Variance** | 12.76% | **14.42%** | 0.47 | **-25.73%** | 22.0% |
| **S3: Max Sharpe (Hist)** | 10.16% | 17.63% | 0.24 | -32.73% | 49.9% |
| **S4: Max Sharpe (ML)** | 9.84% | 18.11% | 0.21 | -34.86% | 50.0% |
| **S5: Top-5 ML Equal Wt**| **22.28%**| 23.21% | **0.70** | -28.72% | **87.5%** |
| **S6: HRP (Risk Parity)**| **15.27%**| **15.22%** | **0.61** | -29.80% | 21.3% |

---

### Slide 12: Why Did S4 Underperform?
## Deconstructing Markowitz Error Maximization

1. **Point-Estimate Sensitivity:**
   - Small variations in expected return inputs flipped weights violently between 0% and 15% cap (`weights_over_time.png`).
2. **Turnover Cost Drag:**
   - S4 churned **50.0%** of its portfolio every month. At 20 bps, transaction drag cost over **1.2% annually**.
3. **The Unbeatable 1/N Baseline (S1):**
   - Incurred only **6.1%** turnover, capturing the size and contrarian rebalancing premia (DeMiguel et al., 2009).
4. **Bootstrap Significance Test (2,000 resamples):**
   - Mean Sharpe difference (S4 minus S1): **-0.385**; 95% CI is **[-0.812, +0.031]**.
   - *Because the 95% CI includes zero, we cannot conclude S1 beats S4 with 95% confidence; however, S4 is not better, and is probably worse ($P(\text{S4} > \text{S1}) = 3.4\%$)*.

---

### Slide 13: Strategy S5: Promising but Unproven
## High Nominal Alpha vs. Real-World Execution Friction

- **High Nominal Performance:**
  - S5 achieved **22.28% CAGR** and a **Sharpe ratio of 0.70** (vs 0.66 for S1).
  - Captures the **+1.90% monthly top-minus-bottom spread** by holding top 5 stocks at 20% each.
- **Why It Remains Unproven:**
  1. **Massive Churn:** Average monthly turnover of **87.5%** — liquidating and replacing almost the entire portfolio every 30 days.
  2. **Cap Violation:** Holds 20% per stock, exceeding the 15% concentration cap enforced on other strategies.
  3. **No Significance Test:** Higher Sharpe ratio is accompanied by high volatility (23.21%) and lacks a formal paired bootstrap significance test against S1.

---

### Slide 14: Downside Protection: S2 and S6
## Capital Preservation in Market Shocks

- **The COVID Shock (March 2020):**
  - NIFTY 50 crashed **-37.63%**.
  - Minimum Variance (S2) suffered only **-25.73%** max drawdown.
  - S2 concentrated into low-beta defensive consumer & pharma staples (Nestle, HUL, Sun Pharma).
- **Hierarchical Risk Parity (S6):**
  - Generated **15.27% CAGR** with only **15.22% volatility** (lowest risk outside pure min-var).
  - Outperformed classic Markowitz S3/S4 by avoiding covariance matrix inversion.

---

### Slide 15: Transaction Cost Sensitivity
## How Trading Friction Impacts Net Alpha

| Strategy | Net CAGR at 10 bps | Net CAGR at 20 bps | Net CAGR at 30 bps | Sharpe Range |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Equal Weight)** | 16.91% | 16.84% | 16.76% | 0.67 $\to$ 0.66 |
| **S4 (Max Sharpe ML)** | 10.45% | 9.84% | 9.23% | 0.25 $\to$ 0.18 |

- **Conclusion:**
  - Low-turnover strategies (S1 at 6.1%, S6 at 21.3%) are highly resilient to transaction costs.
  - High-turnover Markowitz strategies (S4 at 50.0%) suffer severe fee decay (61 bps loss per 10 bps fee increase).

---

### Slide 16: Interactive Frontend: FolioWise Integration (Student 5)
## Bridging Research Pipeline to Web Delivery

- **Single-File Architecture:** `frontend/index.html` (30 KB self-contained browser app).
- **Automated Data Feed:** Generated via `src/build_frontend.py` reading `results/frontend.json`.
- **Verification Diff (`baseline_diff.txt`):** **Only 3 lines changed** in the original baseline:
  1. Line 173: FAQ text updated from "13 large companies" to "27 large companies".
  2. Line 179: `const S` updated with 27 real stock parameters ($m, v$).
  3. Line 180: `const RHO=0.2200, NIFTY={m:0.1137, v:0.1774}` updated with backtest constants.
- **Known Differences Stated (SRS §3.3):**
  - Uses heuristic scoring rule rather than quadratic SLSQP optimizer.
  - Cap defaults to 20% on the page (vs 15% in backtest).
  - Demonstrates model outputs for education; backtest remains the empirical proof.

---

### Slide 17: Limitations & Threats to Validity
## Rigorous Academic Constraints

1. **Short Test Horizon:** 6-year backtest (2020–2026) covers one major market cycle; longer multi-decade historical tests are needed.
2. **No Capital Gains Taxes:** Modeling Indian Short-Term Capital Gains (STCG at 20%) would further penalize high-turnover S5 (87.5%) and S4 (50.0%) compared to S1 (6.1%).
3. **Survivorship Bias:** Universe selected from current NIFTY 50 constituents with unbroken history.
4. **Execution Slippage:** Backtest assumes execution at official day-close without market impact.

---

### Slide 18: Summary & Conclusions
## Takeaways for Quantitative Practitioners

1. **Regularized ML Generates Modest Rank Alpha:**
   - Linear Ridge regression achieves an IC of **+0.0454 (t = 1.52)**: positive, but not statistically significant at the 5% level.
2. **Mean-Variance Optimization Amplifies Noise:**
   - Feeding ML forecasts into Markowitz optimizers leads to error maximization and fatal turnover friction (S4 Sharpe 0.21 vs S0 Benchmark 0.32).
3. **S5 is Promising but Unproven:**
   - Top-5 ML equal-weighting captures strong alpha (Sharpe 0.70 vs 0.66 for S1), but 87.5% turnover makes execution fragile.
4. **Simplicity and Robustness Win:**
   - 1/N Equal Weighting (S1, Sharpe 0.66) and Hierarchical Risk Parity (S6, Sharpe 0.61) deliver the best cost-adjusted risk balance in Indian equities.

---
**Thank You. We welcome questions from the examination committee.**
