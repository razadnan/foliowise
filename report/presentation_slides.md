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
## 13 Backward-Looking Factors (v2.0)

1. **Short Returns:** 1-day, 5-day, 21-day, 63-day trailing returns.
2. **Volatility Metrics:** 21-day and 63-day rolling return standard deviation.
3. **Medium-Term Momentum:** 126-day return skipping the most recent 21 days (`f_mom_126_21`) to isolate intermediate trend from short-term reversal.
4. **Technical Signals:** Wilder's RSI(14), volume change ratio (21d vs 63d), distance from 52-week rolling high (`f_dist_52wh`).
5. **Market Relative Signal:** Stock daily return minus NIFTY 50 return (`f_rel_nifty`).
6. **Macro Regime Features (v2.0 additions):** 21-day NIFTY rolling volatility (`f_mkt_vol_21`); 63-day rolling Pearson correlation of stock to NIFTY (`f_mkt_corr_63`).
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

### Slide 9: Model Prediction Results (v2.0 Walk-Forward)
## Realistic Machine Learning Signals Across 83 Test Months

| Model Architecture | Mean IC | IC t-stat (OLS) | IC t-stat (Newey-West) | ICIR | Directional Acc (%) | Top - Bottom Spread (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | **+0.0382** | 1.24 | **1.55** | **0.138** | **56.29** | **+1.38% / mo** |
| **Gradient Boosting** | +0.0006 | 0.02 | 0.02 | 0.002 | 55.60 | +0.47% / mo |
| **Momentum Baseline** | -0.0113 | -0.36 | -0.35 | -0.040 | 50.53 | -0.04% / mo |
| **Random Forest** | -0.0161 | -0.54 | -0.59 | -0.060 | 54.82 | +0.81% / mo |
| **Neural Network (MLP)**| -0.0171 | -0.57 | -0.62 | -0.064 | 53.04 | +0.71% / mo |
| **Naive Baseline** | -0.0327 | -1.12 | -1.11 | -0.124 | 56.15 | -0.44% / mo |

- **Key Takeaways:**
  - Ridge regression delivers the highest and most consistent Information Coefficient (+0.0382, Newey-West $t=1.55$).
  - Complex non-linear neural networks and unconstrained momentum fail to generate positive out-of-sample rank correlation, validating regularized linear shrinkage.

---

### Slide 10: The Eight Investment Strategies (Student 4)
## Testing Alternative Allocation Frameworks (S0 to S7)

- **S0:** NIFTY 50 Total Return Index (TRI Benchmark including reinvested dividends).
- **S1:** Equal Weight ($1/N$, 3.70% per stock, rebalanced monthly).
- **S2:** Minimum Variance (Ledoit-Wolf shrinkage covariance, 15% cap, no return forecast).
- **S3:** Max Sharpe with Historical Mean Returns (Classic Markowitz, 15% cap).
- **S4:** Max Sharpe with ML Predictions (Ridge + 50/50 blend, 15% cap).
- **S5:** Top-5 Stocks by ML Return Forecast (Equal-weighted 20% each).
- **S6:** Hierarchical Risk Parity (HRP with Ledoit-Wolf covariance, 15% cap).
- **S7:** Top-5 Stocks by Momentum (12M minus 1M price momentum, equal-weighted 20% each).

---

### Slide 11: Main Backtest Results (2020 – 2026)
## Performance Summary (Next-Day Execution, Net of 20 bps Base Cost)

| Strategy | Net CAGR (%) | Volatility (%) | Sharpe Ratio ($r_f=6\%$) | Sortino Ratio | Max Drawdown (%) | One-Way Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0: NIFTY 50 TRI** | 13.27% | 17.81% | 0.41 | 0.58 | -37.54% | 0.00% |
| **S1: Equal Weight** | **17.07%** | 16.34% | **0.68** | **0.99** | -33.18% | 0.12% |
| **S2: Min Variance** | 13.21% | **14.51%** | 0.50 | 0.72 | **-25.94%** | 0.51% |
| **S3: Max Sharpe (Hist)**| 9.24% | 17.68% | 0.18 | 0.26 | -33.40% | 1.20% |
| **S4: Max Sharpe (ML)** | 13.89% | 17.13% | 0.46 | 0.65 | -29.02% | 1.44% |
| **S5: Top-5 ML EW** | **24.00%** | 23.34% | **0.77** | **1.14** | -28.52% | 2.14% |
| **S6: HRP (Ledoit-Wolf)**| 16.09% | 15.18% | **0.66** | **0.96** | -29.80% | 0.48% |
| **S7: Top-5 Momentum** | 6.14% | 21.30% | 0.01 | 0.01 | -40.48% | 1.24% |

---

### Slide 12: Why Did S4 Underperform?
## Deconstructing Markowitz Error Maximization

1. **Point-Estimate Sensitivity:**
   - Small variations in expected return inputs flipped weights violently between 0% and 15% cap (`weights_over_time.png`).
2. **Turnover Cost Drag:**
   - S4 churned **1.44%** (one-way) of its portfolio every month. At 20 bps cost rate, transaction drag costs ~**0.69% annually**.
3. **The Unbeatable 1/N Baseline (S1):**
   - Incurred only **0.12%** one-way turnover, capturing the size and contrarian rebalancing premia (DeMiguel et al., 2009).
4. **Bootstrap Significance Test (2,000 resamples, 21-day blocks):**
   - Mean Sharpe difference (S4 minus S1): **−0.202**; 95% CI: **[−0.607, +0.145]**.
   - $P(\text{S4} > \text{S1}) = 13.3\%$. **Hypothesis that ML mean-variance beats equal-weighting is rejected.**

---

### Slide 13: Strategy S5 & S7: Contrasting Outcomes
## High Nominal Alpha vs. Catastrophic Momentum Failure

**S5 (Top-5 ML Equal Weight) — Promising but Costly:**
- Achieved **24.00% CAGR** and a **Sharpe ratio of 0.77** (highest of all strategies).
- Captures the **+1.38% monthly top-minus-bottom spread** by holding top 5 ML-ranked stocks at 20% each.
- **Why It Remains Exploratory:**
  1. **Concentration Risk:** Annual volatility of 23.34% and tracking error of 12.18%.
  2. **High Churn:** 2.14% monthly one-way turnover (~25.7% annualized); cost-sensitive.
  3. **Cap Exception:** 20% per stock vs 15% enforced on S2–S4 and S6.
  4. **No Formal Significance Test Against S1** — high Sharpe is accompanied by high vol.

**S7 (Top-5 Momentum Equal Weight) — Catastrophic:**
- **6.14% CAGR, Sharpe 0.01, Max Drawdown −40.48%** (worst of all strategies).
- Momentum crashes during the March 2020 COVID shock and 2021 inflation rotation wiped out accumulated alpha.
- Underscores the necessity of L2 regularization (Ridge) and multi-factor diversification over naive trailing momentum.

---

### Slide 14: Downside Protection: S2 and S6
## Capital Preservation in Market Shocks

- **The COVID Shock (March 2020):**
  - NIFTY 50 TRI crashed **−37.54%** peak-to-trough.
  - Minimum Variance (S2) suffered only **−25.94%** max drawdown (best of all strategies).
  - S2 concentrated into low-beta defensive consumer & pharma staples (Nestle, HUL, Sun Pharma).
- **Hierarchical Risk Parity (S6, Ledoit-Wolf):**
  - Generated **16.09% CAGR** with only **15.18% volatility** (lowest excluding min-var).
  - Outperformed classic Markowitz S3/S4 by avoiding covariance matrix inversion altogether.
- **Crisis Regime Performance (σ_NIFTY > 20%):**
  - S2 Sharpe: **0.58**, S6 Sharpe: **0.51** — both dominated S4 (0.18) in crisis conditions.
  - S7 Momentum collapsed to **−1.20% CAGR** (Sharpe −0.35) during crisis periods.

---

### Slide 15: Transaction Cost Sensitivity
## How Trading Friction Impacts Net Alpha

| Strategy | Net Sharpe at 10 bps | Net Sharpe at 20 bps | Net Sharpe at 30 bps | Sharpe Erosion (10→30 bps) |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Equal Weight)** | 0.69 | 0.68 | 0.68 | −0.01 |
| **S2 (Min Variance)** | 0.51 | 0.50 | 0.49 | −0.02 |
| **S4 (ML Max Sharpe)** | 0.48 | 0.46 | 0.43 | **−0.05** |
| **S5 (Top-5 ML)** | 0.80 | 0.77 | 0.74 | **−0.06** |
| **S7 (Momentum)** | 0.03 | 0.01 | −0.01 | **−0.04** |

- **Conclusion:**
  - Low-turnover strategies (S1 at 0.12%, S6 at 0.48% one-way) are highly resilient to transaction costs.
  - S4 (1.44% one-way) erodes ~0.05 Sharpe per 10 bps cost increase; S5 at 2.14% turnover loses ~0.06 per 10 bps step.
  - S7 turns net-negative Sharpe at 30 bps — entirely unviable for retail or institutional execution.

---

### Slide 16: Interactive Frontends: FolioWise (Student 5)
## Two-View Architecture Bridging Research to Users

**Investor View (`frontend/index.html`):**
- Disclaimer banner, inline validation (no alert() pop-ups), "What do these numbers mean?" glossary.
- Honest testing card: Equal-split 16.8% vs ML-guided 9.8% vs NIFTY 50 11.7% — verdict stated plainly.
- Uses heuristic scoring rule (not SLSQP) — this is explicitly declared on the page.
- PDF print layout ("Save as PDF" button).

**Developer Research Dashboard / Admin Panel (`frontend/dashboard.html`):**
- Styled with Coinora coral orange aesthetic and light/dark theme toggle.
- 5 tabs: Overview & Strategy Ladder (E0–E6), Allocation & PIT Universe, Performance vs NIFTY, Costs & Turnover, Model Governance.
- Interactive rebalance action with visual optimizer state, tab switching, and custom toast notification.
- Visual access barrier with session persistence and transparent academic demo security disclosures.
- Cross-linked with Investor View via header and sidebar navigation.

**Single Source of Truth:** Both views read `results/frontend.json`; they cannot disagree after a rerun.

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
   - Linear Ridge regression achieves an IC of **+0.0382 (Newey-West t = 1.55)**: positive, consistent, but not statistically significant at the 5% level.
   - Non-linear models (MLP, Random Forest, Gradient Boosting) achieve near-zero or negative IC — overfitting on return noise.
2. **Mean-Variance Optimization Amplifies Noise:**
   - Feeding ML forecasts into Markowitz optimizers leads to error maximization and turnover friction. S4 Sharpe **0.46** vs S1 Sharpe **0.68**. Bootstrap: $P(\text{S4} > \text{S1}) = 13.3\%$.
3. **S5 is Promising but Exploratory:**
   - Top-5 ML equal-weighting captures the strongest alpha (Sharpe **0.77**, CAGR **24.00%**), but high concentration (23.34% vol) and 2.14% monthly turnover make it costly and unproven against S1.
4. **Simplicity and Robustness Win:**
   - **1/N Equal Weighting (S1, Sharpe 0.68)** and **Hierarchical Risk Parity (S6, Ledoit-Wolf, Sharpe 0.66)** deliver the best cost-adjusted risk balance. Use ML as a *ranking filter* (S5), not a *return vector* for Markowitz.
5. **Negative Results Reported Honestly:**
   - S7 Momentum (Sharpe 0.01, −40.48% drawdown), MLP (IC −0.0171), and the primary S4 hypothesis all failed. This is the core academic contribution.

---
**Thank You. We welcome questions from the examination committee.**
