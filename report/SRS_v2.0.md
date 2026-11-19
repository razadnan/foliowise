# Software Requirements Specification (SRS) v2.0
## FolioWise: ML-Driven Portfolio Optimisation System for Indian Equities

**Document Version:** 2.0  
**Project:** Minor Project – Bachelor of Technology (Computer Science & Engineering)  
**Academic Year:** 2026–2027  
**Date:** November 2026  
**Status:** Approved by Project Guide  

---

## 1. Executive Summary & Changelog

### 1.1 Scope of Version 2.0
Version 2.0 represents a complete methodological upgrade from Version 1.0. While v1.0 established data ingestion, baseline cross-validation, and preliminary portfolio simulation, v2.0 implements institutional-grade quantitative finance rigor:
1. Eliminates execution lookahead bias via **Next-Day Execution**.
2. Standardizes mathematical units across return forecasting, covariance estimation, and turnover friction.
3. Implements an itemized **Indian transaction cost and statutory levy schedule** (STT, Stamp Duty, SEBI turnover fees, NSE transaction charges, and GST).
4. Upgrades benchmark S0 to the **NIFTY 50 Total Return Index (TRI)** to properly account for dividend reinvestment.
5. Replaces backward fills (`bfill`) with causal forward filling and flags missing data points.
6. Introduces market volatility regimes, TreeSHAP feature attributions, neural network benchmarking, strategy S7 (Momentum Top-5), and formal statistical hypothesis testing (block bootstrap Sharpe difference and Newey-West adjusted $t$-statistics).

### 1.2 Pipeline Re-run Plan & Changelog
Every downstream output must be regenerated cleanly according to the following order:

| Stage | Module | Primary Updates in v2.0 |
| :--- | :--- | :--- |
| **Stage 1** | `src/stage1_data.py` | Ingest NIFTY 50 TRI (`data/raw/index_tri.csv`), eliminate `bfill()`, record `filled` flag, audit return spikes > 25%. |
| **Stage 2** | `src/features.py`, `stage2_features.py` | Add 21-day market volatility (`f_mkt_vol_21`) and 63-day stock-market rolling correlation (`f_mkt_corr_63`); omit single-stock sector relatives; rerun 5-date leakage audit. |
| **Stage 3** | `src/stage3_models.py`, `src/shap_analysis.py` | Add Momentum baseline, in-fold scaled neural network, TreeSHAP feature attribution, quintile calibration curves, ICIR calculation. |
| **Stage 4** | `src/stage4_backtest.py` | Implement next-day execution, one-way turnover metric ($0.5\sum\|\Delta w\|$), S0 TRI benchmark, strategy S7 (Momentum Top-5), Ledoit-Wolf HRP, fallback logging. |
| **Stage 5** | `src/stage5_analysis.py` | Sortino ratio, drawdown duration, tracking error, Information Ratio, block bootstrap significance, Newey-West IC $t$-stat, volatility regime breakdown, ablation studies. |
| **Frontend** | `src/build_frontend.py` | Unified JSON export to `results/frontend.json`, dynamic investor view build, developer dashboard integration. |

---

## 2. Methodology & Mathematical Foundations

### 2.1 Next-Day Execution Protocol
In real-world asset management, monthly rebalancing signals cannot be executed at the exact closing price of the signal date $T$ without lookahead bias.
- **Signal Formation ($T$):** On the final trading day of calendar month $M$, after market close ($T$), features, model predictions, and optimal portfolio weights $w^*_{T}$ are computed using data strictly up to date $T$.
- **Trade Execution ($T+1$):** Orders are submitted and executed at the closing price of the *next trading day* ($T+1$). One-way portfolio turnover and transaction friction are deducted on $T+1$.
- **Return Accrual:** The portfolio earns returns under the new allocation starting from the close of $T+1$ to $T+2$.

### 2.2 Mathematical Units & Conversions
All pipeline components adhere to standardized dimensional units:
1. **Predicted Returns:** Forward 21-day returns ($\hat{r}_{21d}$) are converted to annualized compounding units:
   $$\hat{r}_{ann} = (1 + \hat{r}_{21d})^{12} - 1$$
2. **Covariance Matrix:** Sample daily returns covariance $\Sigma_{daily}$ is annualized using standard NSE trading calendar scaling:
   $$\Sigma_{ann} = 252 \cdot \Sigma_{daily}$$
3. **One-Way Portfolio Turnover:** At any rebalance execution date $t$, turnover is defined strictly as one-way trade volume:
   $$\text{Turnover}_t = \frac{1}{2} \sum_{i=1}^{N} |w_{i, t} - w_{i, t-}|$$
   where $w_{i, t-}$ denotes the pre-trade drifted weight of asset $i$ immediately prior to rebalancing.
4. **Transaction Cost Deductions:** Friction is assessed per unit of one-way turnover:
   $$R_{net, t} = R_{gross, t} - c \cdot \text{Turnover}_t$$

### 2.3 Itemized Indian Cost Model
In Indian equity cash delivery transactions, institutional and retail trades incur statutory levies and exchange charges. The cost function models:
- **Securities Transaction Tax (STT):** 0.100% on delivery sell side (0.050% average round-trip).
- **Stamp Duty:** 0.015% on delivery buy side (0.0075% average round-trip).
- **Exchange Turnover Fee (NSE):** 0.00345% on both buy and sell.
- **SEBI Turnover Charge:** ₹10 per crore (0.00010%) on both buy and sell.
- **Brokerage:** Discount institutional rate modeled at 0.030% per leg.
- **Goods and Services Tax (GST):** 18% on (Brokerage + Exchange Fees + SEBI Charges).

**Effective Baseline Cost ($c_{base}$):** 0.20% (20 basis points) per unit of one-way turnover.  
**Sensitivity Analysis Grid:** 0.10% (10 bps), 0.20% (20 bps), and 0.30% (30 bps).

### 2.4 Total Return Index (TRI) Benchmark
The NIFTY 50 Price Index ignores dividend yield (~1.2% to 1.5% annually). Strategy S0 evaluates performance against the **NIFTY 50 Total Return Index (TRI)** (`data/raw/index_tri.csv`), reflecting reinvested gross dividends. The standard price index is retained as a secondary reference.

---

## 3. Strategies and Portfolio Construction

| Strategy | Architecture | Mathematical Formulation / Allocation Rule |
| :--- | :--- | :--- |
| **S0** | Benchmark | NIFTY 50 Total Return Index (TRI) |
| **S1** | Equal Weight | $w_i = 1/N$ for all $N=27$ universe tickers |
| **S2** | Minimum Variance | $\min_w w^T \Sigma_{LW} w$ subject to $\sum w_i = 1$, $0 \le w_i \le 0.15$ |
| **S3** | Historical Max Sharpe | $\max_w \frac{w^T \mu_{hist} - r_f}{\sqrt{w^T \Sigma_{LW} w}}$ subject to $0 \le w_i \le 0.15$ |
| **S4** | ML Max Sharpe (Blend) | $\max_w \frac{w^T \mu_{blend} - r_f}{\sqrt{w^T \Sigma_{LW} w}}$ with $\mu_{blend} = 0.5 \hat{\mu}_{ML} + 0.5 \mu_{hist}$, $0 \le w_i \le 0.15$ |
| **S5** | ML Top-5 Equal Weight | $w_i = 0.20$ for top 5 tickers ranked by $\hat{\mu}_{ML}$; 0 otherwise |
| **S6** | Hierarchical Risk Parity | Tree clustering on distance metric $\sqrt{2(1 - \rho)}$, quasi-diagonalization, bisection recursive risk parity with Ledoit-Wolf covariance |
| **S7** | Momentum Top-5 Equal Weight | $w_i = 0.20$ for top 5 tickers ranked by 12M minus 1M price momentum; 0 otherwise |

*Cap Rule Documentation:* Strategies S5 and S7 hold 5 assets at 20% each. This deliberate exception from the 15% markovitz cap is documented as a high-conviction focused factor strategy.

---

## 4. Evaluation & Statistical Testing Framework

1. **Primary Hypothesis:** Strategy S4 (ML Max Sharpe) achieves a higher net Sharpe ratio than S1 (Equal Weight) after 20 bps transaction costs.
2. **Block Bootstrap Significance:** 1,000 stationary block bootstrap resamples of daily return differences ($R_{S4} - R_{S1}$) with 21-day average block length to preserve autocorrelation. Compute 95% confidence intervals and empirical $p$-value.
3. **Information Coefficient Analysis:** Monthly Spearman rank correlation between predictions and realized 21-day returns. Compute Information Ratio of IC ($\text{ICIR} = \text{Mean}(IC)/\text{Std}(IC)$) and Newey-West autocorrelation-adjusted $t$-statistic.
4. **Macro Regimes:** Segment test period by trailing 21-day NIFTY volatility into Calm, Volatile, and Crisis sub-regimes.
5. **Ablation Studies:** Systematically remove feature groups (Price Momentum, Volatility, Volume, Relative strength) and observe delta in IC and net Sharpe.

---

## 5. Guide Sign-Off and Changelog Verification
- **Lead Investigator:** Minor Project Quantitative Research Group
- **Guide Approval:** Verified for Implementation
- **Target Git Tags:** `v2-baseline` (post Stage 1–5 baseline rerun) and `final-v2` (post full documentation and delivery package).
