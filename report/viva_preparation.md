# ML-Driven Portfolio Optimization for NIFTY 50 Stocks
## Comprehensive Viva & Oral Defense Preparation Pack

This document provides plain-language, rigorous academic answers to the mandatory viva examination questions outlined in Section 9.6 of the Software Requirements Specification (SRS). All five team members should review and understand these core concepts.

---

### Question 1: What is a Sharpe Ratio and why is it useful in portfolio management?
**Plain-Language Answer:**
The Sharpe ratio (William F. Sharpe, 1966) measures how much excess return an investment earns for every unit of total risk (annual volatility) it incurs:
$$\text{Sharpe Ratio} = \frac{\bar{r}_p - r_f}{\sigma_p}$$
where $\bar{r}_p$ is the annualized portfolio return, $r_f$ is the risk-free rate (fixed at 6.0% in our study, representing Indian 91-day Treasury bills), and $\sigma_p$ is the annualized standard deviation of daily portfolio returns.

**Why it is useful:**
- Comparing portfolios solely on raw return (CAGR) is misleading. An aggressive strategy might earn 20% return with 35% volatility and severe 50% drawdowns, whereas a defensive strategy might earn 16% return with only 15% volatility.
- The Sharpe ratio normalizes returns by their risk, allowing investors to evaluate whether extra return comes from genuine skill (alpha) or excessive risk exposure (beta).
- In our study, Strategy S1 (Equal Weight) achieved a Sharpe ratio of **0.66** (CAGR 16.84%, Vol 16.32%), whereas the NIFTY 50 benchmark achieved only **0.32** (CAGR 11.71%, Vol 17.81%), proving that S1 delivered more than double the risk-adjusted efficiency of the market index.

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

**Why Ledoit-Wolf Shrinkage Solves This:**
Olivier Ledoit and Michael Wolf (2004) proved that the optimal estimator is a mathematically sound linear combination of the noisy sample covariance matrix $S$ and a structured, well-conditioned target matrix $F$ (e.g., constant correlation):
$$\hat{\Sigma}_{\text{LW}} = \delta F + (1 - \delta) S, \quad \delta \in [0, 1]$$
- Shrinkage pulls extreme covariances toward their central tendency.
- It guarantees that the covariance matrix is well-conditioned, strictly positive-definite, and non-singular, preventing optimizer crashes and wild portfolio turnover.

---

### Question 5: What is the purpose of the 15% weight cap?
**Plain-Language Answer:**
Under unconstrained mean-variance optimization, the algorithm frequently concentrates 50% to 80% of total wealth into just one or two stocks that happen to have the highest estimated return or lowest historical variance.

**Why the 15% cap is essential:**
1. **Idiosyncratic Risk Defense:** If a company encounters a corporate governance scandal, demerger, or sudden earnings crash, holding an unconstrained 40% position causes catastrophic drawdown. Capping individual stock weights at 15% guarantees diversification across at least 7 distinct companies.
2. **Regulatory & Practical Alignment:** Mutual fund regulations (e.g., SEBI guidelines in India and UCITS in Europe) prohibit open-ended funds from holding more than 10% to 15% of their portfolio in a single corporate issuer.
3. **Taming Estimation Error:** Constraints act as implicit Bayesian shrinkage priors (Frost & Savarino, 1988), preventing the optimizer from exploiting small estimation errors.

---

### Question 6: Why did Machine Learning fail to beat Equal Weight in the Markowitz optimizer (S4 vs S1)?
**Plain-Language Answer:**
This is the core quantitative finding of our project. While our Ridge model achieved a positive Information Coefficient of **IC = +0.0454 (t = 1.52)**—positive, but not statistically significant at the 5% level—Strategy S4 (Max Sharpe with ML returns) underperformed, delivering **9.84% CAGR (Sharpe 0.21)** compared to **16.84% CAGR (Sharpe 0.66)** for Equal Weight (S1).

**The three reasons:**
1. **Error Maximization (Michaud, 1989):** Markowitz optimization acts as an error amplifier. Even with an IC of +0.045, individual monthly return predictions have a large noise band ($\text{RMSE} \approx 8.2\%$). The optimizer takes small predicted return differences between stocks (e.g., +2.1% vs +1.7%) and shifts portfolio allocations drastically, mistaking statistical noise for absolute certainty.
2. **Turnover Friction:** S4 experienced **50.0% average monthly turnover** (rotating half its assets every month). At 20 bps base transaction cost, fee friction dragged performance down by >1.2% every year. Equal Weight (S1) had only **6.1% turnover**, spending almost nothing on trading fees.
3. **The Robustness of 1/N:** As proved by DeMiguel, Garlappi, and Uppal (2009), naive equal weighting eliminates all parameter estimation error entirely ($w_i = 1/N$). In Indian markets, $1/N$ systematically captures the size premium (smaller large-caps outgrowing mega-caps) and automatically executes a contrarian rebalancing strategy (selling winners that grew and buying underperforming stocks each month).
4. **Bootstrap Hypothesis Test Verdict:**
   - Testing the Sharpe difference ($\text{Sharpe}_{\text{S4}} - \text{Sharpe}_{\text{S1}}$) over 2,000 bootstrap resamples yields a mean difference of $-0.385$ and a 95% confidence interval of **[-0.812, +0.031]**.
   - *Because the 95% CI includes zero, we cannot conclude with 95% confidence that S1 beats S4. However, S4 is definitely not better, and is probably worse ($P(\text{S4} > \text{S1}) = 3.4\%$)*.

---

### Question 7: How do we evaluate Strategy S5 (Top-5 ML Equal Weight)?
**Plain-Language Answer:**
Strategy S5 simply selected the top 5 stocks ranked by Ridge predictions and allocated 20% to each, bypassing covariance optimization. It achieved **22.28% CAGR** and a **Sharpe ratio of 0.70** (compared to 0.66 for S1 and 0.32 for NIFTY 50).

However, S5 must be evaluated honestly as **"promising but unproven"**:
1. **Massive Turnover:** S5 churned **87.5% per month** — rotating almost the entire portfolio every 30 days. In real trading, market impact, bid-ask spreads, and securities transaction taxes (STT) would severely punish this turnover.
2. **Cap Violation:** S5 holds 20% per stock, which violates the 15% concentration cap enforced on other strategies. Five equal-weight stocks mathematically cannot satisfy a 15% cap ($1/5 = 0.20$).
3. **No Significance Test:** S5 was not subjected to a paired bootstrap significance test against S1. Its higher Sharpe comes with substantial volatility (23.21%) and may reflect sample luck in a concentrated subset during a bull market.

---

### Question 8: How does the FolioWise web application connect to the backtest, and what are the known differences?
**Plain-Language Answer:**
FolioWise (`frontend/index.html`) is a standalone single-file educational tool that visualizes model outputs. It was built programmatically by `src/build_frontend.py` using `results/frontend.json`.

**Verification Diff:**
The unified diff against the baseline (`frontend/baseline_diff.txt`) confirms that **only 3 lines were changed**:
1. Line 173: FAQ text updated from "13 large companies" to "27 large companies".
2. Line 179: `const S=[...]` updated with live 27-stock data ($m, v$, sector).
3. Line 180: `const RHO=0.2200, NIFTY={m:0.1137, v:0.1774}` updated with empirical test constants.

**Known Differences (SRS §3.3):**
1. **Scoring vs Optimization:** The web page uses a simplified scoring rule that tilts toward higher $m$ and lower $v$ based on the user's selected comfort level, followed by iterative cap normalization. It does not run numerical SLSQP quadratic optimization or Ledoit-Wolf shrinkage, and assumes a constant pairwise correlation ($\rho = 0.22$).
2. **Cap Default:** The web interface defaults to 20% (with 10%, 15%, 30% available), whereas the backtest strictly enforced 15% (except S5).
3. **Growth Projections:** The growth cone on the page is a parametric projection ($m \pm z \cdot v$), not historical backtest paths.
4. **Purpose:** The page is an educational demonstration; the Python backtest is the rigorous empirical evidence.

---

### Question 9: What are the main limitations and future research directions?
**Limitations:**
1. **Short Test Period (6 Years):** 2020–2026 covers one market cycle. Multi-decade validation across secular bear markets is needed.
2. **No Capital Gains Taxes:** Modeling Indian Short-Term Capital Gains (STCG at 20%) would heavily penalize high-turnover S5 (87.5%) and S4 (50.0%) compared to S1 (6.1%).
3. **Survivorship Bias:** The universe is derived from current NIFTY 50 members with unbroken trading history from 2015.
4. **Execution Realism:** Orders execute at closing prices without market impact or slippage.

**Future Extensions:**
- **Turnover Regularization:** Adding a turnover penalty directly to the objective function ($\min w^T \Sigma w - \lambda_1 w^T \mu + \lambda_2 \|w - w_{\text{prev}}\|_1$).
- **Black-Litterman Shrinkage:** Blending ML predictions as views with the equilibrium market portfolio.
- **Alternative Data:** Integrating quarterly earnings surprises and institutional holding changes.
