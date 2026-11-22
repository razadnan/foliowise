# BENCHMARK TRACK METHODOLOGY & EMPIRICAL SPECIFICATION
## Comprehensive Mathematical Specification of Strategies B1 through B4

**Audit Date:** November 2026  
**Module:** `src/optimization/engine.py` & `src/experiments/runner.py`  
**Target Market:** Indian Equity Market (NIFTY 50 Universe)  
**Evaluation Period:** 2020-01-31 to 2026-11-13 (83 monthly rebalance cycles, 1,677 trading sessions)

---

## 1. Executive Summary & Philosophy

In the FolioWise quantitative research architecture, investment strategies are partitioned into two parallel, non-interfering tracks:
1. **The SmartFolio Experiment Ladder (E0 – E6):** Evaluates the incremental impact of machine learning return forecasts (Ridge, XGBoost, LSTM, Ensemble, and Adaptive switching) combined with shrinkage covariance and turnover regularization.
2. **The Portfolio Strategy Benchmark Track (B1 – B4):** Evaluates risk-based and heuristic portfolio construction techniques that do **not** rely on expected-return predictions:
   - **B1:** Hierarchical Risk Parity (HRP)
   - **B2:** Minimum Variance (Min-Var)
   - **B3:** Equal Risk Contribution (ERC / Risk Budgeting)
   - **B4:** Momentum Top-5 (12M minus 1M price momentum)

Both tracks share the **exact same point-in-time universe, historical rebalance dates (83 cycles), next-day ($T+1$) execution timing, intra-month daily weight drift, and Indian statutory transaction cost model**.

---

## 2. Strategy B1: Hierarchical Risk Parity (HRP)

Hierarchical Risk Parity (Lopez de Prado, 2016) builds a diversified portfolio using machine learning (unsupervised hierarchical tree clustering and graph topology) rather than quadratic optimization. It addresses the fundamental flaw of Markowitz mean-variance optimization: the inversion of ill-conditioned covariance matrices ($\Sigma^{-1}$) and sensitivity to return forecast errors.

### Implementation Audit:
In `src/optimization/engine.py` (`PortfolioOptimizer.optimize_hrp`), HRP is implemented through the following exact steps:

#### Step 1: Covariance Input
HRP takes the **Ledoit-Wolf shrinkage covariance matrix** $\Sigma_{\text{LW}} \in \mathbb{R}^{N \times N}$ estimated over a trailing 252-day lookback window:
$$\Sigma_{\text{LW}} = (1 - \delta^*) \Sigma_{\text{sample}} + \delta^* \mu_v I$$
where $\delta^*$ is the optimal shrinkage intensity and $\mu_v = \frac{1}{N} \text{Tr}(\Sigma_{\text{sample}})$.

#### Step 2: Derivation of the True Correlation Matrix
The asset correlation matrix is analytically derived from $\Sigma_{\text{LW}}$:
$$\rho_{i,j} = \frac{\Sigma_{i,j}}{\sqrt{\Sigma_{i,i} \Sigma_{j,j}}}, \quad \rho_{i,j} \in [-1.0, 1.0]$$
*(Audit Note: As confirmed during the independent code audit, the correlation matrix is computed directly from asset variances rather than column-wise DataFrame correlations).*

#### Step 3: Lopez de Prado Distance Metric
The correlation coefficients are transformed into a true metric space distance $D \in \mathbb{R}^{N \times N}$:
$$d_{i,j} = \sqrt{\frac{1}{2}(1 - \rho_{i,j})}, \quad 0 \le d_{i,j} \le 1$$
This distance satisfies non-negativity ($d_{i,j} \ge 0$), symmetry ($d_{i,j} = d_{j,i}$), and triangle inequality ($d_{i,j} \le d_{i,k} + d_{k,j}$).

#### Step 4: Hierarchical Clustering (Tree Construction)
Single linkage agglomerative hierarchical clustering is applied to the condensed distance vector:
$$D(C_A, C_B) = \min_{i \in C_A, j \in C_B} d_{i,j}$$
This generates a hierarchical tree (dendrogram) grouping highly correlated assets into adjacent clusters.

#### Step 5: Quasi-Diagonalization
The rows and columns of the covariance matrix are reordered based on the hierarchical tree linkages (`_get_quasi_diag`). This places highly correlated assets close to each other along the diagonal, transforming the covariance matrix into a block-diagonal structure.

#### Step 6: Recursive Bisection
Weights are allocated through recursive top-down bisection (`_get_rec_bipart`):
1. Initialize portfolio weight $w_i = 1.0$ for all $i$.
2. For each cluster $C$, divide into subclusters $C_1$ and $C_2$.
3. Compute cluster variance:
   $$V_k = \tilde{w}_k^T \Sigma_k \tilde{w}_k, \quad \text{where } \tilde{w}_{k,i} = \frac{1 / \sigma_i^2}{\sum_{j \in C_k} 1 / \sigma_j^2}$$
4. Compute the split factor $\alpha \in [0, 1]$:
   $$\alpha = 1 - \frac{V_1}{V_1 + V_2} = \frac{V_2}{V_1 + V_2}$$
5. Scale cluster weights:
   $$w_{C_1} \leftarrow w_{C_1} \times \alpha, \quad w_{C_2} \leftarrow w_{C_2} \times (1 - \alpha)$$
6. Repeat recursively until each leaf cluster contains a single asset.

#### Step 7: Asset Cap Enforcement
The resulting raw HRP weights are projected onto the constrained simplex using `cap_and_normalize(stock_cap=0.10)`:
$$w_i \le 0.10, \quad \sum_{i=1}^N w_i = 1.0, \quad w_i \ge 0$$

---

## 3. Strategy B2: Minimum Variance (Min-Var)

The Minimum Variance portfolio identifies the unique asset allocation on the efficient frontier that minimizes total portfolio risk, without requiring expected-return predictions.

### Mathematical Formulation:
$$\min_w \quad w^T \Sigma_{\text{LW}} w$$
$$\text{subject to} \quad \sum_{i=1}^N w_i = 1.0$$
$$0 \le w_i \le 0.10 \quad \forall i \in \{1, \dots, N\}$$
$$\sum_{i \in \text{Sector}_k} w_i \le 0.35 \quad \forall k \in \{1, \dots, K\}$$

### Solver & Regularization:
- Solved using Sequential Least Squares Programming (SLSQP) in SciPy.
- Uses Ledoit-Wolf shrinkage covariance $\Sigma_{\text{LW}}$ to guarantee positive definiteness.
- Fallback: If SLSQP fails to converge or constraints are mathematically infeasible, gracefully falls back to inverse-volatility weighting with single-stock caps.

---

## 4. Strategy B3: Equal Risk Contribution (ERC / Risk Budgeting)

Equal Risk Contribution (Maillard, Roncalli, & Teïletche, 2010) allocates capital such that every asset contributes an identical amount to the total portfolio volatility.

### Mathematical Formulation:
Total portfolio volatility is $\sigma_p = \sqrt{w^T \Sigma w}$. The marginal risk contribution of asset $i$ is:
$$\text{MRC}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$
The absolute risk contribution of asset $i$ is:
$$\text{RC}_i = w_i \times \text{MRC}_i = \frac{w_i (\Sigma w)_i}{\sigma_p}$$
The percentage risk contribution is:
$$\text{PRC}_i = \frac{\text{RC}_i}{\sigma_p} = \frac{w_i (\Sigma w)_i}{w^T \Sigma w}$$
In an ERC portfolio with $N$ assets, the target risk budget is equal for all assets: $\text{PRC}_i = \frac{1}{N}$.

The optimization problem minimizes the sum of squared deviations from the target risk budget:
$$\min_w \quad \sum_{i=1}^N \left( \frac{w_i (\Sigma w)_i}{w^T \Sigma w} - \frac{1}{N} \right)^2$$
$$\text{subject to} \quad \sum_{i=1}^N w_i = 1.0, \quad 0 \le w_i \le 0.10$$

---

## 5. Strategy B4: Momentum Top-5 (12M minus 1M)

Momentum Top-5 provides a benchmark for a simple, non-optimized trend heuristic widely deployed in factor investing.

### Mathematical Formulation:
1. For each rebalance date $t$, extract historical daily prices $P_{\tau}$ for all investable tickers.
2. Compute the 12-month minus 1-month price momentum (Jegadeesh & Titman, 1993):
   $$R_{12M-1M, i} = \frac{P_{t-21, i}}{P_{t-252, i}} - 1.0$$
   *(Excluding the most recent 21 trading days eliminates short-term 1-month bid-ask bounce and microstructural reversal).*
3. Rank all assets descending by $R_{12M-1M, i}$.
4. Select the top $k = 5$ assets and assign equal weights:
   $$w_i = \begin{cases} \frac{1}{5} = 0.20 & \text{if asset } i \in \text{Top 5} \\ 0.00 & \text{otherwise} \end{cases}$$

---

## 6. Empirical Verification & Comparison Table

The strategies were executed through `BacktestEngine` with $T+1$ execution and 20 bps transaction costs across all 83 cycles:

| Metric | B1: HRP | B2: Min-Var | B3: ERC | B4: Momentum | E0: 1/N Reference | E3: XGBoost Reference |
|---|---|---|---|---|---|---|
| **CAGR** | **16.78%** | 13.73% | 16.42% | 5.91% | **17.11%** | 15.30% |
| **Annual Volatility** | 15.45% | **14.52%** | 15.57% | 21.38% | 16.32% | 16.12% |
| **Sharpe Ratio** | **0.70** | 0.53 | 0.67 | -0.00 | **0.68** | 0.58 |
| **Sortino Ratio** | **0.96** | 0.75 | 0.92 | -0.01 | 0.93 | 0.81 |
| **Max Drawdown** | -31.01% | **-26.02%** | -31.21% | -41.00% | -33.24% | -27.19% |
| **Calmar Ratio** | **0.54** | 0.53 | 0.53 | 0.14 | 0.51 | 0.56 |
| **Monthly Turnover** | 10.66% | 10.40% | **4.28%** | 26.14% | **3.62%** | 30.81% |
| **Annual Cost Drag** | 30.7 bps | 29.2 bps | **12.3 bps** | 68.6 bps | **10.4 bps** | 87.9 bps |
| **Bootstrap $p$ vs E0** | 0.861 | 0.340 | 0.717 | **0.021\*** | Baseline | 0.536 |
| **Bootstrap $p$ vs E3** | 0.425 | 0.686 | 0.537 | 0.086 | 0.536 | Baseline |

\* Statistically significant at $\alpha = 0.05$.

### Methodological Conclusions:
1. **HRP (B1) combines the best features of 1/N and optimization:** It achieves the highest Sharpe ratio in the system (0.70) while lowering volatility to 15.45% and maximum drawdown to -31.01%.
2. **Min-Var (B2) provides the ultimate drawdown defense:** With a maximum drawdown of only -26.02% and volatility of 14.52%, it is the most defensive strategy in the system.
3. **Unregularized Momentum (B4) is dangerous in Indian equities:** Experiencing a -41.00% drawdown and statistically underperforming 1/N ($p = 0.021$).
