# Feature Engineering and Prediction Target Documentation

**FolioWise Upgrade — Milestone M2: Feature Engineering**  
*Document Version: 1.0 | Date: 2026-11-22*

---

## 1. Executive Summary

Milestone M2 standardizes the quantitative feature set and prediction targets across Indian equities (NIFTY 50 universe). All features are strictly backward-looking at date $t$. The target variable is standardized to a **20-trading-day cross-sectional excess-return rank**, removing market beta noise and aligning models with Spearman Rank Information Coefficient (Rank IC).

---

## 2. Feature Dictionary (21 Standardized Factors)

| Feature Column | Indicator Name | Lookback Horizon | Mathematical Definition | Financial Intuition |
| :--- | :--- | :--- | :--- | :--- |
| `f_ret_1` | 1-Day Return | 1 day | $P_t / P_{t-1} - 1$ | Immediate daily momentum / reversal |
| `f_ret_5` | 1-Week Return | 5 days | $P_t / P_{t-5} - 1$ | Short-term price movement |
| `f_ret_21` | 1-Month Return | 21 days | $P_t / P_{t-21} - 1$ | Medium-term momentum |
| `f_ret_63` | 1-Quarter Return | 63 days | $P_t / P_{t-63} - 1$ | Quarterly trend trajectory |
| `f_vol_21` | Short Volatility | 21 days | $\sigma(R_{t-20:t})$ | Recent realized price volatility |
| `f_vol_63` | Medium Volatility | 63 days | $\sigma(R_{t-62:t})$ | Medium-term idiosyncratic risk |
| `f_mom_126_21` | Intermediate Momentum | 126 days (skip 21) | $P_{t-21} / P_{t-126} - 1$ | Jegadeesh-Titman momentum skipping reversal |
| `f_rsi_14` | Relative Strength Index | 14 days | $100 - \frac{100}{1 + \text{RS}}$ (Wilder) | Overbought (>70) vs Oversold (<30) |
| `f_macd` | MACD Line | 12d / 26d EMA | $\frac{\text{EMA}_{12}(P) - \text{EMA}_{26}(P)}{P_t}$ | Normalized trend convergence |
| `f_macd_sig` | MACD Signal | 9d EMA of MACD | $\text{EMA}_9(\text{MACD})$ | Signal trigger line |
| `f_macd_hist` | MACD Histogram | Diff | $\text{MACD} - \text{Signal}$ | Momentum acceleration |
| `f_sma_50_ratio` | SMA Distance | 50 days | $P_t / \text{SMA}_{50}(P) - 1$ | Intermediate trend distance |
| `f_ema_20_ratio` | EMA Distance | 20 days | $P_t / \text{EMA}_{20}(P) - 1$ | Fast trend distance |
| `f_roc_10` | Rate of Change | 10 days | $P_t / P_{t-10} - 1$ | Short-term momentum velocity |
| `f_volchg` | Volume Surge | 21d / 63d | $\frac{\text{Mean}(V_{t-20:t})}{\text{Mean}(V_{t-62:t})} - 1$ | Institutional accumulation / distribution |
| `f_dist_52wh` | 52-Week High Distance | 252 days | $P_t / \max(P_{t-251:t}) - 1$ | Proximity to anchor high ($\le 0$) |
| `f_rel_nifty` | Relative Alpha Return | 1 day | $R_{i, t} - R_{\text{mkt}, t}$ | Idiosyncratic daily alpha |
| `f_mkt_vol_21` | Market Volatility | 21 days | $\sigma(R_{\text{mkt}, t-20:t})$ | Broad market regime risk |
| `f_mkt_corr_63` | Market Correlation | 63 days | $\text{Corr}(R_{i}, R_{\text{mkt}})_{63}$ | Beta / systematic sensitivity |
| `f_vix_level` | India VIX Level | Point-in-time | $\text{VIX}_t / 100.0$ | Macro volatility regime |
| `f_vix_chg_21` | India VIX Trend | 21 days | $\text{VIX}_t / \text{VIX}_{t-21} - 1$ | Macro risk spike indicator |

---

## 3. Standardized Prediction Targets

FolioWise standardizes model targets in `data/processed/features.csv`:

1. **`target_20d_ret`:** Raw forward 20-trading-day closing return:
   $$R_{i, t \to t+20} = \frac{P_{i, t+20}}{P_{i, t}} - 1$$
2. **`target_excess_20d`:** Excess return over NIFTY 50:
   $$R^{\text{excess}}_{i, t} = R_{i, t \to t+20} - R_{\text{mkt}, t \to t+20}$$
3. **`target_rank_20d` (Primary ML Target):** Normalized cross-sectional percentile rank:
   $$\text{rank}_{i, t} = \frac{\text{Rank}(R^{\text{excess}}_{i, t}) - 1}{N_t - 1} \in [0.0, 1.0]$$
   - Cross-sectional mean: **0.500000**
   - Cross-sectional standard deviation: **~0.2995**
   - Cross-sectional skewness: **~0.00** (symmetric uniform distribution)
4. **`target_21d` (Legacy Backward-Compatible Target):** Retained for full compatibility with earlier models.

---

## 4. Leakage Prevention and Verification Protocol

### 4.1 Strict Leakage Rules
1. **Backward-Looking Invariant:** A feature at row $t$ utilizes strictly prices, volumes, and macro variables $\le t$.
2. **Target Isolation:** Future prices $t+1 \dots t+20$ are used exclusively in target generation columns.
3. **Warm-Up Purging:** Rows where any feature has incomplete warm-up (e.g. 252-day high) are purged.
4. **Walk-Forward Embargo:** Expanding walk-forward training windows purge the trailing 20 days before out-of-sample evaluation to prevent overlapping return contamination.

### 4.2 Multi-Date Truncation Test
Executed across 6 historical cutoff dates:
- `2018-12-31` (Pre-COVID calm regime)
- `2019-12-31` (Pre-COVID market peak)
- `2021-06-30` (Post-COVID recovery)
- `2022-12-30` (Global rate hike cycle)
- `2023-12-29` (NIFTY all-time high rally)
- `2024-12-31` (Structural expansion)

**Result:** **3,402 / 3,402 feature comparisons matched exactly** (max difference = 0.0). Zero look-ahead bias confirmed.
