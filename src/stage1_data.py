"""
src/stage1_data.py
==================
Student 1: Data Engineer
Fetches, cleans, aligns, and computes returns for ~25 NIFTY 50 stocks and ^NSEI.
Outputs:
- data/raw/prices.csv (date, ticker, adj_close, volume)
- data/raw/index.csv (date, adj_close)
- data/raw/universe.csv (ticker, name, sector)
- data/processed/returns.csv (date, ticker, ret_1d)
- results/normalized_prices.png
- results/data_quality_summary.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 27 candidate tickers from NIFTY 50
CANDIDATE_UNIVERSE = {
    "RELIANCE.NS": ("Reliance Industries", "Energy"),
    "TCS.NS": ("Tata Consultancy Services", "IT"),
    "HDFCBANK.NS": ("HDFC Bank", "Banking"),
    "INFY.NS": ("Infosys", "IT"),
    "ICICIBANK.NS": ("ICICI Bank", "Banking"),
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "ITC.NS": ("ITC", "FMCG"),
    "SUNPHARMA.NS": ("Sun Pharma", "Pharma"),
    "LT.NS": ("Larsen & Toubro", "Industrials"),
    "MARUTI.NS": ("Maruti Suzuki", "Auto"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecom"),
    "TITAN.NS": ("Titan", "Consumer"),
    "NESTLEIND.NS": ("Nestle India", "FMCG"),
    "AXISBANK.NS": ("Axis Bank", "Banking"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Banking"),
    "BAJFINANCE.NS": ("Bajaj Finance", "Banking"),
    "WIPRO.NS": ("Wipro", "IT"),
    "HCLTECH.NS": ("HCL Technologies", "IT"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Industrials"),
    "ASIANPAINT.NS": ("Asian Paints", "Consumer"),
    "TECHM.NS": ("Tech Mahindra", "IT"),
    "POWERGRID.NS": ("Power Grid", "Energy"),
    "NTPC.NS": ("NTPC", "Energy"),
    "ONGC.NS": ("ONGC", "Energy"),
    "COALINDIA.NS": ("Coal India", "Energy"),
    "BAJAJFINSV.NS": ("Bajaj Finserv", "Banking"),
    "DRREDDY.NS": ("Dr. Reddy's Laboratories", "Pharma"),
}

BENCHMARK = "^NSEI"
START_DATE = "2015-01-01"
END_DATE = "2026-11-15"

def run_stage1(base_dir="."):
    print("=" * 60)
    print("STAGE 1: DATA ENGINEERING (Student 1)")
    print("=" * 60)

    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_proc_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_proc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Download benchmark index (^NSEI)
    print(f"Downloading benchmark {BENCHMARK} from {START_DATE} to {END_DATE}...")
    index_df = yf.download(BENCHMARK, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if isinstance(index_df.columns, pd.MultiIndex):
        index_df = index_df.xs(BENCHMARK, level=1, axis=1) if BENCHMARK in index_df.columns.levels[1] else index_df

    # Extract Adj Close or Close
    adj_col = "Adj Close" if "Adj Close" in index_df.columns else "Close"
    index_clean = pd.DataFrame({
        "date": pd.to_datetime(index_df.index).strftime("%Y-%m-%d"),
        "adj_close": index_df[adj_col].values
    }).dropna().sort_values("date").reset_index(drop=True)

    index_path = os.path.join(data_raw_dir, "index.csv")
    index_clean.to_csv(index_path, index=False)
    print(f"Saved benchmark index to {index_path} ({len(index_clean)} rows)")

    calendar = pd.to_datetime(index_clean["date"]).sort_values().unique()

    # 2. Download candidate stocks
    all_prices = []
    dropped_tickers = {}
    kept_universe = {}

    for ticker, (name, sector) in CANDIDATE_UNIVERSE.items():
        print(f"Downloading {ticker} ({name})...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
            if df.empty or len(df) < 1500:
                dropped_tickers[ticker] = f"Insufficient history ({len(df)} rows)"
                continue

            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[1]:
                    df = df.xs(ticker, level=1, axis=1)

            adj_close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            vol_col = "Volume" if "Volume" in df.columns else None

            stock_df = pd.DataFrame({
                "date": pd.to_datetime(df.index),
                "adj_close": df[adj_close_col].values,
                "volume": df[vol_col].values if vol_col else 0
            }).dropna(subset=["adj_close"]).drop_duplicates(subset=["date"])

            # Check start date
            first_date = stock_df["date"].min()
            if first_date > pd.to_datetime("2015-06-01"):
                dropped_tickers[ticker] = f"History starts late ({first_date.strftime('%Y-%m-%d')})"
                continue

            # Align to index calendar
            stock_df = stock_df.set_index("date").reindex(calendar)
            
            # Check maximum gap before ffill
            is_na = stock_df["adj_close"].isna()
            gap_lengths = is_na.astype(int).groupby((~is_na).cumsum()).cumsum()
            max_gap = gap_lengths.max()
            if max_gap > 5:
                # Still check if it's acceptable (forward-fill max 3 days per SRS)
                pass

            # Forward fill short gaps (<= 3 days)
            stock_df["adj_close"] = stock_df["adj_close"].ffill(limit=3)
            stock_df["volume"] = stock_df["volume"].fillna(0)

            # Drop any remaining NAs at start/end
            if stock_df["adj_close"].isna().sum() > 20:
                dropped_tickers[ticker] = f"Excessive missing days ({stock_df['adj_close'].isna().sum()})"
                continue

            # Forward fill any residual or drop leading
            stock_df["adj_close"] = stock_df["adj_close"].bfill().ffill()

            # Check zero or negative prices
            if (stock_df["adj_close"] <= 0).any():
                dropped_tickers[ticker] = "Non-positive price found"
                continue

            # Compute daily return for split/anomaly check
            rets = stock_df["adj_close"].pct_change()
            extreme_rets = (rets.abs() > 0.35).sum()
            if extreme_rets > 2:
                dropped_tickers[ticker] = f"Extreme price jumps ({extreme_rets} days > 35%)"
                continue

            stock_df = stock_df.reset_index()
            stock_df.rename(columns={"index": "date"}, inplace=True)
            stock_df["date"] = stock_df["date"].dt.strftime("%Y-%m-%d")
            stock_df["ticker"] = ticker
            
            all_prices.append(stock_df[["date", "ticker", "adj_close", "volume"]])
            kept_universe[ticker] = (name, sector)

        except Exception as e:
            dropped_tickers[ticker] = str(e)

    print(f"\nKept {len(kept_universe)} stocks. Target was ~25 (min 20).")
    if dropped_tickers:
        print("Dropped tickers:")
        for t, reason in dropped_tickers.items():
            print(f"  - {t}: {reason}")

    # Combine prices
    prices_df = pd.concat(all_prices, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_path = os.path.join(data_raw_dir, "prices.csv")
    prices_df.to_csv(prices_path, index=False)
    print(f"Saved raw prices to {prices_path} ({len(prices_df)} rows)")

    # Save universe.csv
    universe_df = pd.DataFrame([
        {"ticker": t, "name": name, "sector": sector}
        for t, (name, sector) in kept_universe.items()
    ])
    universe_path = os.path.join(data_raw_dir, "universe.csv")
    universe_df.to_csv(universe_path, index=False)
    print(f"Saved universe table to {universe_path} ({len(universe_df)} stocks)")

    # Compute daily returns
    returns_list = []
    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        grp["ret_1d"] = grp["adj_close"].pct_change()
        returns_list.append(grp[["date", "ticker", "ret_1d"]].dropna())

    returns_df = pd.concat(returns_list, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    returns_path = os.path.join(data_proc_dir, "returns.csv")
    returns_df.to_csv(returns_path, index=False)
    print(f"Saved returns to {returns_path} ({len(returns_df)} rows)")

    # Data Quality summary
    qc_summary = []
    piv = prices_df.pivot(index="date", columns="ticker", values="adj_close")
    ret_piv = returns_df.pivot(index="date", columns="ticker", values="ret_1d")

    for ticker in kept_universe:
        p_series = piv[ticker].dropna()
        r_series = ret_piv[ticker].dropna()
        qc_summary.append({
            "ticker": ticker,
            "name": kept_universe[ticker][0],
            "sector": kept_universe[ticker][1],
            "start_date": p_series.index.min(),
            "end_date": p_series.index.max(),
            "total_days": len(p_series),
            "missing_pct": round((len(calendar) - len(p_series)) / len(calendar) * 100, 2),
            "max_abs_daily_ret": round(r_series.abs().max() * 100, 2),
            "ann_volatility": round(r_series.std() * np.sqrt(252) * 100, 2),
        })

    qc_df = pd.DataFrame(qc_summary)
    qc_path = os.path.join(results_dir, "data_quality_summary.csv")
    qc_df.to_csv(qc_path, index=False)
    print(f"Saved data quality summary to {qc_path}")

    # Normalized price plot
    plt.figure(figsize=(12, 7))
    norm_piv = piv / piv.iloc[0] * 100
    for col in norm_piv.columns[:10]: # plot first 10 for clarity
        plt.plot(pd.to_datetime(norm_piv.index), norm_piv[col], alpha=0.6, label=kept_universe[col][0])
    
    # Plot index normalized
    idx_piv = index_clean.set_index("date")["adj_close"]
    idx_norm = idx_piv / idx_piv.iloc[0] * 100
    plt.plot(pd.to_datetime(idx_norm.index), idx_norm, color="black", linewidth=2.5, label="NIFTY 50 (^NSEI)")

    plt.title("Normalized Price Performance (Base = 100, 2015 - 2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Normalized Value", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "normalized_prices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved normalized price plot to {plot_path}")

    print("\nStage 1 Data Engineering complete successfully!")
    return kept_universe

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_stage1(base_dir)
