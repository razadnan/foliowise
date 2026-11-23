"""
src/universe/resolver.py
========================
Authoritative Point-in-Time NIFTY 50 Universe Resolver.

Features:
- Exact constituent resolution at date t using entry/exit dates
- Rigorous separation between:
  1. Historical constituent membership (is_pit_constituent)
  2. Historical price availability (has_price)
  3. Feature lookback eligibility (feature_eligible)
  4. Portfolio construction eligibility (portfolio_eligible)
- Mapping between historical tickers and current trading tickers
- Tracking of delisted, merged, and unresolved companies
- Monthly rebalance coverage reporting and survivorship bias boundaries
"""

import os
import json
from typing import Dict, List, Optional, Set, Any, Collection
import numpy as np
import pandas as pd


class UniverseResolver:
    """
    Point-in-Time Universe Resolver for NIFTY 50 Indian Equities.
    
    Tracks historical additions, deletions, mergers, and active membership
    intervals from 2015 to 2026.
    """

    # Authoritative mapping between historical symbols/tickers and current trading tickers
    TICKER_ALIASES = {
        "TATAMOTORS.NS": "TMPV.NS",
        "TATAMOTORS": "TMPV.NS",
        "TMPV": "TMPV.NS",
        "IBULHSGFIN.NS": "SAMMAANCAP.NS",
        "IBULHSGFIN": "SAMMAANCAP.NS",
        "SAMMAANCAP": "SAMMAANCAP.NS",
        "LTIM.NS": "LTM.NS",
        "LTIM": "LTM.NS",
        "LTM": "LTM.NS",
        "BHARTIINFR.NS": "INDUSTOWER.NS",
        "BHARTIINFR": "INDUSTOWER.NS",
        "INDUSTOWER": "INDUSTOWER.NS",
    }

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.pit_file = os.path.join(base_dir, "data", "raw", "universe_pit.csv")
        self.changes_file = os.path.join(base_dir, "data", "raw", "membership_changes.csv")
        self.unresolved_file = os.path.join(base_dir, "data", "unresolved_constituents.json")
        
        if not os.path.exists(self.pit_file) or not os.path.exists(self.changes_file):
            from src.universe.build_universe_history import build_universe_history
            build_universe_history(base_dir=base_dir)

        self.df_pit = pd.read_csv(self.pit_file)
        self.df_changes = pd.read_csv(self.changes_file)

        # Datetime parsing
        self.df_pit["entry_dt"] = pd.to_datetime(self.df_pit["entry_date"])
        self.df_pit["exit_dt"] = pd.to_datetime(
            self.df_pit["exit_date"].replace({"9999-12-31": "2099-12-31"})
        )
        self.df_changes["date_dt"] = pd.to_datetime(self.df_changes["date"])

    def resolve_ticker(self, ticker: str) -> str:
        """Map historical ticker to current trading ticker if alias exists."""
        return self.TICKER_ALIASES.get(ticker, ticker)

    def get_constituents_df(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """
        Return DataFrame of active NIFTY 50 constituents at given date.
        """
        dt = pd.to_datetime(date)
        mask = (self.df_pit["entry_dt"] <= dt) & (dt < self.df_pit["exit_dt"])
        return self.df_pit[mask].copy().reset_index(drop=True)

    def get_constituents(
        self,
        date: str | pd.Timestamp,
        use_ticker: bool = True,
        resolve_aliases: bool = True
    ) -> List[str]:
        """
        Return sorted list of constituent identifiers active at given date.
        
        If resolve_aliases=True, maps historical symbols (e.g., TATAMOTORS.NS)
        to trading symbols (e.g., TMPV.NS).
        """
        df = self.get_constituents_df(date)
        col = "current_ticker" if (use_ticker and resolve_aliases and "current_ticker" in df.columns) else ("ticker" if use_ticker else "symbol")
        tickers = df[col].dropna().unique().tolist()
        if resolve_aliases and use_ticker:
            tickers = [self.resolve_ticker(t) for t in tickers]
        return sorted(list(set(tickers)))

    def get_sector_mapping(self, use_ticker: bool = True, resolve_aliases: bool = True) -> Dict[str, str]:
        """
        Return dictionary mapping constituent ticker/symbol -> sector.
        """
        key_col = "current_ticker" if (use_ticker and resolve_aliases and "current_ticker" in self.df_pit.columns) else ("ticker" if use_ticker else "symbol")
        mapping = {}
        for _, row in self.df_pit.iterrows():
            k = row[key_col]
            if resolve_aliases and use_ticker:
                k = self.resolve_ticker(k)
            mapping[k] = row["sector"]
            # Also record the historical ticker for reverse lookups
            if use_ticker and "ticker" in row:
                mapping[row["ticker"]] = row["sector"]
        return mapping

    def get_membership_changes(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return chronological table of index reconstitutions within optional date window.
        """
        df = self.df_changes.copy()
        if start_date:
            df = df[df["date_dt"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date_dt"] <= pd.to_datetime(end_date)]
        return df.drop(columns=["date_dt"]).sort_values("date").reset_index(drop=True)

    def get_unresolved_constituents(self) -> List[Dict[str, Any]]:
        """Return list of unresolved delisted/merged constituents."""
        if os.path.exists(self.unresolved_file):
            with open(self.unresolved_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def calculate_survivorship_coverage(
        self,
        date: str | pd.Timestamp,
        available_tickers: Collection[str]
    ) -> Dict[str, Any]:
        """
        Calculate point-in-time universe coverage and survivorship bias boundaries.
        """
        dt = pd.to_datetime(date)
        pit_tickers = set(self.get_constituents(dt, use_ticker=True, resolve_aliases=True))
        avail_set = set(available_tickers)

        in_both = pit_tickers.intersection(avail_set)
        missing_pit = sorted(list(pit_tickers - avail_set))
        coverage_ratio = len(in_both) / len(pit_tickers) if pit_tickers else 0.0

        return {
            "date": dt.strftime("%Y-%m-%d"),
            "pit_total_count": len(pit_tickers),
            "available_in_pit": len(in_both),
            "coverage_ratio": round(coverage_ratio, 4),
            "missing_pit_count": len(missing_pit),
            "missing_pit_tickers": missing_pit,
            "survivorship_bias_bound": (
                f"At {dt.strftime('%Y-%m-%d')}, {len(in_both)} of {len(pit_tickers)} NIFTY 50 "
                f"constituents ({coverage_ratio:.1%}) have verified historical prices in the dataset. "
                f"Unobserved attrition ({len(missing_pit)} stocks: {', '.join(missing_pit[:5])}) "
                f"reflects delistings and corporate mergers (e.g. HDFC Ltd). "
                f"Active strategies evaluate strictly within point-in-time verified constituents."
            )
        }

    def generate_rebalance_coverage_audit(
        self,
        rebalance_dates: List[pd.Timestamp],
        prices_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        min_lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Generate detailed historical coverage audit across every monthly rebalance date.
        
        Separates:
        1. Total active PIT constituents (always 50)
        2. Constituents with verified price records on date t
        3. Constituents with sufficient historical observations (min_lookback_days)
        4. Constituents eligible for portfolio optimization
        5. Constituents excluded and itemized exclusion reasons
        """
        prices_df = prices_df.copy()
        prices_df["date_dt"] = pd.to_datetime(prices_df["date"])
        p_dates = prices_df.groupby("ticker")["date_dt"].agg(["min", "max", "count"])

        piv_prices = prices_df.pivot(index="date_dt", columns="ticker", values="adj_close")
        feat_tickers_by_date = {}
        if features_df is not None:
            f_df = features_df.copy()
            f_df["date_dt"] = pd.to_datetime(f_df["date"])
            for d, grp in f_df.groupby("date_dt"):
                feat_tickers_by_date[d] = set(grp["ticker"].dropna().unique())

        audit_rows = []
        for r_date in rebalance_dates:
            dt = pd.to_datetime(r_date)
            date_str = dt.strftime("%Y-%m-%d")
            pit_constituents = self.get_constituents(dt, use_ticker=True, resolve_aliases=True)
            total_pit = len(pit_constituents)

            # 1. Price availability on date dt
            available_prices = set()
            if dt in piv_prices.index:
                valid_p = piv_prices.loc[dt].dropna()
                available_prices = set(valid_p.index).intersection(pit_constituents)
            else:
                # Find nearest prior date within 3 days
                prior_dates = piv_prices.index[piv_prices.index <= dt]
                if len(prior_dates) > 0 and (dt - prior_dates[-1]).days <= 3:
                    valid_p = piv_prices.loc[prior_dates[-1]].dropna()
                    available_prices = set(valid_p.index).intersection(pit_constituents)

            # 2. Lookback sufficiency (at least min_lookback_days prior observations)
            sufficient_history = set()
            for t in available_prices:
                if t in piv_prices.columns:
                    hist_count = piv_prices[t].loc[:dt].dropna().count()
                    if hist_count >= min_lookback_days:
                        sufficient_history.add(t)

            # 3. Feature eligibility
            feature_eligible = set()
            if dt in feat_tickers_by_date:
                feature_eligible = feat_tickers_by_date[dt].intersection(pit_constituents)
            else:
                feature_eligible = sufficient_history

            # 4. Portfolio eligibility (must be PIT constituent + price available + sufficient history)
            portfolio_eligible = sufficient_history.intersection(pit_constituents)

            # 5. Exclusions and itemized reasons
            excluded_tickers = set(pit_constituents) - portfolio_eligible
            exclusion_reasons = {}
            for t in excluded_tickers:
                if t not in available_prices:
                    exclusion_reasons[t] = "MISSING_PRICE_RECORD_OR_DELISTED"
                elif t not in sufficient_history:
                    exclusion_reasons[t] = f"INSUFFICIENT_LOOKBACK_HISTORY (< {min_lookback_days} days)"
                else:
                    exclusion_reasons[t] = "FEATURE_NOT_AVAILABLE"

            audit_rows.append({
                "rebalance_date": date_str,
                "pit_constituents_count": total_pit,
                "prices_available_count": len(available_prices),
                "sufficient_history_count": len(sufficient_history),
                "feature_eligible_count": len(feature_eligible),
                "portfolio_eligible_count": len(portfolio_eligible),
                "excluded_count": len(excluded_tickers),
                "constituent_coverage_pct": round(len(portfolio_eligible) / total_pit * 100, 2) if total_pit > 0 else 0.0,
                "price_coverage_pct": round(len(available_prices) / total_pit * 100, 2) if total_pit > 0 else 0.0,
                "excluded_tickers": sorted(list(excluded_tickers)),
                "exclusion_reasons_json": json.dumps(exclusion_reasons)
            })

        df_audit = pd.DataFrame(audit_rows)
        out_csv = os.path.join(self.base_dir, "results", "historical_rebalance_coverage.csv")
        df_audit.to_csv(out_csv, index=False)
        print(f"Saved historical rebalance coverage audit to {out_csv} ({len(df_audit)} cycles)")
        return df_audit
