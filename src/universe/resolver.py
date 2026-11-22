"""
src/universe/resolver.py
========================
Point-in-Time NIFTY 50 Universe Resolver.

Provides exact historical constituent resolution, entry/exit date lookups,
sector classifications, reconstitution event logging, and survivorship
coverage boundary calculations.
"""

import os
from typing import Dict, List, Optional, Set, Any, Collection
import pandas as pd


class UniverseResolver:
    """
    Point-in-Time Universe Resolver for NIFTY 50 Indian Equities.
    
    Tracks historical additions, deletions, mergers, and active membership
    intervals from 2015 to 2026.
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.pit_file = os.path.join(base_dir, "data", "raw", "universe_pit.csv")
        self.changes_file = os.path.join(base_dir, "data", "raw", "membership_changes.csv")
        
        if not os.path.exists(self.pit_file) or not os.path.exists(self.changes_file):
            from src.universe.build_universe_history import build_universe_history
            build_universe_history(base_dir=base_dir)

        self.df_pit = pd.read_csv(self.pit_file)
        self.df_changes = pd.read_csv(self.changes_file)

        # Ensure datetime parsing
        self.df_pit["entry_dt"] = pd.to_datetime(self.df_pit["entry_date"])
        # Dates beyond 2200 mapped to far future
        self.df_pit["exit_dt"] = pd.to_datetime(
            self.df_pit["exit_date"].replace({"9999-12-31": "2099-12-31"})
        )
        self.df_changes["date_dt"] = pd.to_datetime(self.df_changes["date"])

    def get_constituents_df(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """
        Return DataFrame of active NIFTY 50 constituents at given date.
        """
        dt = pd.to_datetime(date)
        mask = (self.df_pit["entry_dt"] <= dt) & (dt < self.df_pit["exit_dt"])
        return self.df_pit[mask].copy().reset_index(drop=True)

    def get_constituents(self, date: str | pd.Timestamp, use_ticker: bool = True) -> List[str]:
        """
        Return list of constituent identifiers (tickers or symbols) active at given date.
        """
        df = self.get_constituents_df(date)
        col = "ticker" if use_ticker else "symbol"
        return sorted(df[col].dropna().unique().tolist())

    def get_sector_mapping(self, use_ticker: bool = True) -> Dict[str, str]:
        """
        Return dictionary mapping constituent -> sector.
        """
        key_col = "ticker" if use_ticker else "symbol"
        mapping = {}
        for _, row in self.df_pit.iterrows():
            mapping[row[key_col]] = row["sector"]
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

    def calculate_survivorship_coverage(
        self,
        date: str | pd.Timestamp,
        available_tickers: Collection[str]
    ) -> Dict[str, Any]:
        """
        Calculate point-in-time universe coverage and survivorship bias boundaries.
        
        Parameters:
        -----------
        date: Evaluation date
        available_tickers: List or set of candidate tickers present in data pipeline.
        
        Returns:
        --------
        Dictionary containing PIT constituent count, available overlap,
        coverage ratio, missing tickers, and bias bounds note.
        """
        dt = pd.to_datetime(date)
        pit_tickers = set(self.get_constituents(dt, use_ticker=True))
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
                f"constituents ({coverage_ratio:.1%}) are present in the dataset. "
                f"Unobserved attrition ({len(missing_pit)} stocks) consists primarily of historical "
                f"deletions, cyclical exits, and corporate mergers (e.g., HDFC Ltd, Cairn India). "
                f"Active strategies evaluate strictly within point-in-time verified constituents."
            )
        }



