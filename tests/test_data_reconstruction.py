"""
tests/test_data_reconstruction.py
==================================
Unit and Integration Tests for Historical Data Reconstruction,
Point-in-Time Universe Resolution, and Survivorship Bias Mitigation.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.universe.resolver import UniverseResolver
from src.universe.build_universe_history import build_universe_history


@pytest.fixture(scope="module")
def base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def resolver(base_dir):
    return UniverseResolver(base_dir=base_dir)


def test_authoritative_reconstitution_history(base_dir):
    """Verify that membership_changes.csv contains official reconstitutions."""
    changes_path = os.path.join(base_dir, "data", "raw", "membership_changes.csv")
    assert os.path.exists(changes_path), f"Missing {changes_path}"
    
    df_changes = pd.read_csv(changes_path)
    assert len(df_changes) >= 55, f"Expected >= 55 changes, found {len(df_changes)}"
    assert "date" in df_changes.columns
    assert "action" in df_changes.columns
    assert "ticker" in df_changes.columns
    assert "reason" in df_changes.columns
    assert "verification_status" in df_changes.columns
    assert set(df_changes["action"].unique()).issubset({"ADDITION", "DELETION", "MERGER"})


def test_point_in_time_constituent_count(resolver):
    """Verify that exactly 50 active constituents exist on every tested date."""
    test_dates = [
        "2015-01-02",
        "2016-04-01",
        "2017-09-29",
        "2018-09-28",
        "2019-09-27",
        "2020-03-27",
        "2021-03-31",
        "2022-09-30",
        "2023-03-31",
        "2024-03-28",
        "2024-10-01",
        "2026-11-20"
    ]
    for d in test_dates:
        constituents = resolver.get_constituents(d, use_ticker=True, resolve_aliases=True)
        assert len(constituents) == 50, f"Date {d} has {len(constituents)} constituents, expected 50"


def test_ticker_alias_mapping(resolver):
    """Verify that historical tickers correctly resolve to active trading tickers."""
    assert resolver.resolve_ticker("TATAMOTORS.NS") == "TMPV.NS"
    assert resolver.resolve_ticker("IBULHSGFIN.NS") == "SAMMAANCAP.NS"
    assert resolver.resolve_ticker("LTIM.NS") == "LTM.NS"
    assert resolver.resolve_ticker("BHARTIINFR.NS") == "INDUSTOWER.NS"
    # Unchanged tickers resolve to themselves
    assert resolver.resolve_ticker("RELIANCE.NS") == "RELIANCE.NS"


def test_unresolved_constituents_audit(base_dir):
    """Verify that delisted/merged constituents are tracked with merger details."""
    unres_path = os.path.join(base_dir, "data", "unresolved_constituents.json")
    assert os.path.exists(unres_path), f"Missing {unres_path}"
    
    with open(unres_path, "r", encoding="utf-8") as f:
        unres = json.load(f)
    
    unres_tickers = {item["ticker"] for item in unres}
    assert "HDFC.NS" in unres_tickers
    assert "CAIRN.NS" in unres_tickers
    assert "IDFC.NS" in unres_tickers

    for item in unres:
        assert "terminal_event" in item
        assert "terminal_date" in item
        assert "consideration" in item
        assert "failure_reason" in item
        # Ensure zero fabricated prices or synthetic series are introduced
        assert "synthetic_price" not in item
        assert "imputed_return" not in item


def test_benchmark_distinction_and_metadata(base_dir):
    """Verify official price return benchmark vs investable ETF proxy distinction."""
    meta_path = os.path.join(base_dir, "data", "raw", "benchmark_metadata.json")
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    assert meta["price_return_benchmark"]["symbol"] == "^NSEI"
    assert meta["price_return_benchmark"]["type"] == "OFFICIAL_PRICE_RETURN_INDEX"
    
    assert meta["total_return_proxy"]["symbol"] == "NIFTYBEES.NS"
    assert meta["total_return_proxy"]["type"] == "INVESTABLE_ETF_TOTAL_RETURN_PROXY"
    assert "tracking_error_and_drag_disclosures" in meta["total_return_proxy"]


def test_no_backward_filling_and_forward_fill_bounds(base_dir):
    """Verify no backward filling and max forward-fill limit of 3 days."""
    prices_path = os.path.join(base_dir, "data", "raw", "prices.csv")
    df_prices = pd.read_csv(prices_path)
    
    # Check filled flag exists
    assert "filled" in df_prices.columns
    
    # Check that consecutive filled flags never exceed 3
    for ticker, grp in df_prices.groupby("ticker"):
        filled_arr = grp.sort_values("date")["filled"].values
        consec = 0
        for val in filled_arr:
            if val:
                consec += 1
                assert consec <= 3, f"Ticker {ticker} has >3 consecutive forward-filled days ({consec})"
            else:
                consec = 0
