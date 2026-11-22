"""
tests/test_data.py
==================
Automated Data Engineering Unit & Integration Test Suite.

Validates:
1. UniverseResolver: historical point-in-time constituent resolution and survivorship bounds
2. Market Data Quality: non-positive prices, nulls, duplicates, and filled flags
3. Benchmark Separation: Price Return Index vs Total Return Index separation & dividend drift
4. India VIX: range, continuity, and non-null values
5. Anomaly Detection: DataValidator anomaly flagging on injected corruptions
6. Multi-format Storage: Parquet and SQLite integrity
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.universe.resolver import UniverseResolver
from src.data.benchmark import BenchmarkManager
from src.data.validator import DataValidator
from src.data.storage import DataStorage


@pytest.fixture
def resolver():
    return UniverseResolver()


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def storage():
    return DataStorage()


def test_universe_resolver_point_in_time(resolver):
    """Verify point-in-time constituents for specific historical dates."""
    # 2015-01-02: Should have ~47-50 constituents including DLF and Cairn India
    c_2015 = resolver.get_constituents("2015-01-02")
    assert len(c_2015) >= 45
    assert "RELIANCE.NS" in c_2015
    assert "TCS.NS" in c_2015
    assert "DLF.NS" in c_2015  # Active until March 2015

    # 2020-04-01: Should reflect 2020 constituents (e.g., Shree Cement, no Yes Bank)
    c_2020 = resolver.get_constituents("2020-04-01")
    assert "YESBANK.NS" not in c_2020
    assert "SHREECEM.NS" in c_2020
    assert "NESTLEIND.NS" in c_2020

    # 2024-10-01: Recent constituents (Trent, BEL in; Divi's, LTIM out)
    c_2024 = resolver.get_constituents("2024-10-01")
    assert "TRENT.NS" in c_2024
    assert "BEL.NS" in c_2024
    assert "DIVISLAB.NS" not in c_2024


def test_universe_survivorship_coverage_bounds(resolver):
    """Verify calculation of survivorship coverage and bounding notes."""
    active_sample = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    bounds = resolver.calculate_survivorship_coverage("2020-04-01", active_sample)

    assert bounds["pit_total_count"] > 40
    assert bounds["available_in_pit"] == 5
    assert 0.0 < bounds["coverage_ratio"] < 1.0
    assert len(bounds["missing_pit_tickers"]) > 30
    assert "survivorship_bias_bound" in bounds


def test_universe_sector_mapping(resolver):
    """Verify sector classifications are populated and accurate."""
    sectors = resolver.get_sector_mapping()
    assert sectors["RELIANCE.NS"] == "Energy"
    assert sectors["TCS.NS"] == "IT"
    assert sectors["HDFCBANK.NS"] == "Banking"
    assert sectors["SUNPHARMA.NS"] == "Pharma"


def test_market_data_prices_clean():
    """Verify raw prices table has no nulls, non-positive prices, or duplicate records."""
    prices_path = "data/raw/prices.csv"
    assert os.path.exists(prices_path), "prices.csv does not exist"
    df = pd.read_csv(prices_path)

    # Required columns
    assert {"date", "ticker", "adj_close", "volume", "filled"}.issubset(df.columns)
    assert len(df) > 50000

    # Zero nulls in key columns
    assert df["adj_close"].isna().sum() == 0
    assert df["date"].isna().sum() == 0
    assert df["ticker"].isna().sum() == 0

    # Strictly positive prices
    assert (df["adj_close"] <= 0).sum() == 0

    # No duplicates
    assert df.duplicated(subset=["date", "ticker"]).sum() == 0

    # Non-negative volumes
    assert (df["volume"] < 0).sum() == 0


def test_benchmark_price_vs_tri_separation():
    """Verify Price Index and TRI benchmarks are separate and TRI reflects dividend premium."""
    idx_path = "data/raw/index.csv"
    tri_path = "data/raw/index_tri.csv"
    assert os.path.exists(idx_path)
    assert os.path.exists(tri_path)

    df_price = pd.read_csv(idx_path)
    df_tri = pd.read_csv(tri_path)

    assert len(df_price) == len(df_tri)
    assert (df_price["adj_close"] <= 0).sum() == 0
    assert (df_tri["adj_close"] <= 0).sum() == 0

    # Total return over entire period must exceed price return due to dividend reinvestment
    price_return = (df_price["adj_close"].iloc[-1] / df_price["adj_close"].iloc[0]) - 1.0
    tri_return = (df_tri["adj_close"].iloc[-1] / df_tri["adj_close"].iloc[0]) - 1.0

    assert tri_return > price_return
    # Dividend premium over 10 years should be positive (> 20 percentage points)
    assert (tri_return - price_return) > 0.20


def test_india_vix_integrity():
    """Verify India VIX dataset is complete, continuous, and within realistic limits."""
    vix_path = "data/raw/india_vix.csv"
    assert os.path.exists(vix_path)
    df_vix = pd.read_csv(vix_path)

    assert "date" in df_vix.columns
    assert "vix" in df_vix.columns
    assert len(df_vix) > 2500

    assert df_vix["vix"].isna().sum() == 0
    assert (df_vix["vix"] < 5.0).sum() == 0
    assert (df_vix["vix"] > 100.0).sum() == 0
    # Mean VIX in India typically 14 to 20
    assert 12.0 < df_vix["vix"].mean() < 22.0


def test_data_validator_catches_anomalies(validator):
    """Verify DataValidator detects injected duplicates, negative prices, and abnormal spikes."""
    # Corrupt DataFrame with negative price, duplicate, and massive 60% jump
    corrupt_df = pd.DataFrame([
        {"date": "2024-01-01", "ticker": "AAA.NS", "adj_close": 100.0, "volume": 1000, "filled": False},
        {"date": "2024-01-02", "ticker": "AAA.NS", "adj_close": 165.0, "volume": 1200, "filled": False}, # +65% jump
        {"date": "2024-01-03", "ticker": "AAA.NS", "adj_close": -5.0, "volume": 1500, "filled": False},   # negative price
        {"date": "2024-01-03", "ticker": "AAA.NS", "adj_close": 90.0, "volume": 1500, "filled": False},   # duplicate date
    ])

    res = validator.validate_prices(corrupt_df)
    assert not res["passed"]
    assert res["metrics"]["duplicate_records"] > 0
    assert res["metrics"]["non_positive_prices"] > 0
    assert res["metrics"]["abnormal_return_spikes_gt_35pct"] > 0


def test_storage_parquet_and_sqlite(storage):
    """Verify Parquet and SQLite persistence round-trips match original data."""
    test_df = pd.DataFrame({
        "ticker": ["RELIANCE.NS", "TCS.NS"],
        "weight": [0.6, 0.4]
    })
    test_parquet_path = "data/processed/test_roundtrip.parquet"
    storage.save_parquet(test_df, test_parquet_path)
    loaded_parquet = storage.load_parquet(test_parquet_path)
    pd.testing.assert_frame_equal(test_df, loaded_parquet)

    # SQLite
    storage.save_sqlite(test_df, "test_table")
    loaded_sql = storage.load_sqlite("SELECT * FROM test_table")
    pd.testing.assert_frame_equal(test_df, loaded_sql)

    # Cleanup test parquet
    if os.path.exists(test_parquet_path):
        os.remove(test_parquet_path)
