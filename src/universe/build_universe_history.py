"""
src/universe/build_universe_history.py
Generates the authoritative historical NIFTY 50 point-in-time universe records
and semi-annual membership changes (2015 to 2026).
"""

import os
import pandas as pd

def build_universe_history(base_dir="."):
    data_raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_raw_dir, exist_ok=True)

    # 1. Membership Changes Log (2015 to 2026)
    changes = [
        # 2015-03-27 semi-annual reconstitution
        {"date": "2015-03-27", "action": "DELETION", "symbol": "DLF", "ticker": "DLF.NS", "name": "DLF Ltd", "sector": "Real Estate", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "DELETION", "symbol": "JINDALSTEL", "ticker": "JINDALSTEL.NS", "name": "Jindal Steel & Power", "sector": "Metals & Mining", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "DELETION", "symbol": "IDFC", "ticker": "IDFC.NS", "name": "IDFC Ltd", "sector": "Financial Services", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "ADDITION", "symbol": "BOSCHLTD", "ticker": "BOSCHLTD.NS", "name": "Bosch Ltd", "sector": "Auto", "reason": "Included in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "ADDITION", "symbol": "IBULHSGFIN", "ticker": "IBULHSGFIN.NS", "name": "Indiabulls Housing Finance", "sector": "Financial Services", "reason": "Included in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "ADDITION", "symbol": "TECHM", "ticker": "TECHM.NS", "name": "Tech Mahindra", "sector": "IT", "reason": "Included in semi-annual rebalance"},
        {"date": "2015-03-27", "action": "ADDITION", "symbol": "ZEEL", "ticker": "ZEEL.NS", "name": "Zee Entertainment", "sector": "Media", "reason": "Included in semi-annual rebalance"},

        # 2015-09-28
        {"date": "2015-09-28", "action": "DELETION", "symbol": "NMDC", "ticker": "NMDC.NS", "name": "NMDC Ltd", "sector": "Metals & Mining", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2015-09-28", "action": "ADDITION", "symbol": "ADANIPORTS", "ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Services", "reason": "Included in semi-annual rebalance"},

        # 2016-04-01
        {"date": "2016-04-01", "action": "DELETION", "symbol": "CAIRN", "ticker": "CAIRN.NS", "name": "Cairn India", "sector": "Energy", "reason": "Excluded / Vedanta merger"},
        {"date": "2016-04-01", "action": "DELETION", "symbol": "PNB", "ticker": "PNB.NS", "name": "Punjab National Bank", "sector": "Banking", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2016-04-01", "action": "DELETION", "symbol": "VEDL", "ticker": "VEDL.NS", "name": "Vedanta Ltd", "sector": "Metals & Mining", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2016-04-01", "action": "ADDITION", "symbol": "AUROPHARMA", "ticker": "AUROPHARMA.NS", "name": "Aurobindo Pharma", "sector": "Pharma", "reason": "Included in semi-annual rebalance"},
        {"date": "2016-04-01", "action": "ADDITION", "symbol": "BHARTIINFR", "ticker": "INDUSTOWER.NS", "name": "Bharti Infratel (Indus Towers)", "sector": "Telecom", "reason": "Included in semi-annual rebalance"},
        {"date": "2016-04-01", "action": "ADDITION", "symbol": "EICHERMOT", "ticker": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Auto", "reason": "Included in semi-annual rebalance"},

        # 2017-03-31
        {"date": "2017-03-31", "action": "DELETION", "symbol": "BHEL", "ticker": "BHEL.NS", "name": "Bharat Heavy Electricals", "sector": "Industrials", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2017-03-31", "action": "DELETION", "symbol": "IDEA", "ticker": "IDEA.NS", "name": "Vodafone Idea", "sector": "Telecom", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2017-03-31", "action": "ADDITION", "symbol": "IOC", "ticker": "IOC.NS", "name": "Indian Oil Corporation", "sector": "Energy", "reason": "Included in semi-annual rebalance"},

        # 2017-09-29
        {"date": "2017-09-29", "action": "DELETION", "symbol": "ACC", "ticker": "ACC.NS", "name": "ACC Ltd", "sector": "Industrials", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2017-09-29", "action": "DELETION", "symbol": "BANKBARODA", "ticker": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Banking", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2017-09-29", "action": "DELETION", "symbol": "TATAPOWER", "ticker": "TATAPOWER.NS", "name": "Tata Power", "sector": "Energy", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2017-09-29", "action": "ADDITION", "symbol": "BAJFINANCE", "ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Banking", "reason": "Included in semi-annual rebalance"},
        {"date": "2017-09-29", "action": "ADDITION", "symbol": "HINDPETRO", "ticker": "HINDPETRO.NS", "name": "Hindustan Petroleum", "sector": "Energy", "reason": "Included in semi-annual rebalance"},
        {"date": "2017-09-29", "action": "ADDITION", "symbol": "UPL", "ticker": "UPL.NS", "name": "UPL Ltd", "sector": "Commodities", "reason": "Included in semi-annual rebalance"},

        # 2018-04-02
        {"date": "2018-04-02", "action": "DELETION", "symbol": "AMBUJACEM", "ticker": "AMBUJACEM.NS", "name": "Ambuja Cements", "sector": "Industrials", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2018-04-02", "action": "DELETION", "symbol": "AUROPHARMA", "ticker": "AUROPHARMA.NS", "name": "Aurobindo Pharma", "sector": "Pharma", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2018-04-02", "action": "DELETION", "symbol": "BOSCHLTD", "ticker": "BOSCHLTD.NS", "name": "Bosch Ltd", "sector": "Auto", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2018-04-02", "action": "ADDITION", "symbol": "BAJAJFINSV", "ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Banking", "reason": "Included in semi-annual rebalance"},
        {"date": "2018-04-02", "action": "ADDITION", "symbol": "GRASIM", "ticker": "GRASIM.NS", "name": "Grasim Industries", "sector": "Commodities", "reason": "Included in semi-annual rebalance"},
        {"date": "2018-04-02", "action": "ADDITION", "symbol": "TITAN", "ticker": "TITAN.NS", "name": "Titan Company", "sector": "Consumer", "reason": "Included in semi-annual rebalance"},

        # 2018-09-28
        {"date": "2018-09-28", "action": "DELETION", "symbol": "LUPIN", "ticker": "LUPIN.NS", "name": "Lupin Ltd", "sector": "Pharma", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2018-09-28", "action": "ADDITION", "symbol": "JSWSTEEL", "ticker": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Metals & Mining", "reason": "Included in semi-annual rebalance"},

        # 2019-03-29
        {"date": "2019-03-29", "action": "DELETION", "symbol": "HINDPETRO", "ticker": "HINDPETRO.NS", "name": "Hindustan Petroleum", "sector": "Energy", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2019-03-29", "action": "ADDITION", "symbol": "BRITANNIA", "ticker": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "FMCG", "reason": "Included in semi-annual rebalance"},

        # 2019-09-27
        {"date": "2019-09-27", "action": "DELETION", "symbol": "IBULHSGFIN", "ticker": "IBULHSGFIN.NS", "name": "Indiabulls Housing Finance", "sector": "Financial Services", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2019-09-27", "action": "ADDITION", "symbol": "NESTLEIND", "ticker": "NESTLEIND.NS", "name": "Nestle India", "sector": "FMCG", "reason": "Included in semi-annual rebalance"},

        # 2020-03-27
        {"date": "2020-03-27", "action": "DELETION", "symbol": "YESBANK", "ticker": "YESBANK.NS", "name": "Yes Bank", "sector": "Banking", "reason": "RBI restructuring / excluded"},
        {"date": "2020-03-27", "action": "ADDITION", "symbol": "SHREECEM", "ticker": "SHREECEM.NS", "name": "Shree Cement", "sector": "Industrials", "reason": "Included in semi-annual rebalance"},

        # 2020-09-25
        {"date": "2020-09-25", "action": "DELETION", "symbol": "BHARTIINFR", "ticker": "INDUSTOWER.NS", "name": "Bharti Infratel", "sector": "Telecom", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2020-09-25", "action": "DELETION", "symbol": "ZEEL", "ticker": "ZEEL.NS", "name": "Zee Entertainment", "sector": "Media", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2020-09-25", "action": "DELETION", "symbol": "VEDL", "ticker": "VEDL.NS", "name": "Vedanta Ltd", "sector": "Metals & Mining", "reason": "Delisting proposal / excluded"},
        {"date": "2020-09-25", "action": "ADDITION", "symbol": "DIVISLAB", "ticker": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma", "reason": "Included in semi-annual rebalance"},
        {"date": "2020-09-25", "action": "ADDITION", "symbol": "HDFCLIFE", "ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Banking", "reason": "Included in semi-annual rebalance"},
        {"date": "2020-09-25", "action": "ADDITION", "symbol": "SBILIFE", "ticker": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Banking", "reason": "Included in semi-annual rebalance"},

        # 2021-03-31
        {"date": "2021-03-31", "action": "DELETION", "symbol": "GAIL", "ticker": "GAIL.NS", "name": "GAIL India", "sector": "Energy", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2021-03-31", "action": "ADDITION", "symbol": "TATACONSUM", "ticker": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "FMCG", "reason": "Included in semi-annual rebalance"},

        # 2022-03-31
        {"date": "2022-03-31", "action": "DELETION", "symbol": "IOC", "ticker": "IOC.NS", "name": "Indian Oil Corporation", "sector": "Energy", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2022-03-31", "action": "ADDITION", "symbol": "APOLLOHOSP", "ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals Enterprise", "sector": "Pharma", "reason": "Included in semi-annual rebalance"},

        # 2022-09-30
        {"date": "2022-09-30", "action": "DELETION", "symbol": "SHREECEM", "ticker": "SHREECEM.NS", "name": "Shree Cement", "sector": "Industrials", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2022-09-30", "action": "ADDITION", "symbol": "ADANIENT", "ticker": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Metals & Mining", "reason": "Included in semi-annual rebalance"},

        # 2023-07-13
        {"date": "2023-07-13", "action": "MERGER", "symbol": "HDFC", "ticker": "HDFC.NS", "name": "Housing Development Finance Corp", "sector": "Banking", "reason": "Amalgamated into HDFC Bank Ltd"},
        {"date": "2023-07-13", "action": "ADDITION", "symbol": "LTIM", "ticker": "LTIM.NS", "name": "LTIMindtree Ltd", "sector": "IT", "reason": "Included replacing merged HDFC"},

        # 2024-03-28
        {"date": "2024-03-28", "action": "DELETION", "symbol": "UPL", "ticker": "UPL.NS", "name": "UPL Ltd", "sector": "Commodities", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2024-03-28", "action": "ADDITION", "symbol": "SHRIRAMFIN", "ticker": "SHRIRAMFIN.NS", "name": "Shriram Finance", "sector": "Banking", "reason": "Included in semi-annual rebalance"},

        # 2024-09-30
        {"date": "2024-09-30", "action": "DELETION", "symbol": "DIVISLAB", "ticker": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2024-09-30", "action": "DELETION", "symbol": "LTIM", "ticker": "LTIM.NS", "name": "LTIMindtree", "sector": "IT", "reason": "Excluded in semi-annual rebalance"},
        {"date": "2024-09-30", "action": "ADDITION", "symbol": "BEL", "ticker": "BEL.NS", "name": "Bharat Electronics", "sector": "Industrials", "reason": "Included in semi-annual rebalance"},
        {"date": "2024-09-30", "action": "ADDITION", "symbol": "TRENT", "ticker": "TRENT.NS", "name": "Trent Ltd", "sector": "Consumer", "reason": "Included in semi-annual rebalance"},
    ]

    df_changes = pd.DataFrame(changes)
    changes_path = os.path.join(data_raw_dir, "membership_changes.csv")
    df_changes.to_csv(changes_path, index=False)
    print(f"Saved {len(df_changes)} membership changes to {changes_path}")

    # 2. Comprehensive Point-in-Time Membership Records (2015-2026)
    # Master records for all stocks that held NIFTY 50 membership during 2015-2026
    pit_records = [
        # Core foundation members active throughout 2015-2026
        {"symbol": "RELIANCE", "ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "TCS", "ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "INFY", "ticker": "INFY.NS", "name": "Infosys", "sector": "IT", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "HINDUNILVR", "ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "ITC", "ticker": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "SUNPHARMA", "ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "LT", "ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Industrials", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "MARUTI", "ticker": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Auto", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "BHARTIARTL", "ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "AXISBANK", "ticker": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "KOTAKBANK", "ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "HCLTECH", "ticker": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "ULTRACEMCO", "ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Industrials", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "ASIANPAINT", "ticker": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Consumer", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "POWERGRID", "ticker": "POWERGRID.NS", "name": "Power Grid Corp", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "NTPC", "ticker": "NTPC.NS", "name": "NTPC Ltd", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "ONGC", "ticker": "ONGC.NS", "name": "ONGC", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "COALINDIA", "ticker": "COALINDIA.NS", "name": "Coal India", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "DRREDDY", "ticker": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "sector": "Pharma", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "SBIN", "ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "BAJAJ-AUTO", "ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "sector": "Auto", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "M&M", "ticker": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Auto", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "TATAMOTORS", "ticker": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Auto", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "TATASTEEL", "ticker": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals & Mining", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "HINDALCO", "ticker": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Metals & Mining", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "CIPLA", "ticker": "CIPLA.NS", "name": "Cipla Ltd", "sector": "Pharma", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "BPCL", "ticker": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Continuous member"},
        {"symbol": "WIPRO", "ticker": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT", "entry_date": "2015-01-01", "exit_date": "2024-09-30", "status": "REMOVED", "notes": "Continuous until Q3 2024 rebalance"},

        # Entrants during 2015-2026
        {"symbol": "TECHM", "ticker": "TECHM.NS", "name": "Tech Mahindra", "sector": "IT", "entry_date": "2015-03-27", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included March 2015"},
        {"symbol": "ADANIPORTS", "ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Services", "entry_date": "2015-09-28", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2015"},
        {"symbol": "EICHERMOT", "ticker": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Auto", "entry_date": "2016-04-01", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included April 2016"},
        {"symbol": "BAJFINANCE", "ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Banking", "entry_date": "2017-09-29", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2017"},
        {"symbol": "BAJAJFINSV", "ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Banking", "entry_date": "2018-04-02", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included April 2018"},
        {"symbol": "TITAN", "ticker": "TITAN.NS", "name": "Titan Company", "sector": "Consumer", "entry_date": "2018-04-02", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included April 2018"},
        {"symbol": "GRASIM", "ticker": "GRASIM.NS", "name": "Grasim Industries", "sector": "Commodities", "entry_date": "2018-04-02", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included April 2018"},
        {"symbol": "JSWSTEEL", "ticker": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Metals & Mining", "entry_date": "2018-09-28", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2018"},
        {"symbol": "BRITANNIA", "ticker": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "FMCG", "entry_date": "2019-03-29", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included March 2019"},
        {"symbol": "NESTLEIND", "ticker": "NESTLEIND.NS", "name": "Nestle India", "sector": "FMCG", "entry_date": "2019-09-27", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2019"},
        {"symbol": "HDFCLIFE", "ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "sector": "Banking", "entry_date": "2020-09-25", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2020"},
        {"symbol": "SBILIFE", "ticker": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Banking", "entry_date": "2020-09-25", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2020"},
        {"symbol": "TATACONSUM", "ticker": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "FMCG", "entry_date": "2021-03-31", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included March 2021"},
        {"symbol": "APOLLOHOSP", "ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "sector": "Pharma", "entry_date": "2022-03-31", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included March 2022"},
        {"symbol": "ADANIENT", "ticker": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Metals & Mining", "entry_date": "2022-09-30", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2022"},
        {"symbol": "SHRIRAMFIN", "ticker": "SHRIRAMFIN.NS", "name": "Shriram Finance", "sector": "Banking", "entry_date": "2024-03-28", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included March 2024"},
        {"symbol": "BEL", "ticker": "BEL.NS", "name": "Bharat Electronics", "sector": "Industrials", "entry_date": "2024-09-30", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2024"},
        {"symbol": "TRENT", "ticker": "TRENT.NS", "name": "Trent Ltd", "sector": "Consumer", "entry_date": "2024-09-30", "exit_date": "9999-12-31", "status": "ACTIVE", "notes": "Included Sept 2024"},

        # Temporary / Exited Members
        {"symbol": "DLF", "ticker": "DLF.NS", "name": "DLF Ltd", "sector": "Real Estate", "entry_date": "2015-01-01", "exit_date": "2015-03-27", "status": "REMOVED", "notes": "Exited March 2015"},
        {"symbol": "JINDALSTEL", "ticker": "JINDALSTEL.NS", "name": "Jindal Steel & Power", "sector": "Metals & Mining", "entry_date": "2015-01-01", "exit_date": "2015-03-27", "status": "REMOVED", "notes": "Exited March 2015"},
        {"symbol": "IDFC", "ticker": "IDFC.NS", "name": "IDFC Ltd", "sector": "Financial Services", "entry_date": "2015-01-01", "exit_date": "2015-03-27", "status": "REMOVED", "notes": "Exited March 2015"},
        {"symbol": "NMDC", "ticker": "NMDC.NS", "name": "NMDC Ltd", "sector": "Metals & Mining", "entry_date": "2015-01-01", "exit_date": "2015-09-28", "status": "REMOVED", "notes": "Exited Sept 2015"},
        {"symbol": "CAIRN", "ticker": "CAIRN.NS", "name": "Cairn India", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "2016-04-01", "status": "MERGED", "notes": "Merged with Vedanta"},
        {"symbol": "PNB", "ticker": "PNB.NS", "name": "Punjab National Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "2016-04-01", "status": "REMOVED", "notes": "Exited April 2016"},
        {"symbol": "BHEL", "ticker": "BHEL.NS", "name": "Bharat Heavy Electricals", "sector": "Industrials", "entry_date": "2015-01-01", "exit_date": "2017-03-31", "status": "REMOVED", "notes": "Exited March 2017"},
        {"symbol": "IDEA", "ticker": "IDEA.NS", "name": "Vodafone Idea", "sector": "Telecom", "entry_date": "2015-01-01", "exit_date": "2017-03-31", "status": "REMOVED", "notes": "Exited March 2017"},
        {"symbol": "ACC", "ticker": "ACC.NS", "name": "ACC Ltd", "sector": "Industrials", "entry_date": "2015-01-01", "exit_date": "2017-09-29", "status": "REMOVED", "notes": "Exited Sept 2017"},
        {"symbol": "BANKBARODA", "ticker": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "2017-09-29", "status": "REMOVED", "notes": "Exited Sept 2017"},
        {"symbol": "TATAPOWER", "ticker": "TATAPOWER.NS", "name": "Tata Power", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "2017-09-29", "status": "REMOVED", "notes": "Exited Sept 2017"},
        {"symbol": "AMBUJACEM", "ticker": "AMBUJACEM.NS", "name": "Ambuja Cements", "sector": "Industrials", "entry_date": "2015-01-01", "exit_date": "2018-04-02", "status": "REMOVED", "notes": "Exited April 2018"},
        {"symbol": "BOSCHLTD", "ticker": "BOSCHLTD.NS", "name": "Bosch Ltd", "sector": "Auto", "entry_date": "2015-03-27", "exit_date": "2018-04-02", "status": "REMOVED", "notes": "Active 2015-2018"},
        {"symbol": "AUROPHARMA", "ticker": "AUROPHARMA.NS", "name": "Aurobindo Pharma", "sector": "Pharma", "entry_date": "2016-04-01", "exit_date": "2018-04-02", "status": "REMOVED", "notes": "Active 2016-2018"},
        {"symbol": "LUPIN", "ticker": "LUPIN.NS", "name": "Lupin Ltd", "sector": "Pharma", "entry_date": "2015-01-01", "exit_date": "2018-09-28", "status": "REMOVED", "notes": "Exited Sept 2018"},
        {"symbol": "IBULHSGFIN", "ticker": "IBULHSGFIN.NS", "name": "Indiabulls Housing Finance", "sector": "Financial Services", "entry_date": "2015-03-27", "exit_date": "2019-09-27", "status": "REMOVED", "notes": "Active 2015-2019"},
        {"symbol": "YESBANK", "ticker": "YESBANK.NS", "name": "Yes Bank", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "2020-03-27", "status": "REMOVED", "notes": "Exited March 2020"},
        {"symbol": "BHARTIINFR", "ticker": "INDUSTOWER.NS", "name": "Bharti Infratel", "sector": "Telecom", "entry_date": "2016-04-01", "exit_date": "2020-09-25", "status": "REMOVED", "notes": "Active 2016-2020"},
        {"symbol": "ZEEL", "ticker": "ZEEL.NS", "name": "Zee Entertainment", "sector": "Media", "entry_date": "2015-03-27", "exit_date": "2020-09-25", "status": "REMOVED", "notes": "Active 2015-2020"},
        {"symbol": "VEDL", "ticker": "VEDL.NS", "name": "Vedanta Ltd", "sector": "Metals & Mining", "entry_date": "2015-01-01", "exit_date": "2020-09-25", "status": "REMOVED", "notes": "Exited Sept 2020"},
        {"symbol": "GAIL", "ticker": "GAIL.NS", "name": "GAIL India", "sector": "Energy", "entry_date": "2015-01-01", "exit_date": "2021-03-31", "status": "REMOVED", "notes": "Exited March 2021"},
        {"symbol": "IOC", "ticker": "IOC.NS", "name": "Indian Oil Corporation", "sector": "Energy", "entry_date": "2017-03-31", "exit_date": "2022-03-31", "status": "REMOVED", "notes": "Active 2017-2022"},
        {"symbol": "SHREECEM", "ticker": "SHREECEM.NS", "name": "Shree Cement", "sector": "Industrials", "entry_date": "2020-03-27", "exit_date": "2022-09-30", "status": "REMOVED", "notes": "Active 2020-2022"},
        {"symbol": "HDFC", "ticker": "HDFC.NS", "name": "Housing Development Finance Corp", "sector": "Banking", "entry_date": "2015-01-01", "exit_date": "2023-07-13", "status": "MERGED", "notes": "Merged into HDFC Bank July 2023"},
        {"symbol": "HINDPETRO", "ticker": "HINDPETRO.NS", "name": "Hindustan Petroleum", "sector": "Energy", "entry_date": "2017-09-29", "exit_date": "2019-03-29", "status": "REMOVED", "notes": "Active 2017-2019"},
        {"symbol": "UPL", "ticker": "UPL.NS", "name": "UPL Ltd", "sector": "Commodities", "entry_date": "2017-09-29", "exit_date": "2024-03-28", "status": "REMOVED", "notes": "Active 2017-2024"},
        {"symbol": "DIVISLAB", "ticker": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Pharma", "entry_date": "2020-09-25", "exit_date": "2024-09-30", "status": "REMOVED", "notes": "Active 2020-2024"},
        {"symbol": "LTIM", "ticker": "LTIM.NS", "name": "LTIMindtree", "sector": "IT", "entry_date": "2023-07-13", "exit_date": "2024-09-30", "status": "REMOVED", "notes": "Active 2023-2024"},
    ]

    df_pit = pd.DataFrame(pit_records)
    pit_path = os.path.join(data_raw_dir, "universe_pit.csv")
    df_pit.to_csv(pit_path, index=False)
    print(f"Saved {len(df_pit)} point-in-time universe records to {pit_path}")

if __name__ == "__main__":
    build_universe_history()
