"""
src/backtesting/cost_model.py
=============================
Itemized Indian Statutory Transaction Cost Schedule.

Implements Master Implementation Plan Section 8.4:
Decomposes trading friction into Indian equity delivery statutory rates:
- STT (Securities Transaction Tax): 0.100% on buy and sell
- Stamp Duty: 0.015% on buy
- Exchange Transaction Charges (NSE): 0.00325%
- SEBI Regulatory Turnover Fee: 0.00010%
- GST: 18% on (Brokerage + Exchange Charges + SEBI Charges)
- Large-cap liquidity slippage: ~0.050%
"""

from typing import Dict, Any


class IndianTransactionCostModel:
    """
    Itemized cost calculator for Indian equity delivery trading.

    Reconciles two distinct operational assumptions:
    1. Institutional Flat Rate (Backtest Ladder Standard):
       base_rate = 0.0020 (20 bps per one-way turnover), representing institutional
       DMA execution with zero brokerage and ~2.5 bps slippage per leg.
    2. Statutory Itemized Delivery Schedule (Indian Legal Schedule):
       - Buy leg = 16.895 bps (STT 10 bps + Stamp 1.5 bps + NSE 0.325 bps + SEBI 0.01 bps + GST 0.06 bps + Slippage 5 bps)
       - Sell leg = 15.395 bps (STT 10 bps + Stamp 0 bps + NSE 0.325 bps + SEBI 0.01 bps + GST 0.06 bps + Slippage 5 bps)
       - Round-trip delivery friction = 32.291 bps per unit of rebalanced capital.
    """

    def __init__(
        self,
        base_rate: float = 0.0020,  # 20 bps base rate per one-way turnover
        stt_rate: float = 0.0010,   # 10 bps STT (buy and sell)
        stamp_duty_rate: float = 0.00015, # 1.5 bps stamp duty (buy only)
        exchange_turnover_rate: float = 0.0000325, # 0.325 bps NSE exchange charge
        sebi_turnover_rate: float = 0.000001, # 0.010 bps SEBI fee
        gst_rate: float = 0.18, # 18% GST on services
        slippage_rate: float = 0.0005 # 5 bps execution slippage for NIFTY 50
    ):
        self.base_rate = base_rate
        self.stt_rate = stt_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.exchange_turnover_rate = exchange_turnover_rate
        self.sebi_turnover_rate = sebi_turnover_rate
        self.gst_rate = gst_rate
        self.slippage_rate = slippage_rate

    def get_statutory_rates(self) -> Dict[str, float]:
        """
        Returns exact statutory rates for buy and sell delivery legs.
        """
        taxable_services = self.exchange_turnover_rate + self.sebi_turnover_rate
        gst = self.gst_rate * taxable_services
        buy_rate = self.stt_rate + self.stamp_duty_rate + taxable_services + gst + self.slippage_rate
        sell_rate = self.stt_rate + 0.0 + taxable_services + gst + self.slippage_rate
        round_trip = buy_rate + sell_rate
        avg_one_way = round_trip / 2.0
        return {
            "buy_rate_bps": round(buy_rate * 10000.0, 3),
            "sell_rate_bps": round(sell_rate * 10000.0, 3),
            "round_trip_bps": round(round_trip * 10000.0, 3),
            "average_one_way_bps": round(avg_one_way * 10000.0, 3)
        }

    def compute_turnover_cost(self, one_way_turnover: float, fee_rate: float = None) -> float:
        """
        Compute total cost deduction in return terms: cost_rate * one_way_turnover.
        Uses flat institutional base_rate (default 20 bps) unless overridden.
        """
        rate = fee_rate if fee_rate is not None else self.base_rate
        return float(rate * one_way_turnover)

    def compute_statutory_rebalance_cost(self, one_way_turnover: float) -> float:
        """
        Compute exact statutory delivery cost for a rebalance of given one-way turnover.
        One-way turnover T implies selling T and buying T, incurring round_trip_rate * T.
        """
        rates = self.get_statutory_rates()
        round_trip_rate = rates["round_trip_bps"] / 10000.0
        return float(round_trip_rate * one_way_turnover)

    def get_itemized_breakdown(self, trade_value_inr: float, is_buy: bool = True) -> Dict[str, Any]:
        """
        Compute itemized statutory breakdown for a trade of given INR value.
        """
        stt = self.stt_rate * trade_value_inr
        stamp_duty = self.stamp_duty_rate * trade_value_inr if is_buy else 0.0
        exchange_fee = self.exchange_turnover_rate * trade_value_inr
        sebi_fee = self.sebi_turnover_rate * trade_value_inr
        taxable_services = exchange_fee + sebi_fee
        gst = self.gst_rate * taxable_services
        slippage = self.slippage_rate * trade_value_inr

        statutory_total = stt + stamp_duty + exchange_fee + sebi_fee + gst
        grand_total = statutory_total + slippage
        effective_bps = (grand_total / trade_value_inr) * 10000.0 if trade_value_inr > 0 else 0.0

        return {
            "trade_value_inr": trade_value_inr,
            "is_buy": is_buy,
            "stt": round(stt, 2),
            "stamp_duty": round(stamp_duty, 2),
            "exchange_charges": round(exchange_fee, 2),
            "sebi_turnover_fee": round(sebi_fee, 4),
            "gst": round(gst, 2),
            "statutory_charges_inr": round(statutory_total, 2),
            "slippage_inr": round(slippage, 2),
            "grand_total_inr": round(grand_total, 2),
            "effective_bps": round(effective_bps, 2)
        }
