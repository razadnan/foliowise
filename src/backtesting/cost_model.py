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
    """

    def __init__(
        self,
        base_rate: float = 0.0020,  # 20 bps base rate per one-way turnover
        stt_rate: float = 0.0010,   # 10 bps STT
        stamp_duty_rate: float = 0.00015, # 1.5 bps stamp duty (buy)
        exchange_turnover_rate: float = 0.0000325, # NSE exchange charge
        sebi_turnover_rate: float = 0.000001, # SEBI fee
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

    def compute_turnover_cost(self, one_way_turnover: float, fee_rate: float = None) -> float:
        """
        Compute total cost deduction in return terms: cost_rate * one_way_turnover.
        """
        rate = fee_rate if fee_rate is not None else self.base_rate
        return float(rate * one_way_turnover)

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
