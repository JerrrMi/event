"""Order book snapshot schema and derived metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

ORDER_BOOK_COLUMNS: List[str] = [
    "timestamp",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "spread",
    "mid_price",
    "book_imbalance",
]


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Top-of-book snapshot with derived spread and imbalance."""

    timestamp: int
    best_bid_price: float
    best_bid_qty: float
    best_ask_price: float
    best_ask_qty: float
    spread: float
    mid_price: float
    book_imbalance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "best_bid_price": self.best_bid_price,
            "best_bid_qty": self.best_bid_qty,
            "best_ask_price": self.best_ask_price,
            "best_ask_qty": self.best_ask_qty,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "book_imbalance": self.book_imbalance,
        }


def compute_book_metrics(
    best_bid_price: float,
    best_bid_qty: float,
    best_ask_price: float,
    best_ask_qty: float,
) -> tuple[float, float, float]:
    """Return spread, mid_price, and book_imbalance from top-of-book values."""
    spread = best_ask_price - best_bid_price
    mid_price = (best_ask_price + best_bid_price) / 2.0
    total_qty = best_bid_qty + best_ask_qty
    if total_qty > 0:
        book_imbalance = (best_bid_qty - best_ask_qty) / total_qty
    else:
        book_imbalance = 0.0
    return spread, mid_price, book_imbalance


def book_ticker_to_snapshot(payload: Dict[str, Any], *, timestamp_ms: int) -> OrderBookSnapshot:
    """Parse Binance bookTicker JSON into an OrderBookSnapshot."""
    best_bid_price = float(payload["bidPrice"])
    best_bid_qty = float(payload["bidQty"])
    best_ask_price = float(payload["askPrice"])
    best_ask_qty = float(payload["askQty"])
    spread, mid_price, book_imbalance = compute_book_metrics(
        best_bid_price, best_bid_qty, best_ask_price, best_ask_qty
    )
    return OrderBookSnapshot(
        timestamp=timestamp_ms,
        best_bid_price=best_bid_price,
        best_bid_qty=best_bid_qty,
        best_ask_price=best_ask_price,
        best_ask_qty=best_ask_qty,
        spread=spread,
        mid_price=mid_price,
        book_imbalance=book_imbalance,
    )
