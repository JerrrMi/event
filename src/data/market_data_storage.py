"""Append live K-line and order book snapshots to local CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.kline_quality import dedupe_by_timestamp
from src.data.kline_schema import KLINE_COLUMNS
from src.data.order_book_schema import ORDER_BOOK_COLUMNS, OrderBookSnapshot

logger = logging.getLogger(__name__)


def kline_filename(symbol: str, interval: str) -> str:
    return f"{symbol.upper()}_{interval}.csv"


def order_book_filename(symbol: str) -> str:
    return f"{symbol.upper()}_orderbook.csv"


class MarketDataStorage:
    """Persist live market data under a raw data directory."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def kline_path(self, symbol: str, interval: str) -> Path:
        return self.raw_dir / kline_filename(symbol, interval)

    def order_book_path(self, symbol: str) -> Path:
        return self.raw_dir / order_book_filename(symbol)

    def load_klines(
        self,
        symbol: str,
        interval: str,
        *,
        tail: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load stored K-lines, optionally keeping only the most recent ``tail`` rows."""
        path = self.kline_path(symbol, interval)
        if not path.exists():
            return pd.DataFrame(columns=list(KLINE_COLUMNS))

        frame = pd.read_csv(path)
        missing = [column for column in KLINE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"K-line file {path} is missing required columns: {missing}")

        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if tail is not None and tail > 0:
            frame = frame.tail(tail)
        return frame.reset_index(drop=True)

    def append_klines(self, symbol: str, interval: str, frame: pd.DataFrame) -> int:
        """
        Merge new K-lines into the CSV, deduplicating by timestamp.

        Returns the number of newly appended rows.
        """
        if frame.empty:
            return 0

        incoming = frame[KLINE_COLUMNS].copy()
        path = self.kline_path(symbol, interval)
        previous_count = 0

        if path.exists():
            existing = pd.read_csv(path)
            previous_count = len(existing)
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming

        merged, removed = dedupe_by_timestamp(combined)
        if removed:
            logger.debug("Dropped %d duplicate K-line row(s) while merging", removed)

        merged.to_csv(path, index=False)
        appended = len(merged) - previous_count
        if appended > 0:
            logger.info(
                "Appended %d K-line row(s) to %s (total=%d)",
                appended,
                path,
                len(merged),
            )
        return appended

    def append_order_book(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """Append one order book snapshot row."""
        path = self.order_book_path(symbol)
        row = pd.DataFrame([snapshot.to_dict()], columns=ORDER_BOOK_COLUMNS)
        write_header = not path.exists()
        row.to_csv(path, mode="a", header=write_header, index=False)
        logger.info(
            "Saved order book snapshot to %s (bid=%.2f ask=%.2f spread=%.4f)",
            path,
            snapshot.best_bid_price,
            snapshot.best_ask_price,
            snapshot.spread,
        )
