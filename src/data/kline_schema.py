"""K-line column schema and Binance raw response parsing."""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import pandas as pd

KLINE_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

# Binance interval string -> candle length in milliseconds
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


def interval_to_ms(interval: str) -> int:
    """Return candle length in milliseconds for a Binance interval string."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval!r}")
    return INTERVAL_MS[interval]


def klines_to_dataframe(raw_klines: Iterable[Sequence[Any]]) -> pd.DataFrame:
    """Convert Binance kline arrays into a typed DataFrame."""
    rows = [
        {
            "timestamp": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "quote_volume": float(item[7]),
            "trade_count": int(item[8]),
            "taker_buy_base_volume": float(item[9]),
            "taker_buy_quote_volume": float(item[10]),
        }
        for item in raw_klines
    ]
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return frame.sort_values("timestamp").reset_index(drop=True)
