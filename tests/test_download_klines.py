"""Minimal tests for K-line download, parsing, and quality checks."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.binance_klines import BinanceKlineClient
from src.data.download_klines import parse_time_bound
from src.data.kline_quality import dedupe_by_timestamp, find_time_gaps, prepare_klines
from src.data.kline_schema import klines_to_dataframe


def _make_raw_kline(open_time: int, close: float = 100.0) -> List[Any]:
    return [
        open_time,
        str(close),
        str(close + 1),
        str(close - 1),
        str(close),
        "10.0",
        open_time + 59_999,
        "1000.0",
        100,
        "5.0",
        "500.0",
        "0",
    ]


def test_parse_time_bound_date_only() -> None:
    start = parse_time_bound("2026-01-01", is_end=False)
    end = parse_time_bound("2026-02-01", is_end=True)
    assert end > start
    assert end - start == 31 * 86_400_000


def test_klines_to_dataframe_columns() -> None:
    raw = [_make_raw_kline(1_700_000_000_000), _make_raw_kline(1_700_000_060_000)]
    frame = klines_to_dataframe(raw)
    assert list(frame.columns) == [
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
    assert len(frame) == 2
    assert frame["timestamp"].is_monotonic_increasing


def test_dedupe_by_timestamp_keeps_last() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [1000, 1000, 2000],
            "open": [1.0, 9.0, 2.0],
            "high": [1.0, 9.0, 2.0],
            "low": [1.0, 9.0, 2.0],
            "close": [1.0, 9.0, 2.0],
            "volume": [1.0, 9.0, 2.0],
            "quote_volume": [1.0, 9.0, 2.0],
            "trade_count": [1, 9, 2],
            "taker_buy_base_volume": [0.5, 4.5, 1.0],
            "taker_buy_quote_volume": [0.5, 4.5, 1.0],
        }
    )
    deduped, removed = dedupe_by_timestamp(frame)
    assert removed == 1
    assert len(deduped) == 2
    assert deduped.loc[deduped["timestamp"] == 1000, "close"].iloc[0] == 9.0


def test_find_time_gaps_detects_missing_bar() -> None:
    gaps = find_time_gaps(pd.Series([0, 60_000, 180_000]), interval_ms=60_000)
    assert len(gaps) == 1
    assert gaps[0] == (60_000, 180_000, 1)


def test_prepare_klines_continuous() -> None:
    timestamps = [i * 60_000 for i in range(5)]
    frame = klines_to_dataframe([_make_raw_kline(ts) for ts in timestamps])
    cleaned, report = prepare_klines(frame, interval_ms=60_000)
    assert report.is_continuous
    assert report.gap_count == 0
    assert len(cleaned) == 5


@patch.object(BinanceKlineClient, "fetch_page")
def test_client_download_paginates(mock_fetch: MagicMock) -> None:
    interval_ms = 60_000
    start_ms = 0
    page1 = [_make_raw_kline(start_ms + i * interval_ms) for i in range(1000)]
    page2 = [_make_raw_kline(start_ms + (1000 + i) * interval_ms) for i in range(10)]
    mock_fetch.side_effect = [page1, page2]

    client = BinanceKlineClient(market="spot", min_request_interval=0)
    end_ms = start_ms + 1010 * interval_ms
    frame = client.download("BTCUSDT", "1m", start_ms, end_ms)

    assert mock_fetch.call_count == 2
    assert len(frame) == 1010
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].iloc[0] == start_ms


@patch.object(BinanceKlineClient, "fetch_page")
def test_client_download_filters_end_bound(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = [
        _make_raw_kline(0),
        _make_raw_kline(60_000),
        _make_raw_kline(120_000),
    ]
    client = BinanceKlineClient(market="spot", min_request_interval=0)
    frame = client.download("BTCUSDT", "1m", 0, 60_000)
    assert len(frame) == 1
    assert frame["timestamp"].iloc[-1] == 0
