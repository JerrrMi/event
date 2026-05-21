"""Tests for live market data collection, storage, and REST source."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.binance_klines import BinanceKlineClient
from src.data.binance_orderbook import BinanceOrderBookClient
from src.data.collect_live import main as collect_live_main
from src.data.live_collector import LiveMarketCollector
from src.data.market_data_source import RestMarketDataSource, WebSocketMarketDataSource
from src.data.market_data_storage import MarketDataStorage
from src.data.order_book_schema import (
    OrderBookSnapshot,
    book_ticker_to_snapshot,
    compute_book_metrics,
)
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


def test_compute_book_metrics() -> None:
    spread, mid, imbalance = compute_book_metrics(100.0, 3.0, 100.5, 1.0)
    assert spread == pytest.approx(0.5)
    assert mid == pytest.approx(100.25)
    assert imbalance == pytest.approx(0.5)


def test_book_ticker_to_snapshot() -> None:
    payload = {
        "symbol": "BTCUSDT",
        "bidPrice": "65000.10",
        "bidQty": "2.5",
        "askPrice": "65000.20",
        "askQty": "1.5",
    }
    snapshot = book_ticker_to_snapshot(payload, timestamp_ms=1_700_000_000_000)
    assert snapshot.best_bid_price == pytest.approx(65000.10)
    assert snapshot.best_ask_price == pytest.approx(65000.20)
    assert snapshot.spread == pytest.approx(0.10)
    assert snapshot.timestamp == 1_700_000_000_000


def test_storage_append_klines_deduplicates(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    frame1 = klines_to_dataframe([_make_raw_kline(60_000), _make_raw_kline(120_000)])
    frame2 = klines_to_dataframe([_make_raw_kline(120_000, close=200.0), _make_raw_kline(180_000)])

    appended1 = storage.append_klines("BTCUSDT", "1m", frame1)
    appended2 = storage.append_klines("BTCUSDT", "1m", frame2)

    assert appended1 == 2
    assert appended2 == 1

    saved = pd.read_csv(storage.kline_path("BTCUSDT", "1m"))
    assert len(saved) == 3
    assert saved.loc[saved["timestamp"] == 120_000, "close"].iloc[0] == pytest.approx(200.0)


def test_storage_append_order_book(tmp_path) -> None:
    storage = MarketDataStorage(tmp_path)
    snapshot = OrderBookSnapshot(
        timestamp=1_000,
        best_bid_price=100.0,
        best_bid_qty=1.0,
        best_ask_price=100.1,
        best_ask_qty=2.0,
        spread=0.1,
        mid_price=100.05,
        book_imbalance=-1 / 3,
    )

    storage.append_order_book("BTCUSDT", snapshot)
    storage.append_order_book("BTCUSDT", snapshot)

    saved = pd.read_csv(storage.order_book_path("BTCUSDT"))
    assert len(saved) == 2
    assert list(saved.columns) == [
        "timestamp",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "spread",
        "mid_price",
        "book_imbalance",
    ]


class FakeMarketDataSource:
    def __init__(self) -> None:
        self.closed = False

    def fetch_latest_klines(self, symbol: str, interval: str, *, limit: int = 2) -> pd.DataFrame:
        return klines_to_dataframe([_make_raw_kline(60_000)])

    def fetch_order_book(self, symbol: str) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            timestamp=1_000,
            best_bid_price=100.0,
            best_bid_qty=1.0,
            best_ask_price=100.1,
            best_ask_qty=1.0,
            spread=0.1,
            mid_price=100.05,
            book_imbalance=0.0,
        )

    def close(self) -> None:
        self.closed = True


def test_live_collector_poll_once(tmp_path) -> None:
    source = FakeMarketDataSource()
    storage = MarketDataStorage(tmp_path)
    collector = LiveMarketCollector(source, storage, "BTCUSDT", "1m")

    result = collector.poll_once()

    assert result.klines_fetched == 1
    assert result.klines_appended == 1
    assert result.order_book_saved is True
    assert storage.kline_path("BTCUSDT", "1m").exists()
    assert storage.order_book_path("BTCUSDT").exists()


def test_live_collector_stops_on_max_consecutive_errors(tmp_path) -> None:
    class FailingSource(FakeMarketDataSource):
        def fetch_latest_klines(self, symbol: str, interval: str, *, limit: int = 2) -> pd.DataFrame:
            raise RuntimeError("network down")

    source = FailingSource()
    storage = MarketDataStorage(tmp_path)
    collector = LiveMarketCollector(
        source,
        storage,
        "BTCUSDT",
        "1m",
        poll_interval_seconds=1.0,
        max_consecutive_errors=2,
        error_retry_delay_seconds=0.01,
    )

    with pytest.raises(RuntimeError, match="network down"):
        collector.run(max_iterations=None)


@patch.object(BinanceKlineClient, "fetch_page")
def test_fetch_latest_without_start_time(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = [_make_raw_kline(60_000)]
    client = BinanceKlineClient(market="spot", min_request_interval=0)
    frame = client.fetch_latest("BTCUSDT", "1m", limit=1)

    assert len(frame) == 1
    mock_fetch.assert_called_once()
    assert mock_fetch.call_args.kwargs.get("start_ms") is None


@patch("src.data.binance_orderbook.requests.Session.get")
def test_order_book_client_retries_on_network_error(mock_get: MagicMock) -> None:
    mock_get.side_effect = [
        requests.ConnectionError("timeout"),
        MagicMock(status_code=200, json=lambda: {
            "symbol": "BTCUSDT",
            "bidPrice": "1",
            "bidQty": "1",
            "askPrice": "2",
            "askQty": "1",
        }),
    ]
    client = BinanceOrderBookClient(market="spot", min_request_interval=0, retry_backoff=0.01)
    snapshot = client.fetch_snapshot("BTCUSDT", timestamp_ms=123)

    assert snapshot.best_bid_price == pytest.approx(1.0)
    assert mock_get.call_count == 2


def test_rest_market_data_source_implements_protocol() -> None:
    source = RestMarketDataSource(market="spot", min_request_interval=0)
    assert hasattr(source, "fetch_latest_klines")
    assert hasattr(source, "fetch_order_book")
    assert hasattr(source, "close")


def test_websocket_source_is_placeholder() -> None:
    source = WebSocketMarketDataSource()
    with pytest.raises(NotImplementedError):
        source.fetch_latest_klines("BTCUSDT", "1m")


@patch("src.data.collect_live.RestMarketDataSource")
@patch("src.data.collect_live.load_settings")
def test_collect_live_once_cli(mock_settings: MagicMock, mock_source_cls: MagicMock) -> None:
    settings = MagicMock()
    settings.symbol = "BTCUSDT"
    settings.interval = "1m"
    settings.binance_market = "spot"
    settings.data_dir = MagicMock()
    settings.data_dir.__truediv__ = lambda _self, other: MagicMock()
    settings.logs_dir = MagicMock()
    settings.log_level = "INFO"
    settings.live_poll_interval_seconds = 10.0
    settings.live_kline_limit = 2
    settings.live_max_retries = 5
    settings.live_retry_backoff = 1.0
    settings.live_max_consecutive_errors = 10
    settings.live_error_retry_delay_seconds = 5.0
    mock_settings.return_value = settings

    source = MagicMock()
    source.fetch_latest_klines.return_value = klines_to_dataframe([_make_raw_kline(60_000)])
    source.fetch_order_book.return_value = OrderBookSnapshot(
        timestamp=1_000,
        best_bid_price=100.0,
        best_bid_qty=1.0,
        best_ask_price=100.1,
        best_ask_qty=1.0,
        spread=0.1,
        mid_price=100.05,
        book_imbalance=0.0,
    )
    mock_source_cls.return_value = source

    with patch("src.data.collect_live.MarketDataStorage") as mock_storage_cls:
        storage = MagicMock()
        storage.append_klines.return_value = 1
        mock_storage_cls.return_value = storage

        exit_code = collect_live_main(["--once", "--no-log-file", "--output-dir", "data/raw"])

    assert exit_code == 0
    source.close.assert_called_once()
