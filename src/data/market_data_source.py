"""Market data source abstraction; REST implementation with WebSocket swap-in point."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from src.data.binance_klines import BinanceKlineClient
from src.data.binance_orderbook import BinanceOrderBookClient
from src.data.order_book_schema import OrderBookSnapshot


@runtime_checkable
class MarketDataSource(Protocol):
    """
    Interface for live market data providers.

    Implementations may use REST polling (RestMarketDataSource) or WebSocket
    subscriptions; callers depend only on this protocol.
    """

    def fetch_latest_klines(self, symbol: str, interval: str, *, limit: int = 2) -> pd.DataFrame:
        """Return the most recent closed/in-progress K-lines."""

    def fetch_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Return the current top-of-book snapshot."""

    def close(self) -> None:
        """Release connections or background workers."""


class RestMarketDataSource:
    """REST polling implementation of MarketDataSource."""

    def __init__(
        self,
        *,
        market: str = "spot",
        min_request_interval: float = 0.2,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
        timeout: int = 30,
        kline_client: Optional[BinanceKlineClient] = None,
        order_book_client: Optional[BinanceOrderBookClient] = None,
    ) -> None:
        client_kwargs = {
            "market": market,
            "min_request_interval": min_request_interval,
            "max_retries": max_retries,
            "retry_backoff": retry_backoff,
            "timeout": timeout,
        }
        self._kline_client = kline_client or BinanceKlineClient(**client_kwargs)
        self._order_book_client = order_book_client or BinanceOrderBookClient(**client_kwargs)

    def fetch_latest_klines(self, symbol: str, interval: str, *, limit: int = 2) -> pd.DataFrame:
        return self._kline_client.fetch_latest(symbol, interval, limit=limit)

    def fetch_order_book(self, symbol: str) -> OrderBookSnapshot:
        return self._order_book_client.fetch_snapshot(symbol)

    def close(self) -> None:
        self._kline_client._session.close()
        self._order_book_client._session.close()


class WebSocketMarketDataSource:
    """
    Placeholder for a future WebSocket-backed implementation.

    Raises NotImplementedError until real-time subscriptions are wired up.
    """

    def fetch_latest_klines(self, symbol: str, interval: str, *, limit: int = 2) -> pd.DataFrame:
        raise NotImplementedError("WebSocketMarketDataSource is not implemented yet")

    def fetch_order_book(self, symbol: str) -> OrderBookSnapshot:
        raise NotImplementedError("WebSocketMarketDataSource is not implemented yet")

    def close(self) -> None:
        pass
