"""Binance REST top-of-book (bookTicker) client with retry and rate limiting."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

from src.data.binance_klines import BASE_URLS, BinanceAPIError
from src.data.order_book_schema import OrderBookSnapshot, book_ticker_to_snapshot

logger = logging.getLogger(__name__)

BOOK_TICKER_PATHS = {
    "spot": "/api/v3/ticker/bookTicker",
    "futures": "/fapi/v1/ticker/bookTicker",
}

DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MIN_REQUEST_INTERVAL = 0.2
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 1.0


class BinanceOrderBookClient:
    """Fetch best bid/ask from Binance public REST API."""

    def __init__(
        self,
        market: str = "spot",
        *,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        market = market.lower()
        if market not in BASE_URLS:
            raise ValueError(f"Unsupported market: {market!r}. Use 'spot' or 'futures'.")
        self.market = market
        self.base_url = BASE_URLS[market]
        self.book_ticker_path = BOOK_TICKER_PATHS[market]
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self._session = session or requests.Session()
        self._last_request_at: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        logger.warning("Order book request failed, retrying in %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    def fetch_book_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw bookTicker payload for a symbol."""
        params = {"symbol": symbol.upper()}
        url = f"{self.base_url}{self.book_ticker_path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                self._last_request_at = time.monotonic()
            except requests.RequestException as exc:
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise BinanceAPIError(response.status_code, "Unexpected response format")
                return payload

            if response.status_code in {418, 429} or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        self._sleep_backoff(attempt)
                else:
                    self._sleep_backoff(attempt)
                last_error = BinanceAPIError(
                    response.status_code, response.text[:200] or response.reason
                )
                continue

            raise BinanceAPIError(
                response.status_code, response.text[:200] or response.reason
            )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to fetch order book after retries")

    def fetch_snapshot(self, symbol: str, *, timestamp_ms: Optional[int] = None) -> OrderBookSnapshot:
        """Fetch and parse a top-of-book snapshot."""
        payload = self.fetch_book_ticker(symbol)
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        return book_ticker_to_snapshot(payload, timestamp_ms=ts)
