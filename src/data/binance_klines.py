"""Binance REST K-line downloader with pagination and rate limiting."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import requests

from src.data.kline_schema import interval_to_ms, klines_to_dataframe

logger = logging.getLogger(__name__)

MAX_KLINES_PER_REQUEST = 1000
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MIN_REQUEST_INTERVAL = 0.2
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 1.0

KLINES_PATHS = {
    "spot": "/api/v3/klines",
    "futures": "/fapi/v1/klines",
}

BASE_URLS = {
    "spot": "https://api.binance.com",
    "futures": "https://fapi.binance.com",
}


class BinanceAPIError(RuntimeError):
    """Raised when Binance API returns a non-retryable error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Binance API error {status_code}: {message}")


class BinanceKlineClient:
    """Fetch historical K-lines from Binance public REST API."""

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
        self.klines_path = KLINES_PATHS[market]
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

    def fetch_page(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int,
        end_ms: Optional[int] = None,
        limit: int = MAX_KLINES_PER_REQUEST,
    ) -> List[List[Any]]:
        """Fetch one page of K-lines."""
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "limit": min(limit, MAX_KLINES_PER_REQUEST),
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        url = f"{self.base_url}{self.klines_path}"
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
                if not isinstance(payload, list):
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
        raise RuntimeError("Failed to fetch K-lines after retries")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        logger.warning("Request failed, retrying in %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    def download(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        on_page: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Download K-lines for [start_ms, end_ms) with pagination."""
        interval_ms = interval_to_ms(interval)
        if start_ms >= end_ms:
            raise ValueError(f"start_ms ({start_ms}) must be less than end_ms ({end_ms})")

        all_rows: List[Sequence[Any]] = []
        cursor = start_ms
        page_index = 0

        while cursor < end_ms:
            page = self.fetch_page(
                symbol,
                interval,
                start_ms=cursor,
                end_ms=end_ms - 1,
            )
            if not page:
                break

            all_rows.extend(page)
            last_open_time = int(page[-1][0])
            next_cursor = last_open_time + interval_ms

            if on_page is not None:
                on_page(page_index, len(page))

            page_index += 1

            if len(page) < MAX_KLINES_PER_REQUEST:
                break
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        frame = klines_to_dataframe(all_rows)
        if not frame.empty:
            frame = frame[frame["timestamp"] < end_ms].reset_index(drop=True)
        return frame
