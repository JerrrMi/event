"""Live market data polling loop with retry and graceful shutdown."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Optional

from src.data.market_data_source import MarketDataSource
from src.data.market_data_storage import MarketDataStorage
from src.data.order_book_schema import OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PollResult:
    """Outcome of a single live polling iteration."""

    klines_fetched: int
    klines_appended: int
    order_book_saved: bool
    latest_kline_timestamp: Optional[int]
    order_book_timestamp: Optional[int]
    order_book: Optional[OrderBookSnapshot] = None


class LiveMarketCollector:
    """Poll a MarketDataSource and persist snapshots to local storage."""

    def __init__(
        self,
        source: MarketDataSource,
        storage: MarketDataStorage,
        symbol: str,
        interval: str,
        *,
        poll_interval_seconds: float = 10.0,
        kline_limit: int = 2,
        max_consecutive_errors: int = 10,
        error_retry_delay_seconds: float = 5.0,
    ) -> None:
        self.source = source
        self.storage = storage
        self.symbol = symbol.upper()
        self.interval = interval
        self.poll_interval_seconds = poll_interval_seconds
        self.kline_limit = kline_limit
        self.max_consecutive_errors = max_consecutive_errors
        self.error_retry_delay_seconds = error_retry_delay_seconds
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def poll_once(self) -> PollResult:
        """Fetch latest K-lines and order book once, persisting any new data."""
        klines = self.source.fetch_latest_klines(
            self.symbol, self.interval, limit=self.kline_limit
        )
        appended = self.storage.append_klines(self.symbol, self.interval, klines)

        snapshot = self.source.fetch_order_book(self.symbol)
        self.storage.append_order_book(self.symbol, snapshot)

        latest_ts = int(klines["timestamp"].iloc[-1]) if not klines.empty else None
        logger.debug(
            "Poll complete: symbol=%s interval=%s klines=%d appended=%d bid=%.2f ask=%.2f",
            self.symbol,
            self.interval,
            len(klines),
            appended,
            snapshot.best_bid_price,
            snapshot.best_ask_price,
        )
        return PollResult(
            klines_fetched=len(klines),
            klines_appended=appended,
            order_book_saved=True,
            latest_kline_timestamp=latest_ts,
            order_book_timestamp=snapshot.timestamp,
            order_book=snapshot,
        )

    def run(self, *, max_iterations: Optional[int] = None) -> None:
        """
        Run the polling loop until stopped, max_iterations reached, or too many errors.

        Registers SIGINT/SIGTERM handlers for graceful shutdown on supported platforms.
        """
        self._install_signal_handlers()
        iteration = 0
        consecutive_errors = 0

        logger.info(
            "Starting live collector: symbol=%s interval=%s poll=%.1fs",
            self.symbol,
            self.interval,
            self.poll_interval_seconds,
        )

        try:
            while not self._stop_requested:
                if max_iterations is not None and iteration >= max_iterations:
                    logger.info("Reached max iterations (%d), stopping", max_iterations)
                    break

                try:
                    result = self.poll_once()
                    consecutive_errors = 0
                    logger.info(
                        "Poll #%d: fetched %d K-line(s), appended %d, order_book_ts=%s",
                        iteration + 1,
                        result.klines_fetched,
                        result.klines_appended,
                        result.order_book_timestamp,
                    )
                except Exception as exc:
                    consecutive_errors += 1
                    logger.exception(
                        "Poll failed (%d/%d consecutive): %s",
                        consecutive_errors,
                        self.max_consecutive_errors,
                        exc,
                    )
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.error("Too many consecutive errors, stopping collector")
                        raise
                    self._sleep(self.error_retry_delay_seconds)
                    continue

                iteration += 1
                if self._stop_requested:
                    break
                self._sleep(self.poll_interval_seconds)
        finally:
            logger.info("Live collector shutting down")
            self.source.close()

    def _install_signal_handlers(self) -> None:
        def _handle_stop(signum: int, _frame: object) -> None:
            logger.info("Received signal %s, stopping after current poll", signum)
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_stop)
            except (AttributeError, ValueError):
                # SIGTERM may be unavailable on some platforms (e.g. Windows).
                pass

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.5))
