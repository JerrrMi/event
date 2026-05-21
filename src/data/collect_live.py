"""CLI entry point for live REST market data collection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.data.live_collector import LiveMarketCollector
from src.data.market_data_source import RestMarketDataSource
from src.data.market_data_storage import MarketDataStorage
from src.utils.config import load_settings

logger = logging.getLogger(__name__)


def setup_logging(logs_dir: Path, level: str, *, log_file: bool = True) -> None:
    """Configure console and optional file logging for data collection."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        file_handler = logging.FileHandler(logs_dir / "data.log", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect live Binance K-lines and order book via REST polling"
    )
    parser.add_argument("--symbol", default=None, help="Trading pair (default: SYMBOL from .env)")
    parser.add_argument("--interval", default=None, help="K-line interval (default: INTERVAL from .env)")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between polls (default: LIVE_POLL_INTERVAL_SECONDS from .env)",
    )
    parser.add_argument(
        "--kline-limit",
        type=int,
        default=None,
        help="Recent K-lines to fetch each poll (default: LIVE_KLINE_LIMIT from .env)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Raw data directory (default: DATA_DIR/raw from settings)",
    )
    parser.add_argument(
        "--market",
        choices=["spot", "futures"],
        default=None,
        help="Binance market (default: BINANCE_MARKET from .env)",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.2,
        help="Minimum seconds between API requests (default: 0.2)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single poll iteration and exit",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Log to console only (skip logs/data.log)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings(validate=False)
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(settings.logs_dir, log_level, log_file=not args.no_log_file)

    symbol = (args.symbol or settings.symbol).upper()
    interval = args.interval or settings.interval
    poll_interval = (
        args.poll_interval
        if args.poll_interval is not None
        else settings.live_poll_interval_seconds
    )
    kline_limit = args.kline_limit if args.kline_limit is not None else settings.live_kline_limit
    market = args.market or settings.binance_market
    output_dir = Path(args.output_dir) if args.output_dir else settings.data_dir / "raw"

    if poll_interval < 1.0:
        parser.error("--poll-interval must be at least 1 second")
    if kline_limit < 1:
        parser.error("--kline-limit must be at least 1")

    source = RestMarketDataSource(
        market=market,
        min_request_interval=args.min_interval,
        max_retries=settings.live_max_retries,
        retry_backoff=settings.live_retry_backoff,
    )
    storage = MarketDataStorage(output_dir)
    collector = LiveMarketCollector(
        source,
        storage,
        symbol,
        interval,
        poll_interval_seconds=poll_interval,
        kline_limit=kline_limit,
        max_consecutive_errors=settings.live_max_consecutive_errors,
        error_retry_delay_seconds=settings.live_error_retry_delay_seconds,
    )

    try:
        if args.once:
            result = collector.poll_once()
            print(
                f"Fetched {result.klines_fetched} K-line(s), appended {result.klines_appended}, "
                f"order_book_ts={result.order_book_timestamp}"
            )
            source.close()
        else:
            collector.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Live collection failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
