"""Application entry point for live ARIMA prediction and Telegram alerts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from src.data.market_data_source import RestMarketDataSource
from src.data.market_data_storage import MarketDataStorage
from src.live_runner import LiveTradingRunner
from src.utils.config import load_settings

logger = logging.getLogger(__name__)


def setup_logging(logs_dir: Path, level: str) -> None:
    """Configure console and per-module file logging for the live app."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_targets = {
        "": logs_dir / "app.log",
        "src.data": logs_dir / "data.log",
        "src.models": logs_dir / "model.log",
        "src.signals": logs_dir / "signal.log",
        "src.notify": logs_dir / "telegram.log",
    }

    opened_files: list[logging.FileHandler] = []
    for logger_name, log_path in file_targets.items():
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        opened_files.append(file_handler)
        if logger_name:
            module_logger = logging.getLogger(logger_name)
            module_logger.addHandler(file_handler)
            module_logger.propagate = True
        else:
            root.addHandler(file_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live ARIMA prediction runner for Binance event-contract alerts",
    )
    parser.add_argument(
        "--mode",
        choices=["live"],
        default="live",
        help="Run mode (currently only 'live' is supported)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals without sending Telegram messages (overrides DRY_RUN=false in .env)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Allow Telegram pushes even when DRY_RUN=true in .env",
    )
    parser.add_argument("--symbol", default=None, help="Trading pair (default: SYMBOL from .env)")
    parser.add_argument("--interval", default=None, help="K-line interval (default: INTERVAL from .env)")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between cycles (default: LIVE_POLL_INTERVAL_SECONDS from .env)",
    )
    parser.add_argument(
        "--market",
        choices=["spot", "futures"],
        default=None,
        help="Binance market (default: BINANCE_MARKET from .env)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip startup Telegram health-check message",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def resolve_dry_run(args: argparse.Namespace, settings_dry_run: bool) -> bool:
    if args.dry_run and args.no_dry_run:
        raise ValueError("Cannot specify both --dry-run and --no-dry-run")
    if args.dry_run:
        return True
    if args.no_dry_run:
        return False
    return settings_dry_run


def run_live(args: argparse.Namespace) -> int:
    settings = load_settings(validate=False)
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(settings.logs_dir, log_level)

    dry_run = resolve_dry_run(args, settings.dry_run)

    symbol = (args.symbol or settings.symbol).upper()
    interval = args.interval or settings.interval
    poll_interval = (
        args.poll_interval
        if args.poll_interval is not None
        else settings.live_poll_interval_seconds
    )
    market = args.market or settings.binance_market

    if poll_interval < 1.0:
        raise ValueError("--poll-interval must be at least 1 second")

    if not dry_run:
        settings.validate()
    else:
        try:
            settings.validate()
        except Exception as exc:
            logger.warning("Configuration validation warning in dry-run mode: %s", exc)

    source = RestMarketDataSource(
        market=market,
        min_request_interval=0.2,
        max_retries=settings.live_max_retries,
        retry_backoff=settings.live_retry_backoff,
    )
    storage = MarketDataStorage(settings.data_dir / "raw")

    runner = LiveTradingRunner(
        settings,
        source,
        storage,
        dry_run=dry_run,
    )

    if poll_interval != settings.live_poll_interval_seconds:
        runner.collector.poll_interval_seconds = poll_interval
    if symbol != settings.symbol or interval != settings.interval:
        runner.collector.symbol = symbol
        runner.collector.interval = interval

    try:
        if args.once:
            if not args.no_health_check:
                runner.send_startup_health_check()
            result = runner.run_cycle()
            print(
                f"Cycle complete: direction={result.signal.direction} "
                f"confidence={result.signal.confidence:.3f} "
                f"refit={result.refit_performed} telegram={result.telegram_sent}"
            )
            source.close()
        else:
            runner.run(
                send_health_check=not args.no_health_check,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Live runner failed: %s", exc)
        return 1

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode != "live":
        parser.error(f"Unsupported mode: {args.mode!r}")

    try:
        return run_live(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
