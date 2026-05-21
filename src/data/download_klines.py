"""CLI entry point for downloading Binance historical K-lines."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data.binance_klines import BinanceKlineClient
from src.data.kline_quality import QualityReport, prepare_klines
from src.data.kline_schema import interval_to_ms
from src.utils.config import load_settings

logger = logging.getLogger(__name__)


def parse_time_bound(value: str, *, is_end: bool) -> int:
    """
    Parse a CLI time bound to Unix milliseconds (UTC).

    - Date-only ``YYYY-MM-DD``: start -> 00:00:00 UTC; end -> exclusive next midnight
      when ``is_end`` is True (e.g. ``--end 2026-02-01`` excludes Feb 1).
    - ISO datetime with optional ``Z`` suffix is supported.
    """
    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        dt = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if is_end:
            # Exclusive upper bound: end date means up to that calendar day start
            return int(dt.timestamp() * 1000)
        return int(dt.timestamp() * 1000)

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime: {value!r}. Use YYYY-MM-DD or ISO-8601."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def output_filename(symbol: str, interval: str) -> str:
    return f"{symbol.upper()}_{interval}.csv"


def download_and_save(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    *,
    market: str = "spot",
    min_request_interval: float = 0.2,
) -> tuple[Path, QualityReport]:
    """Download K-lines and write CSV to output_dir."""
    client = BinanceKlineClient(market=market, min_request_interval=min_request_interval)
    interval_ms = interval_to_ms(interval)

    logger.info(
        "Downloading %s %s from %s to %s (market=%s)",
        symbol,
        interval,
        start_ms,
        end_ms,
        market,
    )

    def on_page(page_index: int, count: int) -> None:
        logger.debug("Fetched page %d (%d candles)", page_index, count)

    frame = client.download(
        symbol,
        interval,
        start_ms,
        end_ms,
        on_page=on_page,
    )
    cleaned, report = prepare_klines(frame, interval_ms)

    if report.duplicate_count:
        logger.warning("Removed %d duplicate rows", report.duplicate_count)
    if report.gap_count:
        logger.warning(
            "Found %d time gap(s) in downloaded data; first gap: %s",
            report.gap_count,
            report.gaps[0] if report.gaps else None,
        )
    else:
        logger.info("Time continuity check passed (%d rows)", report.row_count)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_filename(symbol, interval)
    cleaned.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(cleaned), out_path)
    return out_path, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Binance historical K-line data to data/raw/"
    )
    parser.add_argument("--symbol", default=None, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", default="1m", help="K-line interval (default: 1m)")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda v: parse_time_bound(v, is_end=False),
        help="Start time (YYYY-MM-DD or ISO-8601 UTC)",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda v: parse_time_bound(v, is_end=True),
        help="End time, exclusive for date-only (YYYY-MM-DD or ISO-8601 UTC)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: DATA_DIR/raw from settings)",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = load_settings(validate=False)
    symbol = (args.symbol or settings.symbol).upper()
    market = args.market or settings.binance_market
    output_dir = Path(args.output_dir) if args.output_dir else settings.data_dir / "raw"

    if args.start >= args.end:
        parser.error(f"--start must be before --end (got {args.start} >= {args.end})")

    try:
        out_path, report = download_and_save(
            symbol,
            args.interval,
            args.start,
            args.end,
            output_dir,
            market=market,
            min_request_interval=args.min_interval,
        )
    except Exception as exc:
        logger.exception("Download failed: %s", exc)
        return 1

    print(report)
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
