"""CLI entry point for rolling backtest."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.backtest.rolling_backtest import (
    BacktestConfig,
    format_summary_report,
    run_rolling_backtest,
    save_backtest_results,
)
from src.data.kline_schema import KLINE_COLUMNS
from src.utils.config import PROJECT_ROOT, load_settings

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def load_klines_csv(path: Path) -> pd.DataFrame:
    """Load klines from a CSV file."""
    frame = pd.read_csv(path)
    missing = [column for column in KLINE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Kline CSV is missing required columns: {missing}")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rolling ARIMA backtest on historical 1-minute klines.",
    )
    parser.add_argument("--symbol", help="Trading symbol, default from .env")
    parser.add_argument("--data", type=Path, required=True, help="Path to kline CSV file")
    parser.add_argument(
        "--orderbook",
        type=Path,
        default=None,
        help="Optional order book CSV for spread/imbalance filters",
    )
    parser.add_argument(
        "--prediction-minutes",
        type=int,
        default=None,
        help="Prediction horizon in minutes, default from .env",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=None,
        help="Rolling training window size, default from .env",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for backtest output files, default data/backtest/",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to .env file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary only, do not write result files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings(args.env_file if args.env_file.exists() else None)
    _configure_logging(settings.log_level)

    config = BacktestConfig.from_settings(settings)
    symbol = (args.symbol or config.symbol).upper()
    train_window = args.train_window if args.train_window is not None else config.train_window
    prediction_minutes = (
        args.prediction_minutes if args.prediction_minutes is not None else config.prediction_minutes
    )

    arima_config = replace(config.arima_config, prediction_minutes=prediction_minutes)
    signal_config = replace(config.signal_config, prediction_minutes=prediction_minutes)
    config = BacktestConfig(
        symbol=symbol,
        interval=config.interval,
        train_window=train_window,
        prediction_minutes=prediction_minutes,
        refit_interval_minutes=config.refit_interval_minutes,
        label_threshold=config.label_threshold,
        payout_ratio=config.payout_ratio,
        arima_config=arima_config,
        signal_config=signal_config,
        apply_cooldown=config.apply_cooldown,
    )

    klines = load_klines_csv(args.data)
    orderbook = None
    if args.orderbook is not None:
        orderbook = pd.read_csv(args.orderbook)

    logger.info(
        "Starting rolling backtest for %s with %s klines from %s",
        config.symbol,
        len(klines),
        args.data,
    )

    records, summary = run_rolling_backtest(
        klines,
        config=config,
        orderbook=orderbook,
    )

    report = format_summary_report(summary)
    print(report)

    if not args.no_save:
        output_dir = args.output_dir or (settings.data_dir / "backtest")
        detail_path, summary_path = save_backtest_results(
            records=records,
            summary=summary,
            output_dir=output_dir,
            symbol=config.symbol,
        )
        print(f"\nSaved detail CSV: {detail_path}")
        print(f"Saved summary JSON: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
