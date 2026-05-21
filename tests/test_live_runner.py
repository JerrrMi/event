"""Tests for the live trading runner and app entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app import build_parser, main, resolve_dry_run
from src.data.live_collector import PollResult
from src.data.order_book_schema import OrderBookSnapshot
from src.live_runner import LiveTradingRunner
from src.models.arima_predictor import ARIMAPredictionResult, DIRECTION_UP
from src.signals.signal_engine import SIGNAL_UP, TradingSignal
from src.utils.config import Settings


def _make_klines(count: int, *, start_ts: int = 1_700_000_000_000) -> pd.DataFrame:
    rows = []
    for index in range(count):
        close = 100.0 + index * 0.01
        rows.append(
            {
                "timestamp": start_ts + index * 60_000,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 10.0,
                "quote_volume": close * 10.0,
                "trade_count": 100,
                "taker_buy_base_volume": 5.0,
                "taker_buy_quote_volume": close * 5.0,
            }
        )
    return pd.DataFrame(rows)


def _make_order_book() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=1_700_000_060_000,
        best_bid_price=100.0,
        best_bid_qty=2.0,
        best_ask_price=100.1,
        best_ask_qty=1.0,
        spread=0.1,
        mid_price=100.05,
        book_imbalance=0.33,
    )


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        symbol="BTCUSDT",
        interval="1m",
        prediction_minutes=10,
        arima_order=(1, 0, 1),
        arima_series_type="log_return",
        use_auto_arima=False,
        auto_arima_max_p=5,
        auto_arima_max_q=5,
        auto_arima_max_d=2,
        direction_threshold=0.0,
        train_window=60,
        refit_interval_minutes=5,
        confidence_threshold=0.70,
        signal_cooldown_minutes=10,
        max_spread_bps=50.0,
        binance_market="spot",
        binance_api_key=None,
        binance_api_secret=None,
        binance_testnet=False,
        telegram_bot_token="123456:ABC-DEF",
        telegram_chat_id="999",
        dry_run=True,
        log_level="INFO",
        live_poll_interval_seconds=10.0,
        live_kline_limit=2,
        live_max_retries=3,
        live_retry_backoff=1.0,
        live_max_consecutive_errors=3,
        live_error_retry_delay_seconds=1.0,
        data_dir=pytest.importorskip("pathlib").Path("data"),
        logs_dir=pytest.importorskip("pathlib").Path("logs"),
        project_root=pytest.importorskip("pathlib").Path("."),
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_prediction(*, current_price: float = 100.59) -> ARIMAPredictionResult:
    return ARIMAPredictionResult(
        success=True,
        predicted_cumulative_return=0.002,
        direction=DIRECTION_UP,
        interval_lower=0.001,
        interval_upper=0.003,
        residual_volatility=0.0005,
        model_order=(1, 0, 1),
        series_type="log_return",
        current_price=current_price,
        prediction_horizon_minutes=10,
        forecast_steps=10,
        train_points=59,
    )


def _make_signal(*, should_push: bool = True) -> TradingSignal:
    from src.signals.signal_engine import ConfidenceComponents

    return TradingSignal(
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_060_000,
        current_price=100.59,
        expiry_timestamp_ms=1_700_000_660_000,
        direction=SIGNAL_UP,
        predicted_cumulative_return=0.002,
        confidence=0.85,
        confidence_threshold=0.70,
        should_push_telegram=should_push,
        is_direction_reversal=False,
        arima_model_order=(1, 0, 1),
        spread_bps=10.0,
        book_imbalance=0.33,
        volume_filter_passed=True,
        spread_filter_passed=True,
        cooldown_blocked=False,
        trigger_summary="test signal",
        risk_note="test",
        components=ConfidenceComponents(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        arima_direction=DIRECTION_UP,
    )


def test_resolve_dry_run_flags() -> None:
    args = build_parser().parse_args(["--dry-run"])
    assert resolve_dry_run(args, settings_dry_run=False) is True

    args = build_parser().parse_args(["--no-dry-run"])
    assert resolve_dry_run(args, settings_dry_run=True) is False

    args = build_parser().parse_args([])
    assert resolve_dry_run(args, settings_dry_run=True) is True


def test_run_cycle_reuses_cached_prediction_when_refit_not_due() -> None:
    settings = _make_settings(refit_interval_minutes=60)
    source = MagicMock()
    storage = MagicMock()
    storage.load_klines.return_value = _make_klines(60)

    collector = MagicMock()
    collector.poll_once.return_value = PollResult(
        klines_fetched=2,
        klines_appended=1,
        order_book_saved=True,
        latest_kline_timestamp=1_700_000_060_000,
        order_book_timestamp=1_700_000_060_000,
        order_book=_make_order_book(),
    )

    signal_engine = MagicMock()
    signal_engine.evaluate.return_value = _make_signal(should_push=False)
    notifier = MagicMock()

    runner = LiveTradingRunner(
        settings,
        source,
        storage,
        collector=collector,
        signal_engine=signal_engine,
        notifier=notifier,
        dry_run=True,
    )

    prediction = _make_prediction()
    runner._cached_prediction = prediction
    runner._last_refit_monotonic = __import__("time").monotonic()

    with patch("src.live_runner.predict_from_klines") as mock_predict:
        result = runner.run_cycle()
        mock_predict.assert_not_called()

    assert result.refit_performed is False
    assert result.prediction.current_price == pytest.approx(100.59)
    signal_engine.evaluate.assert_called_once()


def test_run_cycle_pushes_telegram_when_not_dry_run() -> None:
    settings = _make_settings()
    source = MagicMock()
    storage = MagicMock()
    storage.load_klines.return_value = _make_klines(60)

    collector = MagicMock()
    collector.poll_once.return_value = PollResult(
        klines_fetched=2,
        klines_appended=1,
        order_book_saved=True,
        latest_kline_timestamp=1_700_000_060_000,
        order_book_timestamp=1_700_000_060_000,
        order_book=_make_order_book(),
    )

    signal_engine = MagicMock()
    signal_engine.evaluate.return_value = _make_signal(should_push=True)
    notifier = MagicMock()

    runner = LiveTradingRunner(
        settings,
        source,
        storage,
        collector=collector,
        signal_engine=signal_engine,
        notifier=notifier,
        dry_run=False,
    )

    with patch("src.live_runner.predict_from_klines", return_value=_make_prediction()):
        result = runner.run_cycle()

    assert result.refit_performed is True
    assert result.telegram_sent is True
    notifier.notify_signal.assert_called_once()


def test_run_cycle_dry_run_skips_telegram_send() -> None:
    settings = _make_settings()
    source = MagicMock()
    storage = MagicMock()
    storage.load_klines.return_value = _make_klines(60)

    collector = MagicMock()
    collector.poll_once.return_value = PollResult(
        klines_fetched=2,
        klines_appended=0,
        order_book_saved=True,
        latest_kline_timestamp=1_700_000_060_000,
        order_book_timestamp=1_700_000_060_000,
        order_book=_make_order_book(),
    )

    signal_engine = MagicMock()
    signal_engine.evaluate.return_value = _make_signal(should_push=True)
    notifier = MagicMock()

    runner = LiveTradingRunner(
        settings,
        source,
        storage,
        collector=collector,
        signal_engine=signal_engine,
        notifier=notifier,
        dry_run=True,
    )

    with patch("src.live_runner.predict_from_klines", return_value=_make_prediction()):
        result = runner.run_cycle()

    assert result.telegram_sent is False
    notifier.notify_signal.assert_not_called()


def test_request_stop_propagates_to_collector() -> None:
    settings = _make_settings()
    source = MagicMock()
    storage = MagicMock()
    collector = MagicMock()
    collector.poll_interval_seconds = 10.0

    runner = LiveTradingRunner(
        settings,
        source,
        storage,
        collector=collector,
        dry_run=True,
    )
    runner.request_stop()
    assert runner.stop_requested is True
    collector.request_stop.assert_called_once()


@patch("src.app.LiveTradingRunner")
@patch("src.app.RestMarketDataSource")
@patch("src.app.load_settings")
def test_app_once_mode(
    mock_load_settings: MagicMock,
    _mock_source_cls: MagicMock,
    mock_runner_cls: MagicMock,
) -> None:
    settings = _make_settings()
    mock_load_settings.return_value = settings

    runner = MagicMock()
    runner.run_cycle.return_value = MagicMock(
        signal=MagicMock(direction=SIGNAL_UP, confidence=0.85),
        refit_performed=True,
        telegram_sent=False,
    )
    mock_runner_cls.return_value = runner

    exit_code = main(["--mode", "live", "--once", "--no-health-check"])
    assert exit_code == 0
    runner.run_cycle.assert_called_once()
    runner.run.assert_not_called()
