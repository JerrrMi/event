"""Tests for the signal engine."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.data.order_book_schema import OrderBookSnapshot
from src.models.arima_predictor import ARIMAPredictionResult, DIRECTION_DOWN, DIRECTION_UP
from src.signals.signal_engine import (
    SignalCooldownTracker,
    SignalEngine,
    SignalEngineConfig,
    build_trading_signal,
    compute_confidence_components,
    SIGNAL_DOWN,
    SIGNAL_HOLD,
    SIGNAL_UP,
)
from src.utils.config import Settings


def _make_klines(
    closes: list[float],
    *,
    volumes: list[float] | None = None,
    start_ts: int = 1_000_000,
) -> pd.DataFrame:
    if volumes is None:
        volumes = [100.0] * len(closes)
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "timestamp": start_ts + index * 60_000,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": volumes[index],
                "quote_volume": close * volumes[index],
                "trade_count": 100,
                "taker_buy_base_volume": volumes[index] / 2.0,
                "taker_buy_quote_volume": close * volumes[index] / 2.0,
            }
        )
    return pd.DataFrame(rows)


def _successful_prediction(
    *,
    direction: str = DIRECTION_UP,
    predicted_return: float = 0.002,
    interval_lower: float = 0.0005,
    interval_upper: float = 0.003,
    residual_volatility: float = 0.0003,
    current_price: float = 100_000.0,
) -> ARIMAPredictionResult:
    return ARIMAPredictionResult(
        success=True,
        predicted_cumulative_return=predicted_return,
        direction=direction,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        residual_volatility=residual_volatility,
        model_order=(1, 0, 1),
        series_type="log_return",
        current_price=current_price,
        prediction_horizon_minutes=10,
        forecast_steps=10,
        train_points=100,
    )


def _orderbook_snapshot(
    *,
    spread_bps: float = 5.0,
    book_imbalance: float = 0.2,
    mid_price: float = 100_000.0,
) -> OrderBookSnapshot:
    spread = spread_bps * mid_price / 10_000.0
    half = spread / 2.0
    bid_qty = 10.0 * (1.0 + book_imbalance)
    ask_qty = 10.0 * (1.0 - book_imbalance)
    return OrderBookSnapshot(
        timestamp=int(time.time() * 1000),
        best_bid_price=mid_price - half,
        best_bid_qty=bid_qty,
        best_ask_price=mid_price + half,
        best_ask_qty=ask_qty,
        spread=spread,
        mid_price=mid_price,
        book_imbalance=book_imbalance,
    )


def test_confidence_components_strong_up_signal() -> None:
    prediction = _successful_prediction(direction=DIRECTION_UP, predicted_return=0.005)
    config = SignalEngineConfig(confidence_threshold=0.5, direction_threshold=0.0)
    klines = _make_klines([100.0] * 30, volumes=[100.0] * 30)

    components, volume_passed, spread_passed, spread_bps, imbalance = compute_confidence_components(
        prediction,
        config=config,
        klines=klines,
        orderbook=_orderbook_snapshot(spread_bps=5.0, book_imbalance=0.4),
    )

    assert volume_passed is True
    assert spread_passed is True
    assert spread_bps == pytest.approx(5.0)
    assert imbalance == pytest.approx(0.4)
    assert components.magnitude > 0.5
    assert components.snr > 0.5
    assert components.interval == 1.0
    assert components.imbalance > 0.5


def test_build_signal_hold_when_arima_failed() -> None:
    prediction = ARIMAPredictionResult.failure(
        error_code="FIT_FAILED",
        error_message="fit failed",
    )
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
    )

    assert signal.direction == SIGNAL_HOLD
    assert signal.confidence == 0.0
    assert signal.should_push_telegram is False
    assert "fit failed" in signal.trigger_summary


def test_build_signal_hold_when_confidence_below_threshold() -> None:
    prediction = _successful_prediction(
        direction=DIRECTION_UP,
        predicted_return=0.00001,
        interval_lower=-0.0001,
        interval_upper=0.0001,
        residual_volatility=0.01,
    )
    config = SignalEngineConfig(confidence_threshold=0.99)
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
        config=config,
        klines=_make_klines([100.0] * 25),
        orderbook=_orderbook_snapshot(),
    )

    assert signal.arima_direction == SIGNAL_UP
    assert signal.direction == SIGNAL_HOLD
    assert signal.should_push_telegram is False
    assert any("confidence" in reason for reason in signal.rejection_reasons)


def test_build_signal_up_with_high_confidence() -> None:
    prediction = _successful_prediction(
        direction=DIRECTION_UP,
        predicted_return=0.01,
        interval_lower=0.004,
        interval_upper=0.012,
        residual_volatility=0.0002,
    )
    config = SignalEngineConfig(confidence_threshold=0.3, direction_threshold=0.0)
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
        config=config,
        klines=_make_klines([100.0] * 30, volumes=[200.0] * 30),
        orderbook=_orderbook_snapshot(spread_bps=3.0, book_imbalance=0.5),
    )

    assert signal.direction == SIGNAL_UP
    assert signal.confidence >= config.confidence_threshold
    assert signal.should_push_telegram is True
    assert signal.volume_filter_passed is True
    assert signal.spread_filter_passed is True


def test_build_signal_hold_when_spread_too_wide() -> None:
    prediction = _successful_prediction(direction=DIRECTION_UP, predicted_return=0.01)
    config = SignalEngineConfig(confidence_threshold=0.1, max_spread_bps=10.0)
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
        config=config,
        klines=_make_klines([100.0] * 25),
        orderbook=_orderbook_snapshot(spread_bps=25.0),
    )

    assert signal.direction == SIGNAL_HOLD
    assert signal.spread_filter_passed is False
    assert signal.should_push_telegram is False
    assert any("spread" in reason for reason in signal.rejection_reasons)


def test_build_signal_hold_when_volume_too_low() -> None:
    prediction = _successful_prediction(direction=DIRECTION_UP, predicted_return=0.01)
    config = SignalEngineConfig(confidence_threshold=0.1, min_volume_ratio=0.8)
    volumes = [100.0] * 24 + [5.0]
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
        config=config,
        klines=_make_klines([100.0] * 25, volumes=volumes),
        orderbook=_orderbook_snapshot(),
    )

    assert signal.volume_filter_passed is False
    assert signal.direction == SIGNAL_HOLD
    assert signal.should_push_telegram is False


def test_build_signal_down_with_negative_interval() -> None:
    prediction = _successful_prediction(
        direction=DIRECTION_DOWN,
        predicted_return=-0.008,
        interval_lower=-0.01,
        interval_upper=-0.003,
        residual_volatility=0.0003,
    )
    config = SignalEngineConfig(confidence_threshold=0.3)
    signal = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=1_700_000_000_000,
        config=config,
        klines=_make_klines([100.0] * 30, volumes=[150.0] * 30),
        orderbook=_orderbook_snapshot(book_imbalance=-0.6),
    )

    assert signal.direction == SIGNAL_DOWN
    assert signal.components.interval == 1.0
    assert signal.components.imbalance > 0.5


def test_cooldown_blocks_repeat_push() -> None:
    prediction = _successful_prediction(direction=DIRECTION_UP, predicted_return=0.01)
    config = SignalEngineConfig(confidence_threshold=0.2, signal_cooldown_minutes=10)
    cooldown = SignalCooldownTracker(cooldown_minutes=10)
    ts = 1_700_000_000_000

    first = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=ts,
        config=config,
        klines=_make_klines([100.0] * 25, volumes=[200.0] * 25),
        orderbook=_orderbook_snapshot(),
        cooldown=cooldown,
    )
    cooldown.record_push("BTCUSDT", SIGNAL_UP, ts)

    second = build_trading_signal(
        prediction,
        symbol="BTCUSDT",
        timestamp_ms=ts + 60_000,
        config=config,
        klines=_make_klines([100.0] * 25, volumes=[200.0] * 25),
        orderbook=_orderbook_snapshot(),
        cooldown=cooldown,
    )

    assert first.should_push_telegram is True
    assert second.cooldown_blocked is True
    assert second.should_push_telegram is False


def test_cooldown_allows_push_after_window() -> None:
    cooldown = SignalCooldownTracker(cooldown_minutes=10)
    ts = 1_000_000_000_000
    cooldown.record_push("BTCUSDT", SIGNAL_UP, ts)

    assert cooldown.is_blocked("BTCUSDT", SIGNAL_UP, ts + 9 * 60_000) is True
    assert cooldown.is_blocked("BTCUSDT", SIGNAL_UP, ts + 10 * 60_000) is False


def test_signal_engine_records_push_and_detects_reversal() -> None:
    config = SignalEngineConfig(confidence_threshold=0.2, signal_cooldown_minutes=0)
    engine = SignalEngine(config=config)
    ts = 1_700_000_000_000
    klines = _make_klines([100.0] * 30, volumes=[200.0] * 30)
    orderbook = _orderbook_snapshot()

    up_prediction = _successful_prediction(direction=DIRECTION_UP, predicted_return=0.01)
    down_prediction = _successful_prediction(
        direction=DIRECTION_DOWN,
        predicted_return=-0.01,
        interval_lower=-0.012,
        interval_upper=-0.004,
    )

    up_signal = engine.evaluate(
        up_prediction,
        symbol="BTCUSDT",
        timestamp_ms=ts,
        klines=klines,
        orderbook=orderbook,
    )
    down_signal = engine.evaluate(
        down_prediction,
        symbol="BTCUSDT",
        timestamp_ms=ts + 600_001,
        klines=klines,
        orderbook=orderbook,
    )

    assert up_signal.should_push_telegram is True
    assert down_signal.should_push_telegram is True
    assert down_signal.is_direction_reversal is True


def test_signal_engine_config_from_settings() -> None:
    settings = Settings(
        symbol="BTCUSDT",
        interval="1m",
        prediction_minutes=10,
        arima_order=(1, 0, 1),
        arima_series_type="log_return",
        use_auto_arima=False,
        auto_arima_max_p=5,
        auto_arima_max_q=5,
        auto_arima_max_d=2,
        direction_threshold=0.0001,
        train_window=1440,
        refit_interval_minutes=5,
        confidence_threshold=0.75,
        signal_cooldown_minutes=15,
        max_spread_bps=40.0,
        binance_market="spot",
        binance_api_key=None,
        binance_api_secret=None,
        binance_testnet=False,
        telegram_bot_token=None,
        telegram_chat_id=None,
        dry_run=True,
        log_level="INFO",
        live_poll_interval_seconds=10.0,
        live_kline_limit=2,
        live_max_retries=5,
        live_retry_backoff=1.0,
        live_max_consecutive_errors=10,
        live_error_retry_delay_seconds=5.0,
        data_dir=Settings.from_environ().data_dir,
        logs_dir=Settings.from_environ().logs_dir,
        project_root=Settings.from_environ().project_root,
    )
    config = SignalEngineConfig.from_settings(settings)
    engine = SignalEngine.from_settings(settings)

    assert config.confidence_threshold == 0.75
    assert config.signal_cooldown_minutes == 15
    assert config.max_spread_bps == 40.0
    assert engine.config.confidence_threshold == 0.75


def test_interval_score_partial_when_bounds_straddle_zero() -> None:
    prediction = _successful_prediction(
        direction=DIRECTION_UP,
        predicted_return=0.001,
        interval_lower=-0.0005,
        interval_upper=0.002,
    )
    components, _, _, _, _ = compute_confidence_components(
        prediction,
        config=SignalEngineConfig(),
        arima_direction=SIGNAL_UP,
    )
    assert components.interval == pytest.approx(0.6)
