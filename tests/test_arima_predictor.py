"""Tests for the ARIMA prediction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.arima_predictor import (
    ARIMAErrorCode,
    ARIMAPredictorConfig,
    DIRECTION_DOWN,
    DIRECTION_HOLD,
    DIRECTION_UP,
    SeriesType,
    build_model_series,
    predict_from_klines,
)
from src.utils.config import Settings


def _make_klines(
    closes: list[float],
    *,
    start_ts: int = 1_000_000,
    step_ms: int = 60_000,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    if volumes is None:
        volumes = [10.0 + index for index in range(len(closes))]

    rows = []
    for index, close in enumerate(closes):
        ts = start_ts + index * step_ms
        rows.append(
            {
                "timestamp": ts,
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


def _trending_closes(length: int, *, start: float = 100.0, step: float = 0.05) -> list[float]:
    return [start + index * step for index in range(length)]


def test_build_model_series_log_return() -> None:
    klines = _make_klines([100.0, 101.0, 102.0])
    series = build_model_series(klines, series_type=SeriesType.LOG_RETURN)

    assert len(series) == 2
    assert np.isclose(series.iloc[0], np.log(101.0 / 100.0))
    assert np.isclose(series.iloc[1], np.log(102.0 / 101.0))


def test_build_model_series_price_diff() -> None:
    klines = _make_klines([100.0, 101.0, 99.5])
    series = build_model_series(klines, series_type=SeriesType.PRICE_DIFF)

    assert len(series) == 2
    assert series.iloc[0] == pytest.approx(1.0)
    assert series.iloc[1] == pytest.approx(-1.5)


def test_predict_from_klines_success_fixed_order() -> None:
    klines = _make_klines(_trending_closes(120))
    config = ARIMAPredictorConfig(
        prediction_minutes=10,
        order=(1, 0, 1),
        series_type=SeriesType.LOG_RETURN,
        direction_threshold=0.0,
        min_train_points=30,
    )

    result = predict_from_klines(klines, train_window=100, config=config)

    assert result.success is True
    assert result.error_code is None
    assert result.predicted_cumulative_return is not None
    assert result.direction in {DIRECTION_UP, DIRECTION_DOWN, DIRECTION_HOLD}
    assert result.interval_lower is not None
    assert result.interval_upper is not None
    assert result.interval_lower <= result.interval_upper
    assert result.residual_volatility is not None
    assert result.residual_volatility >= 0.0
    assert result.model_order == (1, 0, 1)
    assert result.series_type == SeriesType.LOG_RETURN.value
    assert result.current_price == pytest.approx(klines.iloc[-1]["close"])
    assert result.prediction_horizon_minutes == 10
    assert result.forecast_steps == 10
    assert result.train_points is not None
    assert result.train_points >= 30


def test_predict_from_klines_price_diff_series() -> None:
    klines = _make_klines(_trending_closes(120))
    config = ARIMAPredictorConfig(
        prediction_minutes=5,
        order=(1, 0, 0),
        series_type=SeriesType.PRICE_DIFF,
    )

    result = predict_from_klines(klines, train_window=80, config=config)

    assert result.success is True
    assert result.series_type == SeriesType.PRICE_DIFF.value


def test_predict_from_klines_insufficient_data() -> None:
    klines = _make_klines(_trending_closes(20))
    config = ARIMAPredictorConfig(min_train_points=30)

    result = predict_from_klines(klines, train_window=20, config=config)

    assert result.success is False
    assert result.error_code == ARIMAErrorCode.INSUFFICIENT_DATA.value


def test_predict_from_klines_constant_series_fails_gracefully() -> None:
    klines = _make_klines([100.0] * 80)
    config = ARIMAPredictorConfig(min_train_points=30)

    result = predict_from_klines(klines, train_window=80, config=config)

    assert result.success is False
    assert result.error_code == ARIMAErrorCode.INSUFFICIENT_DATA.value
    assert "constant" in (result.error_message or "").lower()


def test_predict_from_klines_fit_failure_does_not_raise() -> None:
    klines = _make_klines(_trending_closes(80))
    config = ARIMAPredictorConfig(order=(1, 0, 1))

    with patch(
        "src.models.arima_predictor._fit_fixed_arima",
        side_effect=RuntimeError("singular matrix"),
    ):
        result = predict_from_klines(klines, train_window=60, config=config)

    assert result.success is False
    assert result.error_code == ARIMAErrorCode.FIT_FAILED.value
    assert "singular matrix" in (result.error_detail or "")


def test_predict_from_klines_forecast_failure_does_not_raise() -> None:
    klines = _make_klines(_trending_closes(80))
    config = ARIMAPredictorConfig(order=(1, 0, 1))
    fitted = MagicMock()

    with patch("src.models.arima_predictor._fit_fixed_arima", return_value=fitted):
        with patch(
            "src.models.arima_predictor._forecast_fixed",
            side_effect=ValueError("forecast exploded"),
        ):
            result = predict_from_klines(klines, train_window=60, config=config)

    assert result.success is False
    assert result.error_code == ARIMAErrorCode.FORECAST_FAILED.value


def test_predict_from_klines_auto_arima_success() -> None:
    klines = _make_klines(_trending_closes(120))
    config = ARIMAPredictorConfig(
        use_auto_arima=True,
        prediction_minutes=10,
        direction_threshold=0.0,
    )
    auto_model = MagicMock()
    auto_model.order = (2, 0, 1)
    auto_model.resid = np.array([0.01, -0.02, 0.015, -0.005])
    auto_model.predict.return_value = (
        np.full(10, 0.001),
        np.column_stack([np.full(10, -0.002), np.full(10, 0.004)]),
    )

    with patch("src.models.arima_predictor._fit_auto_arima", return_value=auto_model):
        result = predict_from_klines(klines, train_window=100, config=config)

    assert result.success is True
    assert result.model_order == (2, 0, 1)
    assert result.predicted_cumulative_return == pytest.approx(0.01)
    assert result.direction == DIRECTION_UP


def test_predict_from_klines_auto_arima_failure() -> None:
    klines = _make_klines(_trending_closes(80))
    config = ARIMAPredictorConfig(use_auto_arima=True)

    with patch(
        "src.models.arima_predictor._fit_auto_arima",
        side_effect=RuntimeError("auto search failed"),
    ):
        result = predict_from_klines(klines, train_window=60, config=config)

    assert result.success is False
    assert result.error_code == ARIMAErrorCode.AUTO_ARIMA_FAILED.value


def test_direction_threshold_hold() -> None:
    klines = _make_klines(_trending_closes(80))
    config = ARIMAPredictorConfig(order=(1, 0, 0), direction_threshold=1.0)

    fitted = MagicMock()
    forecast = MagicMock()
    forecast.predicted_mean = np.full(10, 0.0001)
    forecast.conf_int.return_value = np.column_stack(
        [np.full(10, -0.001), np.full(10, 0.001)]
    )
    fitted.get_forecast.return_value = forecast
    fitted.resid = np.array([0.01, -0.01, 0.02, -0.02])

    with patch("src.models.arima_predictor._fit_fixed_arima", return_value=fitted):
        result = predict_from_klines(klines, train_window=60, config=config)

    assert result.success is True
    assert result.direction == DIRECTION_HOLD


def test_arima_predictor_config_from_settings() -> None:
    settings = Settings(
        symbol="BTCUSDT",
        interval="1m",
        prediction_minutes=10,
        arima_order=(2, 0, 1),
        arima_series_type="price_diff",
        use_auto_arima=True,
        auto_arima_max_p=4,
        auto_arima_max_q=3,
        auto_arima_max_d=1,
        direction_threshold=0.001,
        train_window=720,
        refit_interval_minutes=5,
        confidence_threshold=0.7,
        signal_cooldown_minutes=10,
        max_spread_bps=50.0,
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

    config = ARIMAPredictorConfig.from_settings(settings)

    assert config.prediction_minutes == 10
    assert config.order == (2, 0, 1)
    assert config.series_type is SeriesType.PRICE_DIFF
    assert config.use_auto_arima is True
    assert config.direction_threshold == 0.001
