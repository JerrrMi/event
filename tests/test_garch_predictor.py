"""Tests for the GARCH volatility prediction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.garch_predictor import (
    GARCHErrorCode,
    GARCHPredictorConfig,
    VolatilityLevel,
    predict_volatility_from_klines,
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


def _volatile_closes(length: int, *, start: float = 100.0) -> list[float]:
    rng = np.random.default_rng(42)
    prices = [start]
    for _ in range(length - 1):
        shock = rng.normal(loc=0.0, scale=0.002)
        prices.append(prices[-1] * np.exp(shock))
    return prices


def test_predict_volatility_from_klines_success() -> None:
    klines = _make_klines(_volatile_closes(150))
    config = GARCHPredictorConfig(
        prediction_minutes=10,
        order=(1, 1),
        min_train_points=50,
    )

    result = predict_volatility_from_klines(klines, train_window=120, config=config)

    assert result.success is True
    assert result.error_code is None
    assert result.conditional_volatility is not None
    assert result.conditional_volatility >= 0.0
    assert result.forecast_volatility is not None
    assert len(result.forecast_volatility) == 10
    assert all(value >= 0.0 for value in result.forecast_volatility)
    assert result.cumulative_volatility is not None
    assert result.cumulative_volatility >= 0.0
    assert result.volatility_level in {
        VolatilityLevel.LOW.value,
        VolatilityLevel.NORMAL.value,
        VolatilityLevel.HIGH.value,
        VolatilityLevel.EXTREME.value,
    }
    assert result.model_order == (1, 1)
    assert result.current_price == pytest.approx(klines.iloc[-1]["close"])
    assert result.prediction_horizon_minutes == 10
    assert result.train_points is not None
    assert result.train_points >= 50


def test_predict_volatility_from_klines_insufficient_data() -> None:
    klines = _make_klines(_volatile_closes(40))
    config = GARCHPredictorConfig(min_train_points=50)

    result = predict_volatility_from_klines(klines, train_window=40, config=config)

    assert result.success is False
    assert result.error_code == GARCHErrorCode.INSUFFICIENT_DATA.value


def test_predict_volatility_from_klines_constant_series_fails_gracefully() -> None:
    klines = _make_klines([100.0] * 120)
    config = GARCHPredictorConfig(min_train_points=50)

    result = predict_volatility_from_klines(klines, train_window=120, config=config)

    assert result.success is False
    assert result.error_code == GARCHErrorCode.INSUFFICIENT_DATA.value
    assert "constant" in (result.error_message or "").lower()


def test_predict_volatility_from_klines_fit_failure_does_not_raise() -> None:
    klines = _make_klines(_volatile_closes(120))
    config = GARCHPredictorConfig(order=(1, 1), min_train_points=50)

    with patch(
        "src.models.garch_predictor._fit_garch",
        side_effect=RuntimeError("optimizer failed"),
    ):
        result = predict_volatility_from_klines(klines, train_window=100, config=config)

    assert result.success is False
    assert result.error_code == GARCHErrorCode.FIT_FAILED.value
    assert "optimizer failed" in (result.error_detail or "")


def test_predict_volatility_from_klines_forecast_failure_does_not_raise() -> None:
    klines = _make_klines(_volatile_closes(120))
    config = GARCHPredictorConfig(order=(1, 1), min_train_points=50)
    fitted = MagicMock()

    with patch("src.models.garch_predictor._fit_garch", return_value=fitted):
        with patch(
            "src.models.garch_predictor._forecast_garch",
            side_effect=ValueError("forecast exploded"),
        ):
            result = predict_volatility_from_klines(klines, train_window=100, config=config)

    assert result.success is False
    assert result.error_code == GARCHErrorCode.FORECAST_FAILED.value
    assert "forecast exploded" in (result.error_detail or "")


def test_garch_predictor_config_from_settings() -> None:
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
        direction_threshold=0.0,
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
        use_garch=True,
        garch_order=(2, 1),
        garch_mean="zero",
        garch_dist="t",
        garch_min_train_points=120,
        garch_vol_scale=1.5,
        garch_failure_mode="hold",
        aggregation_mode="volatility_adjusted_arima",
        aggregation_min_snr=0.8,
        garch_extreme_vol_action="hold",
        garch_vol_weight=0.35,
        data_dir=Settings.from_environ().data_dir,
        logs_dir=Settings.from_environ().logs_dir,
        project_root=Settings.from_environ().project_root,
    )

    config = GARCHPredictorConfig.from_settings(settings)

    assert config.prediction_minutes == 10
    assert config.order == (2, 1)
    assert config.mean == "zero"
    assert config.dist == "t"
    assert config.min_train_points == 120
    assert config.vol_scale == 1.5
