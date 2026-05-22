"""Tests for the ARIMA-GARCH model aggregator."""

from __future__ import annotations

import math

import pytest

from src.models.arima_predictor import (
    ARIMAErrorCode,
    ARIMAPredictionResult,
    DIRECTION_DOWN,
    DIRECTION_HOLD,
    DIRECTION_UP,
)
from src.models.garch_predictor import GARCHErrorCode, GARCHPredictionResult, VolatilityLevel
from src.models.model_aggregator import (
    AggregatorConfig,
    CombinedPredictionResult,
    aggregate_predictions,
)
from src.signals.signal_engine import SignalEngineConfig, compute_confidence_components
from src.utils.config import Settings


def _successful_arima(
    *,
    direction: str = DIRECTION_UP,
    predicted_return: float = 0.002,
    interval_lower: float = 0.0005,
    interval_upper: float = 0.003,
    residual_volatility: float = 0.0003,
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
        current_price=100_000.0,
        prediction_horizon_minutes=10,
        forecast_steps=10,
        train_points=120,
    )


def _successful_garch(
    *,
    cumulative_volatility: float = 0.001,
    volatility_level: str = VolatilityLevel.NORMAL.value,
) -> GARCHPredictionResult:
    return GARCHPredictionResult(
        success=True,
        conditional_volatility=0.0002,
        forecast_volatility=tuple([0.0002] * 10),
        cumulative_volatility=cumulative_volatility,
        volatility_level=volatility_level,
        model_order=(1, 1),
        train_points=120,
        current_price=100_000.0,
        prediction_horizon_minutes=10,
    )


def _failed_garch() -> GARCHPredictionResult:
    return GARCHPredictionResult.failure(
        error_code=GARCHErrorCode.FIT_FAILED,
        error_message="GARCH fit failed",
        prediction_horizon_minutes=10,
    )


def test_aggregate_successful_arima_and_garch_outputs_direction() -> None:
    arima = _successful_arima(direction=DIRECTION_UP, predicted_return=0.002)
    garch = _successful_garch(cumulative_volatility=0.001)
    config = AggregatorConfig(aggregation_min_snr=0.8)

    result = aggregate_predictions(arima, garch, config=config)

    assert result.success is True
    assert result.direction == DIRECTION_UP
    assert result.arima_direction == DIRECTION_UP
    assert result.predicted_cumulative_return == pytest.approx(0.002)
    assert result.garch_volatility == pytest.approx(0.001)
    assert result.volatility_level == VolatilityLevel.NORMAL.value
    assert result.adjusted_snr == pytest.approx(2.0)
    assert result.aggregation_mode == "volatility_adjusted_arima"
    assert result.rejection_reasons == ()
    assert result.residual_volatility == pytest.approx(
        0.35 * (0.001 / math.sqrt(10)) + 0.65 * 0.0003
    )


def test_aggregate_arima_failure_outputs_hold() -> None:
    arima = ARIMAPredictionResult.failure(
        error_code=ARIMAErrorCode.FIT_FAILED,
        error_message="ARIMA fit failed",
    )
    garch = _successful_garch()

    result = aggregate_predictions(arima, garch)

    assert result.success is False
    assert result.direction == DIRECTION_HOLD
    assert result.rejection_reasons == ("arima_failed",)
    assert result.error_code == ARIMAErrorCode.FIT_FAILED.value


def test_aggregate_garch_failure_hold_mode_outputs_hold() -> None:
    arima = _successful_arima(direction=DIRECTION_DOWN, predicted_return=-0.002)
    config = AggregatorConfig(garch_failure_mode="hold")

    result = aggregate_predictions(arima, _failed_garch(), config=config)

    assert result.success is True
    assert result.direction == DIRECTION_HOLD
    assert result.rejection_reasons == ("garch_failed",)
    assert result.error_code == GARCHErrorCode.FIT_FAILED.value


def test_aggregate_garch_failure_fallback_preserves_arima_direction() -> None:
    arima = _successful_arima(direction=DIRECTION_DOWN, predicted_return=-0.002)
    config = AggregatorConfig(garch_failure_mode="fallback_to_arima")

    result = aggregate_predictions(arima, _failed_garch(), config=config)

    assert result.success is True
    assert result.direction == DIRECTION_DOWN
    assert result.arima_direction == DIRECTION_DOWN
    assert result.residual_volatility == pytest.approx(0.0003)
    assert result.rejection_reasons == ("garch_failed_fallback_to_arima",)
    assert result.adjusted_snr is None


def test_aggregate_extreme_volatility_hold_action_outputs_hold() -> None:
    arima = _successful_arima(direction=DIRECTION_UP, predicted_return=0.002)
    garch = _successful_garch(
        cumulative_volatility=0.001,
        volatility_level=VolatilityLevel.EXTREME.value,
    )
    config = AggregatorConfig(
        aggregation_min_snr=0.1,
        garch_extreme_vol_action="hold",
    )

    result = aggregate_predictions(arima, garch, config=config)

    assert result.direction == DIRECTION_HOLD
    assert "extreme_volatility" in result.rejection_reasons


def test_aggregate_low_adjusted_snr_outputs_hold() -> None:
    arima = _successful_arima(direction=DIRECTION_UP, predicted_return=0.0001)
    garch = _successful_garch(cumulative_volatility=0.001)
    config = AggregatorConfig(aggregation_min_snr=0.8)

    result = aggregate_predictions(arima, garch, config=config)

    assert result.direction == DIRECTION_HOLD
    assert result.adjusted_snr == pytest.approx(0.1)
    assert "low_adjusted_snr" in result.rejection_reasons


def test_combined_prediction_result_compatible_with_signal_engine() -> None:
    arima = _successful_arima(direction=DIRECTION_UP, predicted_return=0.002)
    garch = _successful_garch(cumulative_volatility=0.001)
    combined = aggregate_predictions(arima, garch)

    components, volume_passed, spread_passed, _, _ = compute_confidence_components(
        combined,
        config=SignalEngineConfig(confidence_threshold=0.5, direction_threshold=0.0),
    )

    assert isinstance(combined, CombinedPredictionResult)
    assert combined.success is True
    assert combined.direction == DIRECTION_UP
    assert combined.predicted_cumulative_return is not None
    assert combined.residual_volatility is not None
    assert combined.interval_lower is not None
    assert combined.interval_upper is not None
    assert components.snr > 0.0
    assert volume_passed is True
    assert spread_passed is True


def test_aggregator_config_from_settings() -> None:
    settings = Settings.from_environ()
    config = AggregatorConfig.from_settings(settings)

    assert config.aggregation_mode == settings.aggregation_mode
    assert config.aggregation_min_snr == settings.aggregation_min_snr
    assert config.garch_failure_mode == settings.garch_failure_mode
    assert config.garch_extreme_vol_action == settings.garch_extreme_vol_action
    assert config.garch_vol_weight == settings.garch_vol_weight
    assert config.prediction_minutes == settings.prediction_minutes


def test_aggregator_config_validation_rejects_invalid_failure_mode() -> None:
    with pytest.raises(ValueError, match="garch_failure_mode"):
        AggregatorConfig(garch_failure_mode="ignore")
