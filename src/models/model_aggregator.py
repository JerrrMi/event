"""Combine ARIMA direction forecasts with GARCH volatility risk filtering."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from src.models.arima_predictor import (
    ARIMAPredictionResult,
    DIRECTION_HOLD,
)
from src.models.garch_predictor import GARCHPredictionResult, VolatilityLevel

logger = logging.getLogger(__name__)

_EPSILON = 1e-12
ALLOWED_GARCH_FAILURE_MODES = frozenset({"hold", "fallback_to_arima"})
ALLOWED_GARCH_EXTREME_VOL_ACTIONS = frozenset({"hold", "allow_with_penalty"})
ALLOWED_AGGREGATION_MODES = frozenset({"volatility_adjusted_arima"})


@dataclass(frozen=True)
class AggregatorConfig:
    """Parameters controlling ARIMA-GARCH aggregation and risk filtering."""

    aggregation_mode: str = "volatility_adjusted_arima"
    aggregation_min_snr: float = 0.8
    garch_failure_mode: str = "hold"
    garch_extreme_vol_action: str = "hold"
    garch_vol_weight: float = 0.35
    prediction_minutes: int = 10
    volatility_epsilon: float = _EPSILON

    def __post_init__(self) -> None:
        if self.aggregation_mode not in ALLOWED_AGGREGATION_MODES:
            raise ValueError(
                f"aggregation_mode must be one of {sorted(ALLOWED_AGGREGATION_MODES)}, "
                f"got {self.aggregation_mode!r}"
            )
        if self.aggregation_min_snr < 0:
            raise ValueError(
                f"aggregation_min_snr must be non-negative, got {self.aggregation_min_snr}"
            )
        if self.garch_failure_mode not in ALLOWED_GARCH_FAILURE_MODES:
            raise ValueError(
                f"garch_failure_mode must be one of {sorted(ALLOWED_GARCH_FAILURE_MODES)}, "
                f"got {self.garch_failure_mode!r}"
            )
        if self.garch_extreme_vol_action not in ALLOWED_GARCH_EXTREME_VOL_ACTIONS:
            raise ValueError(
                f"garch_extreme_vol_action must be one of "
                f"{sorted(ALLOWED_GARCH_EXTREME_VOL_ACTIONS)}, "
                f"got {self.garch_extreme_vol_action!r}"
            )
        if not 0.0 <= self.garch_vol_weight <= 1.0:
            raise ValueError(
                f"garch_vol_weight must be in [0.0, 1.0], got {self.garch_vol_weight}"
            )
        if self.prediction_minutes < 1:
            raise ValueError(
                f"prediction_minutes must be >= 1, got {self.prediction_minutes}"
            )
        if self.volatility_epsilon <= 0:
            raise ValueError(
                f"volatility_epsilon must be positive, got {self.volatility_epsilon}"
            )

    @classmethod
    def from_settings(cls, settings) -> AggregatorConfig:
        """Build aggregator config from application Settings."""
        return cls(
            aggregation_mode=getattr(settings, "aggregation_mode", "volatility_adjusted_arima"),
            aggregation_min_snr=getattr(settings, "aggregation_min_snr", 0.8),
            garch_failure_mode=getattr(settings, "garch_failure_mode", "hold"),
            garch_extreme_vol_action=getattr(settings, "garch_extreme_vol_action", "hold"),
            garch_vol_weight=getattr(settings, "garch_vol_weight", 0.35),
            prediction_minutes=getattr(settings, "prediction_minutes", 10),
        )


@dataclass(frozen=True)
class CombinedPredictionResult:
    """ARIMA-GARCH aggregate output compatible with the signal engine core fields."""

    success: bool
    predicted_cumulative_return: Optional[float] = None
    direction: Optional[str] = None
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    residual_volatility: Optional[float] = None
    model_order: Optional[Tuple[int, int, int]] = None
    series_type: Optional[str] = None
    current_price: Optional[float] = None
    prediction_horizon_minutes: Optional[int] = None
    forecast_steps: Optional[int] = None
    train_points: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_detail: Optional[str] = None
    arima_direction: Optional[str] = None
    garch_volatility: Optional[float] = None
    volatility_level: Optional[str] = None
    aggregation_mode: Optional[str] = None
    adjusted_snr: Optional[float] = None
    rejection_reasons: Tuple[str, ...] = ()


def _effective_volatility(
    *,
    arima: ARIMAPredictionResult,
    garch: GARCHPredictionResult,
    config: AggregatorConfig,
    horizon_minutes: int,
) -> Optional[float]:
    """Blend ARIMA residual volatility with GARCH per-step volatility when available."""
    arima_vol = arima.residual_volatility
    garch_per_step: Optional[float] = None
    if garch.success and garch.cumulative_volatility is not None:
        garch_per_step = garch.cumulative_volatility / math.sqrt(max(horizon_minutes, 1))

    if garch_per_step is None:
        return arima_vol
    if arima_vol is None:
        return garch_per_step

    weight = config.garch_vol_weight
    return (1.0 - weight) * arima_vol + weight * garch_per_step


def _compute_adjusted_snr(
    predicted_return: Optional[float],
    cumulative_volatility: Optional[float],
    *,
    epsilon: float,
) -> Optional[float]:
    if predicted_return is None or cumulative_volatility is None:
        return None
    return abs(predicted_return) / max(cumulative_volatility, epsilon)


def _apply_garch_filters(
    *,
    arima_direction: str,
    adjusted_snr: Optional[float],
    volatility_level: Optional[str],
    config: AggregatorConfig,
) -> tuple[str, list[str]]:
    """Apply GARCH-based risk filters and return final direction plus reasons."""
    if arima_direction == DIRECTION_HOLD:
        return DIRECTION_HOLD, []

    direction = arima_direction
    rejection_reasons: list[str] = []

    if (
        volatility_level == VolatilityLevel.EXTREME.value
        and config.garch_extreme_vol_action == "hold"
    ):
        direction = DIRECTION_HOLD
        rejection_reasons.append("extreme_volatility")

    if adjusted_snr is not None and adjusted_snr < config.aggregation_min_snr:
        direction = DIRECTION_HOLD
        rejection_reasons.append("low_adjusted_snr")

    if (
        direction != DIRECTION_HOLD
        and volatility_level == VolatilityLevel.EXTREME.value
        and config.garch_extreme_vol_action == "allow_with_penalty"
    ):
        rejection_reasons.append("extreme_volatility_penalty")

    return direction, rejection_reasons


def _base_fields(
    arima: ARIMAPredictionResult,
    *,
    config: AggregatorConfig,
) -> dict:
    horizon = arima.prediction_horizon_minutes or config.prediction_minutes
    return {
        "predicted_cumulative_return": arima.predicted_cumulative_return,
        "interval_lower": arima.interval_lower,
        "interval_upper": arima.interval_upper,
        "model_order": arima.model_order,
        "series_type": arima.series_type,
        "current_price": arima.current_price,
        "prediction_horizon_minutes": horizon,
        "forecast_steps": arima.forecast_steps,
        "train_points": arima.train_points,
        "arima_direction": arima.direction,
        "aggregation_mode": config.aggregation_mode,
    }


def aggregate_predictions(
    arima: ARIMAPredictionResult,
    garch: GARCHPredictionResult,
    config: Optional[AggregatorConfig] = None,
) -> CombinedPredictionResult:
    """
    Merge ARIMA direction/mean forecasts with GARCH volatility risk filtering.

    ARIMA supplies direction and predicted return; GARCH supplies conditional volatility
    and risk gates (failure mode, extreme volatility, adjusted SNR threshold).
    """
    cfg = config or AggregatorConfig()
    base = _base_fields(arima, config=cfg)
    arima_direction = arima.direction or DIRECTION_HOLD

    if not arima.success:
        return CombinedPredictionResult(
            success=False,
            direction=DIRECTION_HOLD,
            rejection_reasons=("arima_failed",),
            error_code=arima.error_code,
            error_message=arima.error_message,
            error_detail=arima.error_detail,
            garch_volatility=garch.cumulative_volatility if garch.success else None,
            volatility_level=garch.volatility_level if garch.success else None,
            adjusted_snr=None,
            **base,
        )

    if arima_direction == DIRECTION_HOLD:
        return CombinedPredictionResult(
            success=True,
            direction=DIRECTION_HOLD,
            residual_volatility=arima.residual_volatility,
            garch_volatility=garch.cumulative_volatility if garch.success else None,
            volatility_level=garch.volatility_level if garch.success else None,
            adjusted_snr=_compute_adjusted_snr(
                arima.predicted_cumulative_return,
                garch.cumulative_volatility if garch.success else None,
                epsilon=cfg.volatility_epsilon,
            ),
            rejection_reasons=("arima_hold",),
            **base,
        )

    if not garch.success:
        if cfg.garch_failure_mode == "hold":
            return CombinedPredictionResult(
                success=True,
                direction=DIRECTION_HOLD,
                residual_volatility=arima.residual_volatility,
                garch_volatility=None,
                volatility_level=None,
                adjusted_snr=None,
                rejection_reasons=("garch_failed",),
                error_code=garch.error_code,
                error_message=garch.error_message,
                error_detail=garch.error_detail,
                **base,
            )

        return CombinedPredictionResult(
            success=True,
            direction=arima_direction,
            residual_volatility=arima.residual_volatility,
            garch_volatility=None,
            volatility_level=None,
            adjusted_snr=None,
            rejection_reasons=("garch_failed_fallback_to_arima",),
            error_code=garch.error_code,
            error_message=garch.error_message,
            error_detail=garch.error_detail,
            **base,
        )

    horizon = base["prediction_horizon_minutes"] or cfg.prediction_minutes
    adjusted_snr = _compute_adjusted_snr(
        arima.predicted_cumulative_return,
        garch.cumulative_volatility,
        epsilon=cfg.volatility_epsilon,
    )
    direction, rejection_reasons = _apply_garch_filters(
        arima_direction=arima_direction,
        adjusted_snr=adjusted_snr,
        volatility_level=garch.volatility_level,
        config=cfg,
    )

    return CombinedPredictionResult(
        success=True,
        direction=direction,
        residual_volatility=_effective_volatility(
            arima=arima,
            garch=garch,
            config=cfg,
            horizon_minutes=horizon,
        ),
        garch_volatility=garch.cumulative_volatility,
        volatility_level=garch.volatility_level,
        adjusted_snr=adjusted_snr,
        rejection_reasons=tuple(rejection_reasons),
        **base,
    )
