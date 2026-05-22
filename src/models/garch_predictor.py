"""GARCH volatility forecasting for the 10-minute prediction horizon."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.models.arima_predictor import SeriesType, _prepare_klines, build_model_series

logger = logging.getLogger(__name__)

# Log returns are tiny; rescale for arch optimizer stability and undo before output.
_RETURN_SCALE = 10_000.0


class VolatilityLevel(str, Enum):
    """Discrete risk bucket derived from in-sample volatility percentiles."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class GARCHErrorCode(str, Enum):
    """Machine-readable failure codes for downstream logging and filtering."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    FIT_FAILED = "FIT_FAILED"
    FORECAST_FAILED = "FORECAST_FAILED"


@dataclass(frozen=True)
class GARCHPredictorConfig:
    """Parameters controlling GARCH fit and volatility forecast aggregation."""

    prediction_minutes: int = 10
    order: Tuple[int, int] = (1, 1)
    mean: str = "constant"
    dist: str = "normal"
    min_train_points: int = 100
    vol_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.prediction_minutes < 1:
            raise ValueError(
                f"prediction_minutes must be >= 1, got {self.prediction_minutes}"
            )
        p, q = self.order
        if min(p, q) < 0:
            raise ValueError(f"GARCH order values must be non-negative, got {self.order}")
        if self.min_train_points < 10:
            raise ValueError(
                f"min_train_points must be >= 10, got {self.min_train_points}"
            )
        if self.vol_scale <= 0:
            raise ValueError(f"vol_scale must be positive, got {self.vol_scale}")

    @classmethod
    def from_settings(cls, settings) -> GARCHPredictorConfig:
        """Build predictor config from application Settings."""
        return cls(
            prediction_minutes=settings.prediction_minutes,
            order=settings.garch_order,
            mean=getattr(settings, "garch_mean", "constant"),
            dist=getattr(settings, "garch_dist", "normal"),
            min_train_points=getattr(settings, "garch_min_train_points", 100),
            vol_scale=getattr(settings, "garch_vol_scale", 1.0),
        )


@dataclass(frozen=True)
class GARCHPredictionResult:
    """Structured GARCH volatility output consumed by the aggregator and backtest."""

    success: bool
    conditional_volatility: Optional[float] = None
    forecast_volatility: Optional[Tuple[float, ...]] = None
    cumulative_volatility: Optional[float] = None
    volatility_level: Optional[str] = None
    model_order: Optional[Tuple[int, int]] = None
    train_points: Optional[int] = None
    current_price: Optional[float] = None
    prediction_horizon_minutes: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_detail: Optional[str] = None

    @classmethod
    def failure(
        cls,
        *,
        error_code: GARCHErrorCode | str,
        error_message: str,
        error_detail: Optional[str] = None,
        current_price: Optional[float] = None,
        prediction_horizon_minutes: Optional[int] = None,
        train_points: Optional[int] = None,
        model_order: Optional[Tuple[int, int]] = None,
    ) -> GARCHPredictionResult:
        code = error_code.value if isinstance(error_code, GARCHErrorCode) else str(error_code)
        return cls(
            success=False,
            current_price=current_price,
            prediction_horizon_minutes=prediction_horizon_minutes,
            train_points=train_points,
            model_order=model_order,
            error_code=code,
            error_message=error_message,
            error_detail=error_detail,
        )


def _arch_mean(mean: str) -> str:
    mapping = {"constant": "Constant", "zero": "Zero"}
    return mapping.get(mean.lower(), "Constant")


def _arch_dist(dist: str) -> str:
    mapping = {"normal": "normal", "t": "t"}
    return mapping.get(dist.lower(), "normal")


def _classify_volatility_level(
    cumulative_volatility: float,
    historical_cumulative: np.ndarray,
) -> str:
    """Map cumulative volatility to LOW/NORMAL/HIGH/EXTREME using percentiles."""
    history = np.asarray(historical_cumulative, dtype=float)
    history = history[np.isfinite(history)]
    if history.size < 2:
        return VolatilityLevel.NORMAL.value

    p50, p80, p95 = np.percentile(history, [50, 80, 95])
    if cumulative_volatility < p50:
        return VolatilityLevel.LOW.value
    if cumulative_volatility < p80:
        return VolatilityLevel.NORMAL.value
    if cumulative_volatility < p95:
        return VolatilityLevel.HIGH.value
    return VolatilityLevel.EXTREME.value


def _fit_garch(series: pd.Series, config: GARCHPredictorConfig):
    from arch import arch_model

    p, q = config.order
    model = arch_model(
        series * _RETURN_SCALE,
        mean=_arch_mean(config.mean),
        vol="Garch",
        p=p,
        q=q,
        dist=_arch_dist(config.dist),
    )
    return model.fit(disp="off", show_warning=False)


def _forecast_garch(
    fitted_model,
    *,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    forecast = fitted_model.forecast(horizon=steps, reindex=False)
    variance = np.asarray(forecast.variance.iloc[-1], dtype=float)
    if variance.size != steps:
        raise ValueError(
            f"expected {steps} forecast variance steps, got {variance.size}"
        )
    scaled_variance = variance / (_RETURN_SCALE * _RETURN_SCALE)
    step_volatility = np.sqrt(np.maximum(scaled_variance, 0.0))
    return step_volatility, scaled_variance


def predict_volatility_from_klines(
    klines: pd.DataFrame,
    *,
    train_window: int,
    config: Optional[GARCHPredictorConfig] = None,
) -> GARCHPredictionResult:
    """
    Fit GARCH on the last ``train_window`` 1-minute log returns and forecast volatility.

    Returns a structured result. Failures are captured in the result object and never
    propagate as uncaught exceptions.
    """
    cfg = config or GARCHPredictorConfig()

    try:
        if train_window < cfg.min_train_points:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INSUFFICIENT_DATA,
                error_message=(
                    f"train_window ({train_window}) is smaller than min_train_points "
                    f"({cfg.min_train_points})"
                ),
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        if klines is None:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INVALID_INPUT,
                error_message="klines must not be None",
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        prepared = _prepare_klines(klines)
        if prepared.empty:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INSUFFICIENT_DATA,
                error_message="klines dataframe is empty",
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        window = prepared.tail(train_window)
        current_price = float(window.iloc[-1]["close"])
        if current_price <= 0:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INVALID_INPUT,
                error_message=f"latest close price must be positive, got {current_price}",
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        series = build_model_series(window, series_type=SeriesType.LOG_RETURN)
        if len(series) < cfg.min_train_points:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INSUFFICIENT_DATA,
                error_message=(
                    f"not enough valid training points after log_return transform: "
                    f"{len(series)} < {cfg.min_train_points}"
                ),
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
            )

        if series.nunique() <= 1:
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.INSUFFICIENT_DATA,
                error_message="log_return series is constant; cannot fit GARCH",
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
            )

        try:
            fitted = _fit_garch(series, cfg)
        except Exception as exc:
            logger.warning("GARCH fit failed for order %s: %s", cfg.order, exc, exc_info=True)
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.FIT_FAILED,
                error_message=f"GARCH fit failed for order {cfg.order}",
                error_detail=str(exc),
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
                model_order=cfg.order,
            )

        try:
            step_volatility, variance_steps = _forecast_garch(
                fitted,
                steps=cfg.prediction_minutes,
            )
        except Exception as exc:
            logger.warning("GARCH forecast failed: %s", exc, exc_info=True)
            return GARCHPredictionResult.failure(
                error_code=GARCHErrorCode.FORECAST_FAILED,
                error_message="GARCH forecast failed",
                error_detail=str(exc),
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
                model_order=cfg.order,
            )

        conditional_volatility = (
            float(np.asarray(fitted.conditional_volatility.iloc[-1], dtype=float))
            / _RETURN_SCALE
        )
        conditional_volatility *= cfg.vol_scale
        forecast_volatility = tuple(float(value) * cfg.vol_scale for value in step_volatility)
        cumulative_volatility = (
            float(np.sqrt(np.sum(variance_steps))) * cfg.vol_scale
        )

        in_sample_volatility = np.asarray(fitted.conditional_volatility, dtype=float) / _RETURN_SCALE
        historical_cumulative = in_sample_volatility * np.sqrt(cfg.prediction_minutes)
        volatility_level = _classify_volatility_level(
            cumulative_volatility,
            historical_cumulative,
        )

        return GARCHPredictionResult(
            success=True,
            conditional_volatility=conditional_volatility,
            forecast_volatility=forecast_volatility,
            cumulative_volatility=cumulative_volatility,
            volatility_level=volatility_level,
            model_order=cfg.order,
            train_points=len(series),
            current_price=current_price,
            prediction_horizon_minutes=cfg.prediction_minutes,
        )

    except Exception as exc:
        logger.exception("unexpected GARCH prediction failure")
        return GARCHPredictionResult.failure(
            error_code=GARCHErrorCode.INVALID_INPUT,
            error_message="unexpected GARCH prediction failure",
            error_detail=str(exc),
            prediction_horizon_minutes=cfg.prediction_minutes,
        )
