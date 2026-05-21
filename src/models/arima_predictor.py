"""ARIMA training and multi-step forecasting for 10-minute horizon predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data.kline_schema import KLINE_COLUMNS
from src.features.engineering import LABEL_DOWN, LABEL_FLAT, LABEL_UP

logger = logging.getLogger(__name__)

DIRECTION_UP = LABEL_UP
DIRECTION_DOWN = LABEL_DOWN
DIRECTION_HOLD = "HOLD"


class SeriesType(str, Enum):
    """Target series used to fit the ARIMA model."""

    LOG_RETURN = "log_return"
    PRICE_DIFF = "price_diff"


class ARIMAErrorCode(str, Enum):
    """Machine-readable failure codes for downstream logging and filtering."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    FIT_FAILED = "FIT_FAILED"
    FORECAST_FAILED = "FORECAST_FAILED"
    AUTO_ARIMA_FAILED = "AUTO_ARIMA_FAILED"


@dataclass(frozen=True)
class ARIMAPredictorConfig:
    """Parameters controlling ARIMA fit and forecast aggregation."""

    prediction_minutes: int = 10
    order: Tuple[int, int, int] = (1, 0, 1)
    series_type: SeriesType = SeriesType.LOG_RETURN
    use_auto_arima: bool = False
    auto_arima_max_p: int = 5
    auto_arima_max_q: int = 5
    auto_arima_max_d: int = 2
    direction_threshold: float = 0.0
    min_train_points: int = 30
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.prediction_minutes < 1:
            raise ValueError(
                f"prediction_minutes must be >= 1, got {self.prediction_minutes}"
            )
        if self.min_train_points < 10:
            raise ValueError(
                f"min_train_points must be >= 10, got {self.min_train_points}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.direction_threshold < 0:
            raise ValueError(
                f"direction_threshold must be non-negative, got {self.direction_threshold}"
            )

    @classmethod
    def from_settings(cls, settings) -> ARIMAPredictorConfig:
        """Build predictor config from application Settings."""
        series_raw = getattr(settings, "arima_series_type", "log_return")
        try:
            series_type = SeriesType(str(series_raw).lower())
        except ValueError as exc:
            raise ValueError(
                f"arima_series_type must be one of {[item.value for item in SeriesType]}, "
                f"got {series_raw!r}"
            ) from exc

        return cls(
            prediction_minutes=settings.prediction_minutes,
            order=settings.arima_order,
            series_type=series_type,
            use_auto_arima=getattr(settings, "use_auto_arima", False),
            auto_arima_max_p=getattr(settings, "auto_arima_max_p", 5),
            auto_arima_max_q=getattr(settings, "auto_arima_max_q", 5),
            auto_arima_max_d=getattr(settings, "auto_arima_max_d", 2),
            direction_threshold=getattr(settings, "direction_threshold", 0.0),
        )


@dataclass(frozen=True)
class ARIMAPredictionResult:
    """Structured ARIMA output consumed by the signal engine and backtest."""

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

    @classmethod
    def failure(
        cls,
        *,
        error_code: ARIMAErrorCode | str,
        error_message: str,
        error_detail: Optional[str] = None,
        series_type: Optional[str] = None,
        current_price: Optional[float] = None,
        prediction_horizon_minutes: Optional[int] = None,
        train_points: Optional[int] = None,
        model_order: Optional[Tuple[int, int, int]] = None,
    ) -> ARIMAPredictionResult:
        code = error_code.value if isinstance(error_code, ARIMAErrorCode) else str(error_code)
        return cls(
            success=False,
            series_type=series_type,
            current_price=current_price,
            prediction_horizon_minutes=prediction_horizon_minutes,
            train_points=train_points,
            model_order=model_order,
            error_code=code,
            error_message=error_message,
            error_detail=error_detail,
        )


def _require_kline_columns(klines: pd.DataFrame) -> None:
    missing = [column for column in KLINE_COLUMNS if column not in klines.columns]
    if missing:
        raise ValueError(f"klines is missing required columns: {missing}")


def _prepare_klines(klines: pd.DataFrame) -> pd.DataFrame:
    if klines.empty:
        return pd.DataFrame(columns=list(KLINE_COLUMNS))

    _require_kline_columns(klines)
    frame = klines.copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    for column in KLINE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def build_model_series(
    klines: pd.DataFrame,
    *,
    series_type: SeriesType,
) -> pd.Series:
    """Convert close prices into the ARIMA modeling series."""
    frame = _prepare_klines(klines)
    close = frame["close"]
    if series_type is SeriesType.LOG_RETURN:
        series = np.log(close / close.shift(1))
    else:
        series = close.diff()
    return series.dropna().astype(float)


def _direction_from_return(
    cumulative_return: float,
    *,
    threshold: float,
) -> str:
    if cumulative_return > threshold:
        return DIRECTION_UP
    if cumulative_return < -threshold:
        return DIRECTION_DOWN
    return DIRECTION_HOLD


def _to_cumulative_return(
    step_forecast: np.ndarray,
    interval_lower_steps: np.ndarray,
    interval_upper_steps: np.ndarray,
    *,
    series_type: SeriesType,
    current_price: float,
) -> tuple[float, float, float]:
    cumulative_step = float(np.sum(step_forecast))
    cumulative_lower_steps = float(np.sum(interval_lower_steps))
    cumulative_upper_steps = float(np.sum(interval_upper_steps))

    if series_type is SeriesType.LOG_RETURN:
        return cumulative_step, cumulative_lower_steps, cumulative_upper_steps

    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    return (
        cumulative_step / current_price,
        cumulative_lower_steps / current_price,
        cumulative_upper_steps / current_price,
    )


def _fit_fixed_arima(
    series: pd.Series,
    order: Tuple[int, int, int],
):
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(series, order=order)
    return model.fit()


def _fit_auto_arima(series: pd.Series, config: ARIMAPredictorConfig):
    import pmdarima as pm

    return pm.auto_arima(
        series,
        start_p=0,
        start_q=0,
        max_p=config.auto_arima_max_p,
        max_q=config.auto_arima_max_q,
        max_d=config.auto_arima_max_d,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )


def _forecast_fixed(
    fitted_model,
    *,
    steps: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    forecast = fitted_model.get_forecast(steps=steps)
    step_forecast = np.asarray(forecast.predicted_mean, dtype=float)
    conf_int = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
    residuals = np.asarray(fitted_model.resid, dtype=float)
    residual_volatility = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    return step_forecast, conf_int[:, 0], conf_int[:, 1], residual_volatility


def _forecast_auto(
    auto_model,
    *,
    steps: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    step_forecast, conf_int = auto_model.predict(
        n_periods=steps,
        return_conf_int=True,
        alpha=alpha,
    )
    step_forecast = np.asarray(step_forecast, dtype=float)
    conf_int = np.asarray(conf_int, dtype=float)
    residuals = np.asarray(getattr(auto_model, "resid", ()), dtype=float)
    residual_volatility = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    return step_forecast, conf_int[:, 0], conf_int[:, 1], residual_volatility


def predict_from_klines(
    klines: pd.DataFrame,
    *,
    train_window: int,
    config: Optional[ARIMAPredictorConfig] = None,
) -> ARIMAPredictionResult:
    """
    Fit ARIMA on the last ``train_window`` 1-minute bars and forecast the next horizon.

    Returns a structured result. Failures are captured in the result object and never
    propagate as uncaught exceptions.
    """
    cfg = config or ARIMAPredictorConfig()

    try:
        if train_window < cfg.min_train_points:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INSUFFICIENT_DATA,
                error_message=(
                    f"train_window ({train_window}) is smaller than min_train_points "
                    f"({cfg.min_train_points})"
                ),
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        if klines is None:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INVALID_INPUT,
                error_message="klines must not be None",
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        prepared = _prepare_klines(klines)
        if prepared.empty:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INSUFFICIENT_DATA,
                error_message="klines dataframe is empty",
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        window = prepared.tail(train_window)
        current_price = float(window.iloc[-1]["close"])
        if current_price <= 0:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INVALID_INPUT,
                error_message=f"latest close price must be positive, got {current_price}",
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
            )

        series = build_model_series(window, series_type=cfg.series_type)
        if len(series) < cfg.min_train_points:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INSUFFICIENT_DATA,
                error_message=(
                    f"not enough valid training points after series transform: "
                    f"{len(series)} < {cfg.min_train_points}"
                ),
                series_type=cfg.series_type.value,
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
            )

        if series.nunique() <= 1:
            return ARIMAPredictionResult.failure(
                error_code=ARIMAErrorCode.INSUFFICIENT_DATA,
                error_message="modeling series is constant; cannot fit ARIMA",
                series_type=cfg.series_type.value,
                current_price=current_price,
                prediction_horizon_minutes=cfg.prediction_minutes,
                train_points=len(series),
            )

        alpha = 1.0 - cfg.confidence_level
        model_order = cfg.order

        if cfg.use_auto_arima:
            try:
                auto_model = _fit_auto_arima(series, cfg)
            except Exception as exc:
                logger.warning("auto_arima fit failed: %s", exc, exc_info=True)
                return ARIMAPredictionResult.failure(
                    error_code=ARIMAErrorCode.AUTO_ARIMA_FAILED,
                    error_message="auto_arima failed to select and fit a model",
                    error_detail=str(exc),
                    series_type=cfg.series_type.value,
                    current_price=current_price,
                    prediction_horizon_minutes=cfg.prediction_minutes,
                    train_points=len(series),
                )

            if auto_model is None:
                return ARIMAPredictionResult.failure(
                    error_code=ARIMAErrorCode.AUTO_ARIMA_FAILED,
                    error_message="auto_arima returned no model",
                    series_type=cfg.series_type.value,
                    current_price=current_price,
                    prediction_horizon_minutes=cfg.prediction_minutes,
                    train_points=len(series),
                )

            model_order = tuple(int(part) for part in auto_model.order)
            try:
                step_forecast, lower_steps, upper_steps, residual_volatility = _forecast_auto(
                    auto_model,
                    steps=cfg.prediction_minutes,
                    alpha=alpha,
                )
            except Exception as exc:
                logger.warning("auto_arima forecast failed: %s", exc, exc_info=True)
                return ARIMAPredictionResult.failure(
                    error_code=ARIMAErrorCode.FORECAST_FAILED,
                    error_message="auto_arima forecast failed",
                    error_detail=str(exc),
                    series_type=cfg.series_type.value,
                    current_price=current_price,
                    prediction_horizon_minutes=cfg.prediction_minutes,
                    train_points=len(series),
                    model_order=model_order,
                )
        else:
            try:
                fitted = _fit_fixed_arima(series, cfg.order)
            except Exception as exc:
                logger.warning("ARIMA fit failed for order %s: %s", cfg.order, exc, exc_info=True)
                return ARIMAPredictionResult.failure(
                    error_code=ARIMAErrorCode.FIT_FAILED,
                    error_message=f"ARIMA fit failed for order {cfg.order}",
                    error_detail=str(exc),
                    series_type=cfg.series_type.value,
                    current_price=current_price,
                    prediction_horizon_minutes=cfg.prediction_minutes,
                    train_points=len(series),
                    model_order=cfg.order,
                )

            try:
                step_forecast, lower_steps, upper_steps, residual_volatility = _forecast_fixed(
                    fitted,
                    steps=cfg.prediction_minutes,
                    alpha=alpha,
                )
            except Exception as exc:
                logger.warning("ARIMA forecast failed: %s", exc, exc_info=True)
                return ARIMAPredictionResult.failure(
                    error_code=ARIMAErrorCode.FORECAST_FAILED,
                    error_message="ARIMA forecast failed",
                    error_detail=str(exc),
                    series_type=cfg.series_type.value,
                    current_price=current_price,
                    prediction_horizon_minutes=cfg.prediction_minutes,
                    train_points=len(series),
                    model_order=cfg.order,
                )

        cumulative_return, interval_lower, interval_upper = _to_cumulative_return(
            step_forecast,
            lower_steps,
            upper_steps,
            series_type=cfg.series_type,
            current_price=current_price,
        )
        direction = _direction_from_return(
            cumulative_return,
            threshold=cfg.direction_threshold,
        )

        return ARIMAPredictionResult(
            success=True,
            predicted_cumulative_return=cumulative_return,
            direction=direction,
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            residual_volatility=residual_volatility,
            model_order=model_order,
            series_type=cfg.series_type.value,
            current_price=current_price,
            prediction_horizon_minutes=cfg.prediction_minutes,
            forecast_steps=cfg.prediction_minutes,
            train_points=len(series),
        )

    except Exception as exc:
        logger.exception("unexpected ARIMA prediction failure")
        return ARIMAPredictionResult.failure(
            error_code=ARIMAErrorCode.INVALID_INPUT,
            error_message="unexpected ARIMA prediction failure",
            error_detail=str(exc),
            prediction_horizon_minutes=cfg.prediction_minutes,
        )
