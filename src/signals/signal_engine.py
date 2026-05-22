"""Convert ARIMA predictions into filtered UP / DOWN / HOLD trading signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.kline_schema import KLINE_COLUMNS
from src.data.order_book_schema import ORDER_BOOK_COLUMNS, OrderBookSnapshot
from src.features.engineering import LABEL_DOWN, LABEL_UP
from src.models.arima_predictor import (
    ARIMAPredictionResult,
    DIRECTION_DOWN,
    DIRECTION_HOLD,
    DIRECTION_UP,
)
from src.models.model_aggregator import CombinedPredictionResult

logger = logging.getLogger(__name__)

PredictionInput = ARIMAPredictionResult | CombinedPredictionResult

SIGNAL_UP = LABEL_UP
SIGNAL_DOWN = LABEL_DOWN
SIGNAL_HOLD = DIRECTION_HOLD

_EPSILON = 1e-12


@dataclass(frozen=True)
class ConfidenceComponents:
    """Per-factor confidence contributions in [0, 1]."""

    magnitude: float
    snr: float
    interval: float
    volume: float
    spread: float
    imbalance: float

    def weighted_score(self, weights: Tuple[float, ...]) -> float:
        values = (
            self.magnitude,
            self.snr,
            self.interval,
            self.volume,
            self.spread,
            self.imbalance,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.0
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight


@dataclass(frozen=True)
class SignalEngineConfig:
    """Parameters for direction mapping, confidence scoring, and push gating."""

    confidence_threshold: float = 0.70
    signal_cooldown_minutes: int = 10
    max_spread_bps: float = 50.0
    direction_threshold: float = 0.0
    prediction_minutes: int = 10
    min_volume_ratio: float = 0.5
    volume_lookback: int = 20
    component_weights: Tuple[float, float, float, float, float, float] = (
        0.20,
        0.25,
        0.20,
        0.15,
        0.10,
        0.10,
    )
    snr_scale: float = 2.0
    magnitude_scale: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
        if self.signal_cooldown_minutes < 0:
            raise ValueError(
                f"signal_cooldown_minutes must be non-negative, "
                f"got {self.signal_cooldown_minutes}"
            )
        if self.max_spread_bps <= 0:
            raise ValueError(f"max_spread_bps must be positive, got {self.max_spread_bps}")
        if self.direction_threshold < 0:
            raise ValueError(
                f"direction_threshold must be non-negative, got {self.direction_threshold}"
            )
        if self.prediction_minutes < 1:
            raise ValueError(
                f"prediction_minutes must be >= 1, got {self.prediction_minutes}"
            )
        if not 0.0 < self.min_volume_ratio <= 1.0:
            raise ValueError(
                f"min_volume_ratio must be in (0, 1], got {self.min_volume_ratio}"
            )
        if self.volume_lookback < 1:
            raise ValueError(
                f"volume_lookback must be >= 1, got {self.volume_lookback}"
            )
        if len(self.component_weights) != 6:
            raise ValueError("component_weights must have 6 values")
        if any(weight < 0 for weight in self.component_weights):
            raise ValueError("component_weights must be non-negative")

    @classmethod
    def from_settings(cls, settings) -> SignalEngineConfig:
        return cls(
            confidence_threshold=settings.confidence_threshold,
            signal_cooldown_minutes=settings.signal_cooldown_minutes,
            max_spread_bps=settings.max_spread_bps,
            direction_threshold=settings.direction_threshold,
            prediction_minutes=settings.prediction_minutes,
            min_volume_ratio=getattr(settings, "min_volume_ratio", 0.5),
            volume_lookback=getattr(settings, "volume_lookback", 20),
        )


@dataclass
class TradingSignal:
    """Structured signal consumed by Telegram, logging, and backtest."""

    symbol: str
    timestamp_ms: int
    current_price: Optional[float]
    expiry_timestamp_ms: int
    direction: str
    predicted_cumulative_return: Optional[float]
    confidence: float
    confidence_threshold: float
    should_push_telegram: bool
    is_direction_reversal: bool
    arima_model_order: Optional[Tuple[int, int, int]]
    spread_bps: Optional[float]
    book_imbalance: Optional[float]
    volume_filter_passed: bool
    spread_filter_passed: bool
    cooldown_blocked: bool
    trigger_summary: str
    risk_note: str
    components: ConfidenceComponents
    arima_direction: Optional[str] = None
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)
    garch_volatility: Optional[float] = None
    volatility_level: Optional[str] = None
    aggregation_mode: Optional[str] = None
    adjusted_snr: Optional[float] = None

    @property
    def is_actionable(self) -> bool:
        return self.direction in {SIGNAL_UP, SIGNAL_DOWN}


class SignalCooldownTracker:
    """Track last Telegram push time per symbol and direction."""

    def __init__(self, cooldown_minutes: int) -> None:
        self.cooldown_minutes = cooldown_minutes
        self._last_push_ms: Dict[Tuple[str, str], int] = {}
        self._last_pushed_direction: Dict[str, str] = {}

    def is_blocked(self, symbol: str, direction: str, timestamp_ms: int) -> bool:
        if self.cooldown_minutes <= 0:
            return False
        if direction not in {SIGNAL_UP, SIGNAL_DOWN}:
            return False
        last_ms = self._last_push_ms.get((symbol, direction))
        if last_ms is None:
            return False
        cooldown_ms = self.cooldown_minutes * 60_000
        return timestamp_ms - last_ms < cooldown_ms

    def record_push(self, symbol: str, direction: str, timestamp_ms: int) -> None:
        if direction not in {SIGNAL_UP, SIGNAL_DOWN}:
            return
        self._last_push_ms[(symbol, direction)] = timestamp_ms
        self._last_pushed_direction[symbol] = direction

    def is_reversal(self, symbol: str, direction: str) -> bool:
        previous = self._last_pushed_direction.get(symbol)
        if previous is None:
            return False
        if direction not in {SIGNAL_UP, SIGNAL_DOWN}:
            return False
        return previous != direction


def _clamp01(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _latest_kline_volume(klines: Optional[pd.DataFrame], lookback: int) -> tuple[Optional[float], Optional[float]]:
    if klines is None or klines.empty:
        return None, None
    missing = [column for column in ("volume",) if column not in klines.columns]
    if missing:
        return None, None

    frame = klines.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    volumes = pd.to_numeric(frame["volume"], errors="coerce").dropna()
    if volumes.empty:
        return None, None

    current_volume = float(volumes.iloc[-1])
    window = volumes.tail(lookback)
    median_volume = float(window.median()) if not window.empty else None
    return current_volume, median_volume


def _extract_orderbook_metrics(
    orderbook: Optional[pd.DataFrame | OrderBookSnapshot | dict],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if orderbook is None:
        return None, None, None

    if isinstance(orderbook, OrderBookSnapshot):
        spread_bps = (
            orderbook.spread / orderbook.mid_price * 10_000.0
            if orderbook.mid_price > 0
            else None
        )
        return spread_bps, orderbook.book_imbalance, orderbook.mid_price

    if isinstance(orderbook, dict):
        mid = float(orderbook.get("mid_price", 0.0))
        spread = float(orderbook.get("spread", 0.0))
        spread_bps = spread / mid * 10_000.0 if mid > 0 else None
        return spread_bps, float(orderbook.get("book_imbalance", 0.0)), mid

    if isinstance(orderbook, pd.DataFrame):
        if orderbook.empty:
            return None, None, None
        missing = [column for column in ORDER_BOOK_COLUMNS if column not in orderbook.columns]
        if missing:
            return None, None, None
        row = orderbook.sort_values("timestamp").iloc[-1]
        mid = float(row["mid_price"])
        spread = float(row["spread"])
        spread_bps = spread / mid * 10_000.0 if mid > 0 else None
        return spread_bps, float(row["book_imbalance"]), mid

    return None, None, None


def _score_magnitude(
    predicted_return: float,
    *,
    direction: str,
    direction_threshold: float,
    magnitude_scale: float,
) -> float:
    if direction == SIGNAL_HOLD:
        return 0.0
    reference = max(direction_threshold, 1e-6)
    return _clamp01(abs(predicted_return) / (reference * magnitude_scale))


def _score_snr(
    predicted_return: float,
    residual_volatility: Optional[float],
    *,
    horizon_minutes: int,
    snr_scale: float,
) -> float:
    if residual_volatility is None or residual_volatility <= _EPSILON:
        return 0.0
    noise = residual_volatility * np.sqrt(max(horizon_minutes, 1))
    snr = abs(predicted_return) / max(noise, _EPSILON)
    return _clamp01(snr / snr_scale)


def _score_interval(
    predicted_return: float,
    interval_lower: Optional[float],
    interval_upper: Optional[float],
    *,
    direction: str,
) -> float:
    if direction == SIGNAL_HOLD:
        return 0.5

    lower = interval_lower if interval_lower is not None else predicted_return
    upper = interval_upper if interval_upper is not None else predicted_return

    if direction == SIGNAL_UP:
        if lower > 0 and upper > 0:
            return 1.0
        if predicted_return > 0:
            return 0.6
        return 0.0

    if direction == SIGNAL_DOWN:
        if lower < 0 and upper < 0:
            return 1.0
        if predicted_return < 0:
            return 0.6
        return 0.0

    return 0.5


def _score_volume(
    klines: Optional[pd.DataFrame],
    *,
    lookback: int,
    min_volume_ratio: float,
) -> tuple[float, bool]:
    current_volume, median_volume = _latest_kline_volume(klines, lookback)
    if current_volume is None or median_volume is None or median_volume <= _EPSILON:
        return 0.5, True

    ratio = current_volume / median_volume
    passed = ratio >= min_volume_ratio
    if ratio >= 1.0:
        return 1.0, passed
    if passed:
        return _clamp01(0.5 + 0.5 * (ratio - min_volume_ratio) / (1.0 - min_volume_ratio)), passed
    return _clamp01(ratio / min_volume_ratio * 0.5), passed


def _score_spread(spread_bps: Optional[float], *, max_spread_bps: float) -> tuple[float, bool]:
    if spread_bps is None:
        return 0.5, True
    if spread_bps > max_spread_bps:
        return 0.0, False
    return _clamp01(1.0 - 0.5 * spread_bps / max_spread_bps), True


def _is_combined_prediction(prediction: PredictionInput) -> bool:
    return isinstance(prediction, CombinedPredictionResult)


def _resolve_arima_direction(prediction: PredictionInput) -> str:
    if isinstance(prediction, CombinedPredictionResult):
        return prediction.arima_direction or prediction.direction or SIGNAL_HOLD
    return prediction.direction or SIGNAL_HOLD


def _resolve_aggregated_direction(prediction: PredictionInput) -> str:
    return prediction.direction or SIGNAL_HOLD


def _extract_aggregation_fields(
    prediction: PredictionInput,
) -> tuple[Optional[float], Optional[str], Optional[str], Optional[float], Tuple[str, ...]]:
    if not isinstance(prediction, CombinedPredictionResult):
        return None, None, None, None, ()
    return (
        prediction.garch_volatility,
        prediction.volatility_level,
        prediction.aggregation_mode,
        prediction.adjusted_snr,
        prediction.rejection_reasons,
    )


def _score_imbalance(book_imbalance: Optional[float], *, direction: str) -> float:
    if book_imbalance is None or direction == SIGNAL_HOLD:
        return 0.5
    normalized = float(np.clip(book_imbalance, -1.0, 1.0))
    if direction == SIGNAL_UP:
        return _clamp01((normalized + 1.0) / 2.0)
    if direction == SIGNAL_DOWN:
        return _clamp01((1.0 - normalized) / 2.0)
    return 0.5


def compute_confidence_components(
    prediction: PredictionInput,
    *,
    config: SignalEngineConfig,
    klines: Optional[pd.DataFrame] = None,
    orderbook: Optional[pd.DataFrame | OrderBookSnapshot | dict] = None,
    arima_direction: Optional[str] = None,
) -> tuple[ConfidenceComponents, bool, bool, Optional[float], Optional[float]]:
    """Compute factor scores and market filter pass flags."""
    direction = arima_direction or prediction.direction or SIGNAL_HOLD
    predicted_return = prediction.predicted_cumulative_return or 0.0

    magnitude = _score_magnitude(
        predicted_return,
        direction=direction,
        direction_threshold=config.direction_threshold,
        magnitude_scale=config.magnitude_scale,
    )
    snr = _score_snr(
        predicted_return,
        prediction.residual_volatility,
        horizon_minutes=config.prediction_minutes,
        snr_scale=config.snr_scale,
    )
    interval = _score_interval(
        predicted_return,
        prediction.interval_lower,
        prediction.interval_upper,
        direction=direction,
    )
    volume, volume_passed = _score_volume(
        klines,
        lookback=config.volume_lookback,
        min_volume_ratio=config.min_volume_ratio,
    )
    spread_bps, book_imbalance, _ = _extract_orderbook_metrics(orderbook)
    spread, spread_passed = _score_spread(spread_bps, max_spread_bps=config.max_spread_bps)
    imbalance = _score_imbalance(book_imbalance, direction=direction)

    components = ConfidenceComponents(
        magnitude=magnitude,
        snr=snr,
        interval=interval,
        volume=volume,
        spread=spread,
        imbalance=imbalance,
    )
    return components, volume_passed, spread_passed, spread_bps, book_imbalance


def build_trading_signal(
    prediction: PredictionInput,
    *,
    symbol: str,
    timestamp_ms: int,
    config: Optional[SignalEngineConfig] = None,
    klines: Optional[pd.DataFrame] = None,
    orderbook: Optional[pd.DataFrame | OrderBookSnapshot | dict] = None,
    cooldown: Optional[SignalCooldownTracker] = None,
) -> TradingSignal:
    """
    Map an ARIMA prediction to a trading signal with confidence and push gating.

    UP/DOWN are emitted only when confidence meets the threshold and market filters pass.
    Telegram push additionally requires cooldown clearance.
    """
    cfg = config or SignalEngineConfig()
    horizon_minutes = prediction.prediction_horizon_minutes or cfg.prediction_minutes
    expiry_ms = timestamp_ms + horizon_minutes * 60_000
    risk_note = "本工具仅提供预测提醒，不构成投资建议，请人工确认后再参与事件合约。"

    is_combined = _is_combined_prediction(prediction)
    garch_volatility, volatility_level, aggregation_mode, adjusted_snr, aggregation_reasons = (
        _extract_aggregation_fields(prediction)
    )

    if not prediction.success:
        empty = ConfidenceComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        message = prediction.error_message or "ARIMA prediction failed"
        trigger_parts = [message]
        if is_combined:
            trigger_parts.insert(0, "ARIMA-GARCH")
            if volatility_level:
                trigger_parts.append(f"volatility_level={volatility_level}")
            if adjusted_snr is not None:
                trigger_parts.append(f"adjusted_snr={adjusted_snr:.3f}")
            if aggregation_reasons:
                trigger_parts.append("blocked: " + "; ".join(aggregation_reasons))
        return TradingSignal(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            current_price=prediction.current_price,
            expiry_timestamp_ms=expiry_ms,
            direction=SIGNAL_HOLD,
            predicted_cumulative_return=None,
            confidence=0.0,
            confidence_threshold=cfg.confidence_threshold,
            should_push_telegram=False,
            is_direction_reversal=False,
            arima_model_order=prediction.model_order,
            spread_bps=None,
            book_imbalance=None,
            volume_filter_passed=False,
            spread_filter_passed=False,
            cooldown_blocked=False,
            trigger_summary="; ".join(trigger_parts),
            risk_note=risk_note,
            components=empty,
            arima_direction=None,
            rejection_reasons=tuple(aggregation_reasons) or (message,),
            garch_volatility=garch_volatility,
            volatility_level=volatility_level,
            aggregation_mode=aggregation_mode,
            adjusted_snr=adjusted_snr,
        )

    arima_direction = _resolve_arima_direction(prediction)
    components, volume_passed, spread_passed, spread_bps, book_imbalance = (
        compute_confidence_components(
            prediction,
            config=cfg,
            klines=klines,
            orderbook=orderbook,
            arima_direction=arima_direction,
        )
    )
    confidence = _clamp01(components.weighted_score(cfg.component_weights))

    rejection_reasons: list[str] = list(aggregation_reasons)
    if arima_direction == SIGNAL_HOLD and not is_combined:
        rejection_reasons.append("predicted move below direction threshold")
    if not volume_passed:
        rejection_reasons.append("volume below minimum relative to recent median")
    if not spread_passed:
        rejection_reasons.append("order book spread exceeds configured maximum")

    direction = _resolve_aggregated_direction(prediction) if is_combined else arima_direction
    if not is_combined and rejection_reasons and arima_direction in {SIGNAL_UP, SIGNAL_DOWN}:
        direction = SIGNAL_HOLD
    elif is_combined and direction in {SIGNAL_UP, SIGNAL_DOWN}:
        if not volume_passed or not spread_passed:
            direction = SIGNAL_HOLD

    meets_threshold = confidence >= cfg.confidence_threshold
    if not meets_threshold and direction in {SIGNAL_UP, SIGNAL_DOWN}:
        rejection_reasons.append(
            f"confidence {confidence:.3f} below threshold {cfg.confidence_threshold:.3f}"
        )
        direction = SIGNAL_HOLD

    cooldown_tracker = cooldown or SignalCooldownTracker(cfg.signal_cooldown_minutes)
    cooldown_blocked = cooldown_tracker.is_blocked(symbol, arima_direction, timestamp_ms)
    is_reversal = cooldown_tracker.is_reversal(symbol, arima_direction)

    should_push = (
        direction in {SIGNAL_UP, SIGNAL_DOWN}
        and meets_threshold
        and volume_passed
        and spread_passed
        and not cooldown_blocked
    )

    if cooldown_blocked and direction in {SIGNAL_UP, SIGNAL_DOWN}:
        rejection_reasons.append(
            f"cooldown active for {symbol} {arima_direction} ({cfg.signal_cooldown_minutes} min)"
        )

    source_label = "ARIMA-GARCH" if is_combined else "ARIMA"
    trigger_parts = [f"{source_label} {arima_direction}"]
    if is_combined:
        if volatility_level:
            trigger_parts.append(f"volatility_level={volatility_level}")
        if adjusted_snr is not None:
            trigger_parts.append(f"adjusted_snr={adjusted_snr:.3f}")
        if garch_volatility is not None:
            trigger_parts.append(f"garch_vol={garch_volatility:.6f}")
        if direction != arima_direction:
            trigger_parts.append(f"aggregation={direction}")
    trigger_parts.extend(
        [
            f"confidence={confidence:.3f}",
            f"magnitude={components.magnitude:.2f}",
            f"snr={components.snr:.2f}",
            f"interval={components.interval:.2f}",
        ]
    )
    if is_reversal and should_push:
        trigger_parts.append("direction reversal")
    if rejection_reasons:
        trigger_parts.append("blocked: " + "; ".join(rejection_reasons))

    return TradingSignal(
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        current_price=prediction.current_price,
        expiry_timestamp_ms=expiry_ms,
        direction=direction,
        predicted_cumulative_return=prediction.predicted_cumulative_return,
        confidence=confidence,
        confidence_threshold=cfg.confidence_threshold,
        should_push_telegram=should_push,
        is_direction_reversal=is_reversal and should_push,
        arima_model_order=prediction.model_order,
        spread_bps=spread_bps,
        book_imbalance=book_imbalance,
        volume_filter_passed=volume_passed,
        spread_filter_passed=spread_passed,
        cooldown_blocked=cooldown_blocked,
        trigger_summary="; ".join(trigger_parts),
        risk_note=risk_note,
        components=components,
        arima_direction=arima_direction,
        rejection_reasons=tuple(rejection_reasons),
        garch_volatility=garch_volatility,
        volatility_level=volatility_level,
        aggregation_mode=aggregation_mode,
        adjusted_snr=adjusted_snr,
    )


class SignalEngine:
    """Stateful signal generator with Telegram cooldown tracking."""

    def __init__(
        self,
        config: Optional[SignalEngineConfig] = None,
        *,
        cooldown: Optional[SignalCooldownTracker] = None,
    ) -> None:
        self.config = config or SignalEngineConfig()
        self.cooldown = cooldown or SignalCooldownTracker(self.config.signal_cooldown_minutes)

    @classmethod
    def from_settings(cls, settings) -> SignalEngine:
        config = SignalEngineConfig.from_settings(settings)
        return cls(config=config)

    def evaluate(
        self,
        prediction: PredictionInput,
        *,
        symbol: str,
        timestamp_ms: int,
        klines: Optional[pd.DataFrame] = None,
        orderbook: Optional[pd.DataFrame | OrderBookSnapshot | dict] = None,
    ) -> TradingSignal:
        signal = build_trading_signal(
            prediction,
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            config=self.config,
            klines=klines,
            orderbook=orderbook,
            cooldown=self.cooldown,
        )
        if signal.should_push_telegram:
            self.cooldown.record_push(symbol, signal.direction, timestamp_ms)
            logger.info(
                "Telegram push signal %s %s confidence=%.3f",
                symbol,
                signal.direction,
                signal.confidence,
            )
        else:
            logger.debug(
                "Signal held %s arima=%s confidence=%.3f reasons=%s",
                symbol,
                signal.arima_direction,
                signal.confidence,
                signal.rejection_reasons,
            )
        return signal


def validate_klines_for_signal(klines: pd.DataFrame) -> None:
    """Ensure klines contain columns required for volume filtering."""
    missing = [column for column in KLINE_COLUMNS if column not in klines.columns]
    if missing:
        raise ValueError(f"klines is missing required columns: {missing}")
