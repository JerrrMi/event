"""Rolling walk-forward backtest without look-ahead bias."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from src.features.engineering import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    FeatureConfig,
    compute_backtest_labels,
)
from src.models.arima_predictor import (
    ARIMAPredictorConfig,
    ARIMAPredictionResult,
    predict_from_klines,
)
from src.models.garch_predictor import GARCHPredictorConfig, predict_volatility_from_klines
from src.models.model_aggregator import (
    AggregatorConfig,
    CombinedPredictionResult,
    aggregate_predictions,
)
from src.signals.signal_engine import (
    SIGNAL_DOWN,
    SIGNAL_HOLD,
    SIGNAL_UP,
    SignalEngine,
    SignalEngineConfig,
    TradingSignal,
)

logger = logging.getLogger(__name__)

PredictionResult = Union[ARIMAPredictionResult, CombinedPredictionResult]
PredictFn = Callable[..., PredictionResult]

MODEL_SOURCE_ARIMA = "arima"
MODEL_SOURCE_ARIMA_GARCH = "arima_garch"


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters controlling the rolling backtest simulation."""

    symbol: str = "BTCUSDT"
    interval: str = "1m"
    train_window: int = 1440
    prediction_minutes: int = 10
    refit_interval_minutes: int = 5
    label_threshold: float = 0.0
    payout_ratio: float = 0.80
    arima_config: ARIMAPredictorConfig = field(default_factory=ARIMAPredictorConfig)
    garch_config: GARCHPredictorConfig = field(default_factory=GARCHPredictorConfig)
    aggregator_config: AggregatorConfig = field(default_factory=AggregatorConfig)
    signal_config: SignalEngineConfig = field(default_factory=SignalEngineConfig)
    use_garch: bool = False
    apply_cooldown: bool = True

    def __post_init__(self) -> None:
        if self.train_window < 1:
            raise ValueError(f"train_window must be >= 1, got {self.train_window}")
        if self.prediction_minutes < 1:
            raise ValueError(
                f"prediction_minutes must be >= 1, got {self.prediction_minutes}"
            )
        if self.refit_interval_minutes < 1:
            raise ValueError(
                f"refit_interval_minutes must be >= 1, got {self.refit_interval_minutes}"
            )
        if self.label_threshold < 0:
            raise ValueError(
                f"label_threshold must be non-negative, got {self.label_threshold}"
            )
        if self.payout_ratio <= 0:
            raise ValueError(f"payout_ratio must be positive, got {self.payout_ratio}")

    @classmethod
    def from_settings(cls, settings) -> BacktestConfig:
        return cls(
            symbol=settings.symbol,
            interval=settings.interval,
            train_window=settings.train_window,
            prediction_minutes=settings.prediction_minutes,
            refit_interval_minutes=settings.refit_interval_minutes,
            label_threshold=settings.direction_threshold,
            arima_config=ARIMAPredictorConfig.from_settings(settings),
            garch_config=GARCHPredictorConfig.from_settings(settings),
            aggregator_config=AggregatorConfig.from_settings(settings),
            signal_config=SignalEngineConfig.from_settings(settings),
            use_garch=getattr(settings, "use_garch", False),
            apply_cooldown=True,
        )


@dataclass
class BacktestRecord:
    """One simulated minute in the rolling backtest."""

    index: int
    timestamp_ms: int
    current_price: float
    arima_direction: Optional[str]
    signal_direction: str
    actual_direction: Optional[str]
    future_log_return: Optional[float]
    confidence: float
    is_actionable: bool
    is_correct: Optional[bool]
    arima_success: bool
    model_refit: bool
    predicted_cumulative_return: Optional[float]
    should_push_telegram: bool
    garch_success: bool = False
    garch_volatility: Optional[float] = None
    volatility_level: Optional[str] = None
    aggregation_direction: Optional[str] = None
    adjusted_snr: Optional[float] = None
    aggregation_rejection_reasons: Optional[str] = None
    model_source: str = MODEL_SOURCE_ARIMA


@dataclass
class DailyBacktestStats:
    """Aggregated metrics for a single UTC calendar day."""

    date: str
    total_minutes: int
    signal_count: int
    wins: int
    losses: int
    pushes: int
    win_rate: Optional[float]
    simplified_pnl: float


@dataclass
class BacktestSummary:
    """Aggregated backtest metrics."""

    symbol: str
    interval: str
    prediction_minutes: int
    train_window: int
    refit_interval_minutes: int
    total_minutes: int
    evaluable_minutes: int
    arima_success_count: int
    signal_count: int
    signal_frequency: float
    up_signal_count: int
    down_signal_count: int
    up_win_rate: Optional[float]
    down_win_rate: Optional[float]
    overall_win_rate: Optional[float]
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    max_consecutive_losses: int
    simplified_pnl: float
    simplified_return: float
    win_count: int
    loss_count: int
    push_count: int
    daily_stats: list[DailyBacktestStats]
    garch_success_count: int = 0
    aggregation_hold_count: int = 0
    extreme_vol_hold_count: int = 0
    average_garch_volatility: Optional[float] = None
    average_adjusted_snr: Optional[float] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["daily_stats"] = [asdict(item) for item in self.daily_stats]
        return payload


def _prepare_klines(klines: pd.DataFrame) -> pd.DataFrame:
    from src.models.arima_predictor import _prepare_klines as prepare

    return prepare(klines)


def make_aggregated_predict_fn(
    *,
    garch_config: GARCHPredictorConfig,
    aggregator_config: AggregatorConfig,
) -> PredictFn:
    """Build a predict_fn that runs ARIMA, GARCH, and aggregation on past klines only."""

    def predict_fn(
        klines: pd.DataFrame,
        *,
        train_window: int,
        config: ARIMAPredictorConfig,
    ) -> CombinedPredictionResult:
        arima = predict_from_klines(klines, train_window=train_window, config=config)
        garch = predict_volatility_from_klines(
            klines,
            train_window=train_window,
            config=garch_config,
        )
        return aggregate_predictions(arima, garch, config=aggregator_config)

    return predict_fn


def _resolve_predict_fn(
    *,
    config: BacktestConfig,
    predict_fn: Optional[PredictFn],
) -> PredictFn:
    if predict_fn is not None:
        return predict_fn
    if config.use_garch:
        return make_aggregated_predict_fn(
            garch_config=config.garch_config,
            aggregator_config=config.aggregator_config,
        )
    return predict_from_klines


def _format_rejection_reasons(reasons: tuple[str, ...] | list[str] | None) -> Optional[str]:
    if not reasons:
        return None
    return "; ".join(reasons)


def _extract_backtest_prediction_fields(
    prediction: PredictionResult,
) -> dict:
    """Map ARIMA or combined prediction into BacktestRecord diagnostic fields."""
    if isinstance(prediction, CombinedPredictionResult):
        garch_success = prediction.garch_volatility is not None
        return {
            "model_source": MODEL_SOURCE_ARIMA_GARCH,
            "arima_direction": prediction.arima_direction if prediction.success else None,
            "aggregation_direction": prediction.direction,
            "garch_success": garch_success,
            "garch_volatility": prediction.garch_volatility,
            "volatility_level": prediction.volatility_level,
            "adjusted_snr": prediction.adjusted_snr,
            "aggregation_rejection_reasons": _format_rejection_reasons(
                prediction.rejection_reasons
            ),
        }

    return {
        "model_source": MODEL_SOURCE_ARIMA,
        "arima_direction": prediction.direction if prediction.success else None,
        "aggregation_direction": prediction.direction if prediction.success else None,
        "garch_success": False,
        "garch_volatility": None,
        "volatility_level": None,
        "adjusted_snr": None,
        "aggregation_rejection_reasons": None,
    }


def _truncate_orderbook(
    orderbook: Optional[pd.DataFrame],
    *,
    cutoff_ms: int,
) -> Optional[pd.DataFrame]:
    if orderbook is None or orderbook.empty:
        return orderbook
    frame = orderbook.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame[frame["timestamp"] <= cutoff_ms].reset_index(drop=True)


def _evaluate_signal_outcome(
    signal_direction: str,
    actual_direction: Optional[str],
) -> tuple[Optional[bool], str]:
    """Return (is_correct, outcome) where outcome is win/loss/push/none."""
    if signal_direction not in {SIGNAL_UP, SIGNAL_DOWN}:
        return None, "none"
    if actual_direction is None or pd.isna(actual_direction):
        return None, "none"
    if actual_direction == LABEL_FLAT:
        return None, "push"
    if signal_direction == actual_direction:
        return True, "win"
    return False, "loss"


def _max_consecutive_losses(outcomes: list[str]) -> int:
    max_streak = 0
    current = 0
    for outcome in outcomes:
        if outcome == "loss":
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _balanced_accuracy(
    predicted: list[str],
    actual: list[str],
) -> Optional[float]:
    classes = [SIGNAL_UP, SIGNAL_DOWN]
    recalls: list[float] = []
    for label in classes:
        mask = [item == label for item in actual]
        if not any(mask):
            continue
        correct = sum(
            1 for pred, act, include in zip(predicted, actual, mask) if include and pred == act
        )
        recalls.append(correct / sum(mask))
    if not recalls:
        return None
    return float(np.mean(recalls))


def compute_backtest_summary(
    records: list[BacktestRecord],
    *,
    config: BacktestConfig,
) -> BacktestSummary:
    """Aggregate per-minute records into summary metrics."""
    evaluable = [record for record in records if record.actual_direction is not None]
    actionable = [record for record in evaluable if record.is_actionable]

    outcomes: list[str] = []
    for record in actionable:
        _, outcome = _evaluate_signal_outcome(record.signal_direction, record.actual_direction)
        outcomes.append(outcome)

    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    pushes = outcomes.count("push")
    decided = wins + losses

    up_signals = [record for record in actionable if record.signal_direction == SIGNAL_UP]
    down_signals = [record for record in actionable if record.signal_direction == SIGNAL_DOWN]

    def _win_rate(subset: list[BacktestRecord]) -> Optional[float]:
        local_outcomes = [
            _evaluate_signal_outcome(item.signal_direction, item.actual_direction)[1]
            for item in subset
        ]
        local_wins = local_outcomes.count("win")
        local_losses = local_outcomes.count("loss")
        if local_wins + local_losses == 0:
            return None
        return local_wins / (local_wins + local_losses)

    directional = [
        record
        for record in evaluable
        if record.arima_direction in {SIGNAL_UP, SIGNAL_DOWN}
        and record.actual_direction in {LABEL_UP, LABEL_DOWN}
    ]
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    if directional:
        predicted = [record.arima_direction for record in directional]
        actual = [record.actual_direction for record in directional]
        accuracy = sum(pred == act for pred, act in zip(predicted, actual)) / len(directional)
        balanced_accuracy = _balanced_accuracy(predicted, actual)

    simplified_pnl = wins * config.payout_ratio - losses * 1.0
    signal_count = len(actionable)
    evaluable_minutes = len(evaluable)

    daily_map: dict[str, DailyBacktestStats] = {}
    for record in actionable:
        day = datetime.fromtimestamp(record.timestamp_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )
        if day not in daily_map:
            daily_map[day] = DailyBacktestStats(
                date=day,
                total_minutes=0,
                signal_count=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=None,
                simplified_pnl=0.0,
            )
        _, outcome = _evaluate_signal_outcome(record.signal_direction, record.actual_direction)
        daily_map[day].signal_count += 1
        if outcome == "win":
            daily_map[day].wins += 1
            daily_map[day].simplified_pnl += config.payout_ratio
        elif outcome == "loss":
            daily_map[day].losses += 1
            daily_map[day].simplified_pnl -= 1.0
        elif outcome == "push":
            daily_map[day].pushes += 1

    for record in evaluable:
        day = datetime.fromtimestamp(record.timestamp_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )
        if day not in daily_map:
            daily_map[day] = DailyBacktestStats(
                date=day,
                total_minutes=0,
                signal_count=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=None,
                simplified_pnl=0.0,
            )
        daily_map[day].total_minutes += 1

    daily_stats: list[DailyBacktestStats] = []
    for day in sorted(daily_map):
        item = daily_map[day]
        decided_daily = item.wins + item.losses
        item.win_rate = item.wins / decided_daily if decided_daily else None
        daily_stats.append(item)

    garch_volatilities = [
        record.garch_volatility for record in records if record.garch_volatility is not None
    ]
    adjusted_snrs = [record.adjusted_snr for record in records if record.adjusted_snr is not None]

    return BacktestSummary(
        symbol=config.symbol,
        interval=config.interval,
        prediction_minutes=config.prediction_minutes,
        train_window=config.train_window,
        refit_interval_minutes=config.refit_interval_minutes,
        total_minutes=len(records),
        evaluable_minutes=evaluable_minutes,
        arima_success_count=sum(1 for record in records if record.arima_success),
        signal_count=signal_count,
        signal_frequency=(signal_count / evaluable_minutes if evaluable_minutes else 0.0),
        up_signal_count=len(up_signals),
        down_signal_count=len(down_signals),
        up_win_rate=_win_rate(up_signals),
        down_win_rate=_win_rate(down_signals),
        overall_win_rate=(wins / decided if decided else None),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        max_consecutive_losses=_max_consecutive_losses(outcomes),
        simplified_pnl=simplified_pnl,
        simplified_return=(simplified_pnl / signal_count if signal_count else 0.0),
        win_count=wins,
        loss_count=losses,
        push_count=pushes,
        daily_stats=daily_stats,
        garch_success_count=sum(1 for record in records if record.garch_success),
        aggregation_hold_count=sum(
            1
            for record in records
            if record.aggregation_direction == SIGNAL_HOLD
        ),
        extreme_vol_hold_count=sum(
            1
            for record in records
            if record.aggregation_direction == SIGNAL_HOLD
            and record.aggregation_rejection_reasons
            and "extreme_volatility" in record.aggregation_rejection_reasons
        ),
        average_garch_volatility=(
            float(np.mean(garch_volatilities)) if garch_volatilities else None
        ),
        average_adjusted_snr=(float(np.mean(adjusted_snrs)) if adjusted_snrs else None),
    )


def records_to_dataframe(records: list[BacktestRecord]) -> pd.DataFrame:
    """Convert backtest records to a flat DataFrame for export."""
    return pd.DataFrame([asdict(record) for record in records])


def save_backtest_results(
    *,
    records: list[BacktestRecord],
    summary: BacktestSummary,
    output_dir: Path,
    symbol: str,
) -> tuple[Path, Path]:
    """Persist detailed records and summary JSON under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"{symbol}_backtest_{stamp}.csv"
    summary_path = output_dir / f"{symbol}_backtest_{stamp}_summary.json"

    records_to_dataframe(records).to_csv(detail_path, index=False)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return detail_path, summary_path


def run_rolling_backtest(
    klines: pd.DataFrame,
    *,
    config: Optional[BacktestConfig] = None,
    orderbook: Optional[pd.DataFrame] = None,
    predict_fn: Optional[PredictFn] = None,
) -> tuple[list[BacktestRecord], BacktestSummary]:
    """
    Walk forward minute-by-minute using only past bars for model training.

    When ``use_garch`` is enabled (or an aggregated ``predict_fn`` is supplied),
    ARIMA and GARCH share the same truncated history window before aggregation.

    Labels are computed from future prices for evaluation only and are never
    passed into model training or signal inputs.
    """
    cfg = config or BacktestConfig()
    resolved_predict_fn = _resolve_predict_fn(config=cfg, predict_fn=predict_fn)
    frame = _prepare_klines(klines)
    if frame.empty:
        summary = compute_backtest_summary([], config=cfg)
        return [], summary

    feature_cfg = FeatureConfig(
        interval=cfg.interval,
        prediction_minutes=cfg.prediction_minutes,
        label_threshold=cfg.label_threshold,
    )
    labels = compute_backtest_labels(
        frame,
        interval=cfg.interval,
        prediction_minutes=cfg.prediction_minutes,
        label_threshold=cfg.label_threshold,
    )

    min_history = max(
        cfg.train_window,
        cfg.signal_config.volume_lookback + 1,
        cfg.garch_config.min_train_points if cfg.use_garch else 0,
    )
    last_index = len(frame) - cfg.prediction_minutes - 1
    start_index = min_history - 1

    if last_index < start_index:
        logger.warning(
            "Not enough klines for rolling backtest: need at least %s bars, got %s",
            min_history + cfg.prediction_minutes,
            len(frame),
        )
        summary = compute_backtest_summary([], config=cfg)
        return [], summary

    signal_engine = SignalEngine(config=cfg.signal_config)
    if not cfg.apply_cooldown:
        from src.signals.signal_engine import SignalCooldownTracker

        signal_engine.cooldown = SignalCooldownTracker(cooldown_minutes=0)

    records: list[BacktestRecord] = []
    cached_prediction: Optional[PredictionResult] = None
    steps_since_refit = cfg.refit_interval_minutes

    for index in range(start_index, last_index + 1):
        timestamp_ms = int(frame.iloc[index]["timestamp"])
        current_price = float(frame.iloc[index]["close"])
        history = frame.iloc[: index + 1]
        book_history = _truncate_orderbook(orderbook, cutoff_ms=timestamp_ms)

        should_refit = cached_prediction is None or steps_since_refit >= cfg.refit_interval_minutes
        if should_refit:
            cached_prediction = resolved_predict_fn(
                history,
                train_window=cfg.train_window,
                config=cfg.arima_config,
            )
            steps_since_refit = 0
        else:
            steps_since_refit += 1

        prediction = cached_prediction
        assert prediction is not None

        prediction_fields = _extract_backtest_prediction_fields(prediction)

        signal: TradingSignal = signal_engine.evaluate(
            prediction,
            symbol=cfg.symbol,
            timestamp_ms=timestamp_ms,
            klines=history,
            orderbook=book_history,
        )

        actual_direction = labels.iloc[index]["label_direction"]
        if pd.isna(actual_direction):
            actual_direction = None
        else:
            actual_direction = str(actual_direction)

        future_log_return = labels.iloc[index]["future_log_return"]
        if pd.isna(future_log_return):
            future_log_return = None
        else:
            future_log_return = float(future_log_return)

        is_correct, _ = _evaluate_signal_outcome(signal.direction, actual_direction)

        records.append(
            BacktestRecord(
                index=index,
                timestamp_ms=timestamp_ms,
                current_price=current_price,
                arima_direction=prediction_fields["arima_direction"],
                signal_direction=signal.direction,
                actual_direction=actual_direction,
                future_log_return=future_log_return,
                confidence=signal.confidence,
                is_actionable=signal.is_actionable,
                is_correct=is_correct,
                arima_success=prediction.success,
                model_refit=should_refit,
                predicted_cumulative_return=prediction.predicted_cumulative_return,
                should_push_telegram=signal.should_push_telegram,
                garch_success=prediction_fields["garch_success"],
                garch_volatility=prediction_fields["garch_volatility"],
                volatility_level=prediction_fields["volatility_level"],
                aggregation_direction=prediction_fields["aggregation_direction"],
                adjusted_snr=prediction_fields["adjusted_snr"],
                aggregation_rejection_reasons=prediction_fields["aggregation_rejection_reasons"],
                model_source=prediction_fields["model_source"],
            )
        )

    summary = compute_backtest_summary(records, config=cfg)
    return records, summary


def format_summary_report(summary: BacktestSummary) -> str:
    """Render a human-readable backtest report."""
    lines = [
        f"Symbol: {summary.symbol}",
        f"Interval: {summary.interval}",
        f"Prediction horizon: {summary.prediction_minutes} minutes",
        f"Train window: {summary.train_window}",
        f"Refit interval: {summary.refit_interval_minutes} minutes",
        "",
        f"Total simulated minutes: {summary.total_minutes}",
        f"Evaluable minutes: {summary.evaluable_minutes}",
        f"ARIMA success count: {summary.arima_success_count}",
        f"GARCH success count: {summary.garch_success_count}",
        f"Aggregation hold count: {summary.aggregation_hold_count}",
        f"Extreme volatility hold count: {summary.extreme_vol_hold_count}",
        f"Average GARCH volatility: {_fmt_float(summary.average_garch_volatility)}",
        f"Average adjusted SNR: {_fmt_float(summary.average_adjusted_snr)}",
        "",
        f"Signal count: {summary.signal_count}",
        f"Signal frequency: {summary.signal_frequency:.4f}",
        f"UP signals: {summary.up_signal_count}",
        f"DOWN signals: {summary.down_signal_count}",
        "",
        f"UP win rate: {_fmt_rate(summary.up_win_rate)}",
        f"DOWN win rate: {_fmt_rate(summary.down_win_rate)}",
        f"Overall win rate: {_fmt_rate(summary.overall_win_rate)}",
        f"Accuracy (directional ARIMA): {_fmt_rate(summary.accuracy)}",
        f"Balanced accuracy: {_fmt_rate(summary.balanced_accuracy)}",
        "",
        f"Wins / losses / pushes: {summary.win_count} / {summary.loss_count} / {summary.push_count}",
        f"Max consecutive losses: {summary.max_consecutive_losses}",
        f"Simplified PnL: {summary.simplified_pnl:.4f}",
        f"Simplified return per signal: {summary.simplified_return:.4f}",
        "",
        "Daily breakdown:",
    ]

    if not summary.daily_stats:
        lines.append("  (no daily stats)")
    else:
        for day in summary.daily_stats:
            lines.append(
                f"  {day.date}: minutes={day.total_minutes}, signals={day.signal_count}, "
                f"win_rate={_fmt_rate(day.win_rate)}, pnl={day.simplified_pnl:.4f}"
            )

    return "\n".join(lines)


def _fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"
