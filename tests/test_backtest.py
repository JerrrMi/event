"""Tests for the rolling backtest module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pandas as pd

from src.backtest.rolling_backtest import (
    BacktestConfig,
    BacktestRecord,
    MODEL_SOURCE_ARIMA,
    MODEL_SOURCE_ARIMA_GARCH,
    compute_backtest_summary,
    make_aggregated_predict_fn,
    records_to_dataframe,
    run_rolling_backtest,
    save_backtest_results,
)
from src.features.engineering import LABEL_DOWN, LABEL_FLAT, LABEL_UP
from src.models.arima_predictor import (
    ARIMAPredictorConfig,
    ARIMAPredictionResult,
    DIRECTION_DOWN,
    DIRECTION_HOLD,
    DIRECTION_UP,
)
from src.models.garch_predictor import GARCHPredictionResult, GARCHPredictorConfig, VolatilityLevel
from src.models.model_aggregator import AggregatorConfig, CombinedPredictionResult
from src.signals.signal_engine import SIGNAL_DOWN, SIGNAL_HOLD, SIGNAL_UP, SignalEngineConfig


def _make_klines(
    closes: list[float],
    *,
    start_ts: int = 1_700_000_000_000,
    step_ms: int = 60_000,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    if volumes is None:
        volumes = [100.0] * len(closes)

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


def _successful_prediction(*, direction: str = DIRECTION_UP) -> ARIMAPredictionResult:
    return ARIMAPredictionResult(
        success=True,
        predicted_cumulative_return=0.002 if direction == DIRECTION_UP else -0.002,
        direction=direction,
        interval_lower=-0.001 if direction == DIRECTION_DOWN else 0.0005,
        interval_upper=0.003 if direction == DIRECTION_UP else 0.001,
        residual_volatility=0.0002,
        model_order=(1, 0, 1),
        series_type="log_return",
        current_price=100.0,
        prediction_horizon_minutes=10,
        forecast_steps=10,
        train_points=100,
    )


def _combined_prediction(
    *,
    arima_direction: str = DIRECTION_UP,
    aggregation_direction: str = DIRECTION_UP,
    garch_volatility: float = 0.0015,
    volatility_level: str = VolatilityLevel.NORMAL.value,
    adjusted_snr: float = 1.2,
    rejection_reasons: tuple[str, ...] = (),
) -> CombinedPredictionResult:
    predicted_return = 0.002 if arima_direction == DIRECTION_UP else -0.002
    return CombinedPredictionResult(
        success=True,
        predicted_cumulative_return=predicted_return,
        direction=aggregation_direction,
        interval_lower=-0.001,
        interval_upper=0.003,
        residual_volatility=0.0002,
        model_order=(1, 0, 1),
        series_type="log_return",
        current_price=100.0,
        prediction_horizon_minutes=10,
        forecast_steps=10,
        train_points=100,
        arima_direction=arima_direction,
        garch_volatility=garch_volatility,
        volatility_level=volatility_level,
        aggregation_mode="volatility_adjusted_arima",
        adjusted_snr=adjusted_snr,
        rejection_reasons=rejection_reasons,
    )


def test_compute_backtest_summary_metrics() -> None:
    records = [
        BacktestRecord(
            index=0,
            timestamp_ms=1_700_000_000_000,
            current_price=100.0,
            arima_direction=SIGNAL_UP,
            signal_direction=SIGNAL_UP,
            actual_direction=LABEL_UP,
            future_log_return=0.001,
            confidence=0.9,
            is_actionable=True,
            is_correct=True,
            arima_success=True,
            model_refit=True,
            predicted_cumulative_return=0.002,
            should_push_telegram=True,
        ),
        BacktestRecord(
            index=1,
            timestamp_ms=1_700_000_060_000,
            current_price=101.0,
            arima_direction=SIGNAL_DOWN,
            signal_direction=SIGNAL_DOWN,
            actual_direction=LABEL_UP,
            future_log_return=0.001,
            confidence=0.9,
            is_actionable=True,
            is_correct=False,
            arima_success=True,
            model_refit=False,
            predicted_cumulative_return=-0.002,
            should_push_telegram=True,
        ),
        BacktestRecord(
            index=2,
            timestamp_ms=1_700_000_120_000,
            current_price=102.0,
            arima_direction=SIGNAL_UP,
            signal_direction=SIGNAL_HOLD,
            actual_direction=LABEL_FLAT,
            future_log_return=0.0,
            confidence=0.4,
            is_actionable=False,
            is_correct=None,
            arima_success=True,
            model_refit=False,
            predicted_cumulative_return=0.0001,
            should_push_telegram=False,
        ),
    ]

    summary = compute_backtest_summary(records, config=BacktestConfig(payout_ratio=0.8))

    assert summary.signal_count == 2
    assert summary.evaluable_minutes == 3
    assert summary.signal_frequency == pytest.approx(2 / 3)
    assert summary.overall_win_rate == pytest.approx(0.5)
    assert summary.win_count == 1
    assert summary.loss_count == 1
    assert summary.max_consecutive_losses == 1
    assert summary.simplified_pnl == pytest.approx(0.8 - 1.0)
    assert summary.balanced_accuracy == pytest.approx(0.5)


def test_compute_backtest_summary_garch_metrics() -> None:
    records = [
        BacktestRecord(
            index=0,
            timestamp_ms=1_700_000_000_000,
            current_price=100.0,
            arima_direction=SIGNAL_UP,
            signal_direction=SIGNAL_HOLD,
            actual_direction=LABEL_UP,
            future_log_return=0.001,
            confidence=0.4,
            is_actionable=False,
            is_correct=None,
            arima_success=True,
            model_refit=True,
            predicted_cumulative_return=0.002,
            should_push_telegram=False,
            garch_success=True,
            garch_volatility=0.001,
            volatility_level=VolatilityLevel.EXTREME.value,
            aggregation_direction=SIGNAL_HOLD,
            adjusted_snr=0.5,
            aggregation_rejection_reasons="extreme_volatility",
            model_source=MODEL_SOURCE_ARIMA_GARCH,
        ),
        BacktestRecord(
            index=1,
            timestamp_ms=1_700_000_060_000,
            current_price=101.0,
            arima_direction=SIGNAL_UP,
            signal_direction=SIGNAL_UP,
            actual_direction=LABEL_UP,
            future_log_return=0.001,
            confidence=0.9,
            is_actionable=True,
            is_correct=True,
            arima_success=True,
            model_refit=False,
            predicted_cumulative_return=0.002,
            should_push_telegram=True,
            garch_success=True,
            garch_volatility=0.003,
            volatility_level=VolatilityLevel.NORMAL.value,
            aggregation_direction=SIGNAL_UP,
            adjusted_snr=1.5,
            aggregation_rejection_reasons=None,
            model_source=MODEL_SOURCE_ARIMA_GARCH,
        ),
    ]

    summary = compute_backtest_summary(records, config=BacktestConfig(payout_ratio=0.8))

    assert summary.garch_success_count == 2
    assert summary.aggregation_hold_count == 1
    assert summary.extreme_vol_hold_count == 1
    assert summary.average_garch_volatility == pytest.approx(0.002)
    assert summary.average_adjusted_snr == pytest.approx(1.0)


def test_records_to_dataframe_includes_garch_fields() -> None:
    record = BacktestRecord(
        index=0,
        timestamp_ms=1_700_000_000_000,
        current_price=100.0,
        arima_direction=SIGNAL_UP,
        signal_direction=SIGNAL_UP,
        actual_direction=LABEL_UP,
        future_log_return=0.001,
        confidence=0.9,
        is_actionable=True,
        is_correct=True,
        arima_success=True,
        model_refit=True,
        predicted_cumulative_return=0.002,
        should_push_telegram=True,
        garch_success=True,
        garch_volatility=0.0015,
        volatility_level=VolatilityLevel.NORMAL.value,
        aggregation_direction=SIGNAL_UP,
        adjusted_snr=1.2,
        aggregation_rejection_reasons="low_adjusted_snr",
        model_source=MODEL_SOURCE_ARIMA_GARCH,
    )

    frame = records_to_dataframe([record])

    assert "garch_success" in frame.columns
    assert "garch_volatility" in frame.columns
    assert "volatility_level" in frame.columns
    assert "aggregation_direction" in frame.columns
    assert "adjusted_snr" in frame.columns
    assert "aggregation_rejection_reasons" in frame.columns
    assert "model_source" in frame.columns
    assert frame.iloc[0]["model_source"] == MODEL_SOURCE_ARIMA_GARCH


def test_save_backtest_results_writes_garch_summary_fields(tmp_path) -> None:
    records = [
        BacktestRecord(
            index=0,
            timestamp_ms=1_700_000_000_000,
            current_price=100.0,
            arima_direction=SIGNAL_UP,
            signal_direction=SIGNAL_UP,
            actual_direction=LABEL_UP,
            future_log_return=0.001,
            confidence=0.9,
            is_actionable=True,
            is_correct=True,
            arima_success=True,
            model_refit=True,
            predicted_cumulative_return=0.002,
            should_push_telegram=True,
            garch_success=True,
            garch_volatility=0.0015,
            volatility_level=VolatilityLevel.NORMAL.value,
            aggregation_direction=SIGNAL_UP,
            adjusted_snr=1.2,
            model_source=MODEL_SOURCE_ARIMA_GARCH,
        )
    ]
    summary = compute_backtest_summary(records, config=BacktestConfig())

    detail_path, summary_path = save_backtest_results(
        records=records,
        summary=summary,
        output_dir=tmp_path,
        symbol="BTCUSDT",
    )

    detail = pd.read_csv(detail_path)
    assert "garch_success" in detail.columns
    assert detail.iloc[0]["model_source"] == MODEL_SOURCE_ARIMA_GARCH

    payload = summary_path.read_text(encoding="utf-8")
    assert "garch_success_count" in payload
    assert "average_adjusted_snr" in payload


def test_run_rolling_backtest_with_aggregated_predict_fn() -> None:
    klines = _make_klines([100.0 + index * 0.1 for index in range(80)])
    config = BacktestConfig(
        train_window=30,
        prediction_minutes=5,
        refit_interval_minutes=1,
        use_garch=True,
        arima_config=ARIMAPredictorConfig(
            prediction_minutes=5,
            order=(1, 0, 0),
            min_train_points=10,
        ),
        garch_config=GARCHPredictorConfig(prediction_minutes=5, min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    def aggregated_predict(history, *, train_window, config):
        return _combined_prediction(
            arima_direction=DIRECTION_UP,
            aggregation_direction=DIRECTION_UP,
            garch_volatility=0.0015,
            adjusted_snr=1.2,
        )

    records, summary = run_rolling_backtest(
        klines,
        config=config,
        predict_fn=aggregated_predict,
    )

    assert records
    assert all(record.model_source == MODEL_SOURCE_ARIMA_GARCH for record in records)
    assert all(record.garch_success for record in records)
    assert all(record.aggregation_direction == SIGNAL_UP for record in records)
    assert summary.garch_success_count == len(records)
    assert summary.average_garch_volatility == pytest.approx(0.0015)


def test_run_rolling_backtest_arima_only_sets_model_source() -> None:
    klines = _make_klines([100.0 + index * 0.05 for index in range(70)])
    config = BacktestConfig(
        train_window=25,
        prediction_minutes=5,
        refit_interval_minutes=1,
        use_garch=False,
        arima_config=ARIMAPredictorConfig(prediction_minutes=5, order=(1, 0, 0), min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    records, summary = run_rolling_backtest(
        klines,
        config=config,
        predict_fn=lambda history, *, train_window, config: _successful_prediction(
            direction=DIRECTION_UP
        ),
    )

    assert records
    assert all(record.model_source == MODEL_SOURCE_ARIMA for record in records)
    assert all(not record.garch_success for record in records)
    assert summary.garch_success_count == 0


def test_make_aggregated_predict_fn_runs_without_future_data() -> None:
    klines = _make_klines([100.0 + index * 0.02 for index in range(120)])
    config = BacktestConfig(
        train_window=40,
        prediction_minutes=5,
        refit_interval_minutes=1,
        use_garch=True,
        arima_config=ARIMAPredictorConfig(
            prediction_minutes=5,
            order=(1, 0, 0),
            min_train_points=10,
        ),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.99,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    with (
        patch("src.backtest.rolling_backtest.predict_from_klines") as mock_arima,
        patch("src.backtest.rolling_backtest.predict_volatility_from_klines") as mock_garch,
        patch("src.backtest.rolling_backtest.aggregate_predictions") as mock_aggregate,
    ):
        mock_arima.return_value = _successful_prediction(direction=DIRECTION_UP)
        mock_garch.return_value = GARCHPredictionResult(
            success=True,
            cumulative_volatility=0.0015,
            volatility_level=VolatilityLevel.NORMAL.value,
            model_order=(1, 1),
            train_points=40,
            current_price=100.0,
            prediction_horizon_minutes=5,
        )
        mock_aggregate.return_value = _combined_prediction()

        predict_fn = make_aggregated_predict_fn(
            garch_config=config.garch_config,
            aggregator_config=AggregatorConfig(prediction_minutes=5, aggregation_min_snr=0.1),
        )

        records, summary = run_rolling_backtest(klines, config=config, predict_fn=predict_fn)

    assert records
    assert mock_arima.called
    assert mock_garch.called
    assert mock_aggregate.called
    for call in mock_arima.call_args_list:
        history = call.args[0]
        assert len(history) <= len(klines)
        last_ts = history.iloc[-1]["timestamp"]
        assert all(history["timestamp"] <= last_ts)
    assert summary.garch_success_count == len(records)


def test_run_rolling_backtest_uses_only_past_data() -> None:
    klines = _make_klines([100.0 + index * 0.1 for index in range(80)])
    config = BacktestConfig(
        train_window=30,
        prediction_minutes=5,
        refit_interval_minutes=1,
        arima_config=ARIMAPredictorConfig(
            prediction_minutes=5,
            order=(1, 0, 0),
            min_train_points=10,
        ),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    seen_lengths: list[int] = []

    def tracking_predict(history, *, train_window, config):
        seen_lengths.append(len(history))
        return _successful_prediction(direction=DIRECTION_UP)

    records, summary = run_rolling_backtest(
        klines,
        config=config,
        predict_fn=tracking_predict,
    )

    assert records
    assert seen_lengths
    assert max(seen_lengths) <= len(klines)
    for length in seen_lengths:
        assert length <= len(klines)
    assert summary.total_minutes == len(records)


def test_run_rolling_backtest_reuses_prediction_between_refits() -> None:
    klines = _make_klines([100.0 + index * 0.05 for index in range(70)])
    config = BacktestConfig(
        train_window=25,
        prediction_minutes=5,
        refit_interval_minutes=5,
        arima_config=ARIMAPredictorConfig(prediction_minutes=5, order=(1, 0, 0), min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    call_count = 0

    def counting_predict(history, *, train_window, config):
        nonlocal call_count
        call_count += 1
        return _successful_prediction(direction=DIRECTION_UP)

    records, _ = run_rolling_backtest(
        klines,
        config=config,
        predict_fn=counting_predict,
    )

    refit_count = sum(1 for record in records if record.model_refit)
    assert call_count == refit_count
    assert call_count < len(records)


def test_run_rolling_backtest_evaluates_future_direction() -> None:
    closes = [100.0] * 40
    for index in range(40, 80):
        closes.append(100.0 + (index - 40) * 0.5)

    klines = _make_klines(closes)
    config = BacktestConfig(
        train_window=25,
        prediction_minutes=5,
        refit_interval_minutes=1,
        arima_config=ARIMAPredictorConfig(prediction_minutes=5, order=(1, 0, 0), min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    records, summary = run_rolling_backtest(
        klines,
        config=config,
        predict_fn=lambda history, *, train_window, config: _successful_prediction(
            direction=DIRECTION_UP
        ),
    )

    actionable = [record for record in records if record.is_actionable]
    assert actionable
    assert any(record.actual_direction == LABEL_UP for record in actionable)
    assert summary.overall_win_rate is not None


@patch("src.backtest.rolling_backtest.predict_from_klines")
def test_integration_with_real_arima_predictor(mock_predict: MagicMock) -> None:
    mock_predict.side_effect = lambda history, *, train_window, config: _successful_prediction()

    klines = _make_klines([100.0 + index * 0.02 for index in range(60)])
    config = BacktestConfig(
        train_window=30,
        prediction_minutes=5,
        refit_interval_minutes=10,
        arima_config=ARIMAPredictorConfig(prediction_minutes=5, order=(1, 0, 1), min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.99,
            prediction_minutes=5,
        ),
        apply_cooldown=False,
    )

    records, summary = run_rolling_backtest(klines, config=config, predict_fn=mock_predict)

    assert mock_predict.called
    for call in mock_predict.call_args_list:
        history = call.args[0]
        assert len(history) <= len(klines)
        last_ts = history.iloc[-1]["timestamp"]
        assert all(history["timestamp"] <= last_ts)


def test_run_rolling_backtest_insufficient_data_returns_empty() -> None:
    klines = _make_klines([100.0] * 10)
    config = BacktestConfig(train_window=30, prediction_minutes=5)

    records, summary = run_rolling_backtest(klines, config=config)

    assert records == []
    assert summary.total_minutes == 0


def test_labels_not_in_training_input() -> None:
    """Ensure label columns are never passed into the ARIMA training slice."""
    klines = _make_klines([100.0 + index * 0.03 for index in range(65)])
    config = BacktestConfig(
        train_window=25,
        prediction_minutes=5,
        refit_interval_minutes=1,
        arima_config=ARIMAPredictorConfig(prediction_minutes=5, order=(1, 0, 0), min_train_points=10),
        signal_config=SignalEngineConfig(
            confidence_threshold=0.01,
            prediction_minutes=5,
            min_volume_ratio=0.01,
            max_spread_bps=10_000.0,
        ),
        apply_cooldown=False,
    )

    forbidden = {"future_close", "future_log_return", "label_direction"}

    def inspect_predict(history, *, train_window, config):
        assert not forbidden.intersection(history.columns)
        return _successful_prediction(direction=DIRECTION_UP)

    run_rolling_backtest(klines, config=config, predict_fn=inspect_predict)
