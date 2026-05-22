"""Live trading loop: market data, ARIMA-GARCH prediction, signals, and Telegram."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, replace
from typing import Optional, Union

import pandas as pd

from src.data.live_collector import LiveMarketCollector, PollResult
from src.data.market_data_source import MarketDataSource
from src.data.market_data_storage import MarketDataStorage
from src.data.order_book_schema import OrderBookSnapshot
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
from src.notify.telegram import TelegramNotifier
from src.signals.signal_engine import SignalEngine, TradingSignal
from src.utils.config import Settings

logger = logging.getLogger(__name__)
model_logger = logging.getLogger("src.models")
signal_logger = logging.getLogger("src.signals")

PredictionResult = Union[ARIMAPredictionResult, CombinedPredictionResult]


@dataclass(frozen=True)
class LiveCycleResult:
    """Outcome of one live prediction cycle."""

    poll_result: PollResult
    prediction: PredictionResult
    signal: TradingSignal
    refit_performed: bool
    telegram_sent: bool


class LiveTradingRunner:
    """
    Orchestrate live data collection, model refit, signal evaluation, and alerts.

    When ``use_garch`` is enabled, refits ARIMA and GARCH at most once per
    ``refit_interval_minutes`` and caches the aggregated prediction. With GARCH disabled,
    only ARIMA is fitted and cached. Between refits, the cached prediction is reused while
    order book and volume filters are re-evaluated each cycle.
    """

    def __init__(
        self,
        settings: Settings,
        source: MarketDataSource,
        storage: MarketDataStorage,
        *,
        collector: Optional[LiveMarketCollector] = None,
        predictor_config: Optional[ARIMAPredictorConfig] = None,
        signal_engine: Optional[SignalEngine] = None,
        notifier: Optional[TelegramNotifier] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.settings = settings
        self.source = source
        self.storage = storage
        self.dry_run = settings.dry_run if dry_run is None else dry_run

        self.collector = collector or LiveMarketCollector(
            source,
            storage,
            settings.symbol,
            settings.interval,
            poll_interval_seconds=settings.live_poll_interval_seconds,
            kline_limit=max(settings.live_kline_limit, 2),
            max_consecutive_errors=settings.live_max_consecutive_errors,
            error_retry_delay_seconds=settings.live_error_retry_delay_seconds,
        )

        self.predictor_config = predictor_config or ARIMAPredictorConfig.from_settings(settings)
        self.garch_config = GARCHPredictorConfig.from_settings(settings)
        self.aggregator_config = AggregatorConfig.from_settings(settings)
        self.signal_engine = signal_engine or SignalEngine.from_settings(settings)
        self.notifier = notifier or TelegramNotifier.from_settings(settings, dry_run=self.dry_run)

        self._stop_requested = False
        self._cached_prediction: Optional[PredictionResult] = None
        self._last_refit_monotonic: float = 0.0
        self._last_order_book: Optional[OrderBookSnapshot] = None
        self._latest_kline_timestamp: Optional[int] = None

    def request_stop(self) -> None:
        """Request graceful shutdown after the current cycle."""
        self._stop_requested = True
        self.collector.request_stop()

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def send_startup_health_check(self) -> None:
        """Send a Telegram health-check message on startup."""
        try:
            self.notifier.send_health_check(self.settings)
            logger.info("Startup health check completed (dry_run=%s)", self.dry_run)
        except Exception as exc:
            logger.warning("Startup health check failed: %s", exc, exc_info=True)

    def _ensure_training_klines(self) -> pd.DataFrame:
        """Load recent klines from storage, bootstrapping from API when needed."""
        klines = self.storage.load_klines(
            self.settings.symbol,
            self.settings.interval,
            tail=self.settings.train_window,
        )
        if len(klines) >= self.settings.train_window:
            return klines

        needed = self.settings.train_window
        logger.info(
            "Local kline history insufficient (%d/%d), fetching from API",
            len(klines),
            needed,
        )
        fetched = self.source.fetch_latest_klines(
            self.settings.symbol,
            self.settings.interval,
            limit=min(needed, 1000),
        )
        if not fetched.empty:
            self.storage.append_klines(self.settings.symbol, self.settings.interval, fetched)
            klines = self.storage.load_klines(
                self.settings.symbol,
                self.settings.interval,
                tail=self.settings.train_window,
            )
        return klines

    def _should_refit(self, latest_kline_timestamp: Optional[int]) -> bool:
        if self._cached_prediction is None:
            return True

        elapsed_minutes = (time.monotonic() - self._last_refit_monotonic) / 60.0
        if elapsed_minutes >= self.settings.refit_interval_minutes:
            return True

        if (
            latest_kline_timestamp is not None
            and self._latest_kline_timestamp is not None
            and latest_kline_timestamp > self._latest_kline_timestamp
        ):
            return True

        return False

    def _run_arima_prediction(self, klines: pd.DataFrame) -> ARIMAPredictionResult:
        result = predict_from_klines(
            klines,
            train_window=self.settings.train_window,
            config=self.predictor_config,
        )
        if result.success:
            model_logger.info(
                "ARIMA prediction %s direction=%s return=%.6f order=%s",
                self.settings.symbol,
                result.direction,
                result.predicted_cumulative_return or 0.0,
                result.model_order,
            )
        else:
            model_logger.warning(
                "ARIMA prediction failed for %s: %s (%s)",
                self.settings.symbol,
                result.error_message,
                result.error_code,
            )
        return result

    def _run_combined_prediction(self, klines: pd.DataFrame) -> CombinedPredictionResult:
        arima = self._run_arima_prediction(klines)
        garch = predict_volatility_from_klines(
            klines,
            train_window=self.settings.train_window,
            config=self.garch_config,
        )
        if garch.success:
            model_logger.info(
                "GARCH volatility %s cumulative_vol=%.6f level=%s order=%s",
                self.settings.symbol,
                garch.cumulative_volatility or 0.0,
                garch.volatility_level,
                garch.model_order,
            )
        else:
            model_logger.warning(
                "GARCH volatility failed for %s: %s (%s)",
                self.settings.symbol,
                garch.error_message,
                garch.error_code,
            )

        combined = aggregate_predictions(arima, garch, config=self.aggregator_config)
        model_logger.info(
            "ARIMA-GARCH aggregation %s arima_direction=%s garch_vol=%s "
            "volatility_level=%s direction=%s adjusted_snr=%s",
            self.settings.symbol,
            combined.arima_direction,
            combined.garch_volatility,
            combined.volatility_level,
            combined.direction,
            combined.adjusted_snr,
        )
        return combined

    def _run_prediction(self, klines: pd.DataFrame) -> PredictionResult:
        if self.settings.use_garch:
            return self._run_combined_prediction(klines)
        return self._run_arima_prediction(klines)

    def _apply_latest_price(
        self,
        prediction: PredictionResult,
        klines: pd.DataFrame,
    ) -> PredictionResult:
        """Refresh current_price on a cached prediction from the latest close."""
        if klines.empty or not prediction.success:
            return prediction

        latest_close = float(klines.iloc[-1]["close"])
        if prediction.current_price == latest_close:
            return prediction

        if isinstance(prediction, CombinedPredictionResult):
            return replace(prediction, current_price=latest_close)

        return ARIMAPredictionResult(
            success=prediction.success,
            predicted_cumulative_return=prediction.predicted_cumulative_return,
            direction=prediction.direction,
            interval_lower=prediction.interval_lower,
            interval_upper=prediction.interval_upper,
            residual_volatility=prediction.residual_volatility,
            model_order=prediction.model_order,
            series_type=prediction.series_type,
            current_price=latest_close,
            prediction_horizon_minutes=prediction.prediction_horizon_minutes,
            forecast_steps=prediction.forecast_steps,
            train_points=prediction.train_points,
            error_code=prediction.error_code,
            error_message=prediction.error_message,
            error_detail=prediction.error_detail,
        )

    def run_cycle(self) -> LiveCycleResult:
        """Execute one full live cycle: poll, predict, signal, optional Telegram push."""
        poll_result = self.collector.poll_once()
        latest_ts = poll_result.latest_kline_timestamp
        if latest_ts is not None:
            self._latest_kline_timestamp = latest_ts

        klines = self._ensure_training_klines()
        refit = self._should_refit(latest_ts)

        if refit:
            prediction = self._run_prediction(klines)
            self._cached_prediction = prediction
            self._last_refit_monotonic = time.monotonic()
        else:
            prediction = self._apply_latest_price(self._cached_prediction, klines)
            model_logger.debug("Reusing cached prediction for %s", self.settings.symbol)

        timestamp_ms = latest_ts or int(time.time() * 1000)
        orderbook = poll_result.order_book or self._last_order_book
        if poll_result.order_book is not None:
            self._last_order_book = poll_result.order_book

        signal = self.signal_engine.evaluate(
            prediction,
            symbol=self.settings.symbol,
            timestamp_ms=timestamp_ms,
            klines=klines,
            orderbook=orderbook,
        )

        signal_logger.info(
            "Signal %s %s confidence=%.3f push=%s summary=%s",
            signal.symbol,
            signal.direction,
            signal.confidence,
            signal.should_push_telegram,
            signal.trigger_summary,
        )

        telegram_sent = False
        if signal.should_push_telegram:
            if self.dry_run:
                logger.info(
                    "Dry-run: would push Telegram signal %s %s confidence=%.3f",
                    signal.symbol,
                    signal.direction,
                    signal.confidence,
                )
            else:
                try:
                    self.notifier.notify_signal(signal)
                    telegram_sent = True
                except Exception as exc:
                    logger.exception("Failed to send Telegram signal: %s", exc)
                    self._send_alert_safe(
                        "Telegram signal delivery failed",
                        f"Failed to push {signal.symbol} {signal.direction} signal.",
                        exception=exc,
                    )

        return LiveCycleResult(
            poll_result=poll_result,
            prediction=prediction,
            signal=signal,
            refit_performed=refit,
            telegram_sent=telegram_sent,
        )

    def _send_alert_safe(
        self,
        title: str,
        message: str,
        *,
        exception: Optional[BaseException] = None,
    ) -> None:
        try:
            self.notifier.send_alert(title, message, source="live_runner", exception=exception)
        except Exception as alert_exc:
            logger.warning("Failed to send Telegram alert: %s", alert_exc, exc_info=True)

    def run(
        self,
        *,
        max_iterations: Optional[int] = None,
        send_health_check: bool = True,
    ) -> None:
        """Run the live loop until stopped, max_iterations reached, or too many errors."""
        self._install_signal_handlers()

        logger.info(
            "Starting live runner: symbol=%s interval=%s dry_run=%s refit=%dm poll=%.1fs",
            self.settings.symbol,
            self.settings.interval,
            self.dry_run,
            self.settings.refit_interval_minutes,
            self.settings.live_poll_interval_seconds,
        )

        if send_health_check:
            self.send_startup_health_check()

        iteration = 0
        consecutive_errors = 0

        try:
            while not self._stop_requested:
                if max_iterations is not None and iteration >= max_iterations:
                    logger.info("Reached max iterations (%d), stopping", max_iterations)
                    break

                try:
                    result = self.run_cycle()
                    consecutive_errors = 0
                    logger.info(
                        "Cycle #%d: klines_appended=%d direction=%s confidence=%.3f refit=%s telegram=%s",
                        iteration + 1,
                        result.poll_result.klines_appended,
                        result.signal.direction,
                        result.signal.confidence,
                        result.refit_performed,
                        result.telegram_sent,
                    )
                except Exception as exc:
                    consecutive_errors += 1
                    logger.exception(
                        "Live cycle failed (%d/%d consecutive): %s",
                        consecutive_errors,
                        self.settings.live_max_consecutive_errors,
                        exc,
                    )
                    if consecutive_errors >= self.settings.live_max_consecutive_errors:
                        self._send_alert_safe(
                            "Live runner stopped",
                            "Too many consecutive cycle failures.",
                            exception=exc,
                        )
                        raise
                    self._sleep(self.settings.live_error_retry_delay_seconds)
                    continue

                iteration += 1
                if self._stop_requested:
                    break
                self._sleep(self.settings.live_poll_interval_seconds)
        finally:
            logger.info("Live runner shutting down")
            self.source.close()

    def _install_signal_handlers(self) -> None:
        def _handle_stop(signum: int, _frame: object) -> None:
            logger.info("Received signal %s, stopping after current cycle", signum)
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_stop)
            except (AttributeError, ValueError):
                pass

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.5))
