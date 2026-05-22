"""Tests for Telegram notification formatting and delivery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.notify.telegram import (
    TelegramAPIError,
    TelegramConfigError,
    TelegramNotifier,
    escape_html,
    format_alert_message,
    format_health_check_message,
    format_signal_message,
    main,
)
from src.signals.signal_engine import (
    ConfidenceComponents,
    SIGNAL_DOWN,
    SIGNAL_UP,
    TradingSignal,
)
from src.utils.config import Settings


def _sample_signal(**overrides) -> TradingSignal:
    defaults = {
        "symbol": "BTCUSDT",
        "timestamp_ms": 1_700_000_000_000,
        "current_price": 42_500.12,
        "expiry_timestamp_ms": 1_700_000_600_000,
        "direction": SIGNAL_UP,
        "predicted_cumulative_return": 0.0025,
        "confidence": 0.82,
        "confidence_threshold": 0.70,
        "should_push_telegram": True,
        "is_direction_reversal": False,
        "arima_model_order": (1, 0, 1),
        "spread_bps": 4.5,
        "book_imbalance": 0.3,
        "volume_filter_passed": True,
        "spread_filter_passed": True,
        "cooldown_blocked": False,
        "trigger_summary": "ARIMA UP; confidence=0.820; magnitude=0.85",
        "risk_note": "本工具仅提供预测提醒，不构成投资建议，请人工确认后再参与事件合约。",
        "components": ConfidenceComponents(0.85, 0.80, 0.90, 0.75, 0.95, 0.70),
    }
    defaults.update(overrides)
    return TradingSignal(**defaults)


def _sample_settings(**overrides) -> Settings:
    base = Settings.from_environ()
    values = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "prediction_minutes": 10,
        "arima_order": (1, 0, 1),
        "arima_series_type": "log_return",
        "use_auto_arima": False,
        "auto_arima_max_p": 5,
        "auto_arima_max_q": 5,
        "auto_arima_max_d": 2,
        "direction_threshold": 0.0,
        "train_window": 1440,
        "refit_interval_minutes": 5,
        "use_garch": True,
        "garch_order": (1, 1),
        "garch_mean": "constant",
        "garch_dist": "normal",
        "garch_min_train_points": 100,
        "garch_vol_scale": 1.0,
        "garch_failure_mode": "hold",
        "aggregation_mode": "volatility_adjusted_arima",
        "aggregation_min_snr": 0.8,
        "garch_extreme_vol_action": "hold",
        "garch_vol_weight": 0.35,
        "confidence_threshold": 0.70,
        "signal_cooldown_minutes": 10,
        "max_spread_bps": 50.0,
        "binance_market": "spot",
        "binance_api_key": None,
        "binance_api_secret": None,
        "binance_testnet": False,
        "telegram_bot_token": "123456789:AAExampleTokenValue",
        "telegram_chat_id": "987654321",
        "dry_run": True,
        "log_level": "INFO",
        "live_poll_interval_seconds": 10.0,
        "live_kline_limit": 2,
        "live_max_retries": 5,
        "live_retry_backoff": 1.0,
        "live_max_consecutive_errors": 10,
        "live_error_retry_delay_seconds": 5.0,
        "data_dir": base.data_dir,
        "logs_dir": base.logs_dir,
        "project_root": base.project_root,
    }
    values.update(overrides)
    return Settings(**values)


def test_escape_html_escapes_special_characters() -> None:
    assert escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_format_signal_message_contains_required_fields() -> None:
    message = format_signal_message(_sample_signal())

    assert "BTCUSDT" in message
    assert "涨 (UP)" in message
    assert "42,500.12" in message
    assert "2023-11-14 22:13:20 UTC" in message
    assert "2023-11-14 22:23:20 UTC" in message
    assert "82.0%" in message
    assert "ARIMA UP; confidence=0.820; magnitude=0.85" in message
    assert "不自动下单" in message
    assert "人工确认" in message


def test_format_signal_message_marks_direction_reversal() -> None:
    message = format_signal_message(
        _sample_signal(direction=SIGNAL_DOWN, is_direction_reversal=True)
    )

    assert "跌 (DOWN)" in message
    assert "方向反转" in message


def test_format_signal_message_shows_garch_context_when_present() -> None:
    message = format_signal_message(
        _sample_signal(
            trigger_summary="ARIMA-GARCH UP; confidence=0.820",
            volatility_level="HIGH",
            garch_volatility=0.001234,
            adjusted_snr=1.25,
        )
    )

    assert "波动等级:" in message
    assert "HIGH" in message
    assert "GARCH 波动率:" in message
    assert "0.001234" in message
    assert "调整后 SNR:" in message
    assert "1.250" in message
    assert "不自动下单" in message
    assert "人工确认" in message


def test_format_signal_message_omits_garch_context_when_absent() -> None:
    message = format_signal_message(_sample_signal())

    assert "波动等级:" not in message
    assert "GARCH 波动率:" not in message
    assert "调整后 SNR:" not in message


def test_format_health_check_message_contains_runtime_info() -> None:
    settings = _sample_settings(dry_run=False)
    message = format_health_check_message(
        symbol=settings.symbol,
        interval=settings.interval,
        prediction_minutes=settings.prediction_minutes,
        confidence_threshold=settings.confidence_threshold,
        dry_run=False,
        train_window=settings.train_window,
        arima_order=settings.arima_order,
        use_garch=settings.use_garch,
        garch_order=settings.garch_order,
    )

    assert "ARIMA-GARCH 预测工具已启动" in message
    assert "ARIMA-GARCH 聚合" in message
    assert "BTCUSDT" in message
    assert "1m" in message
    assert "70.0%" in message
    assert "live（允许推送信号）" in message


def test_format_health_check_message_arima_only_mode() -> None:
    settings = _sample_settings(use_garch=False)
    message = format_health_check_message(
        symbol=settings.symbol,
        interval=settings.interval,
        prediction_minutes=settings.prediction_minutes,
        confidence_threshold=settings.confidence_threshold,
        dry_run=True,
        train_window=settings.train_window,
        arima_order=settings.arima_order,
        use_garch=False,
    )

    assert "ARIMA 预测工具已启动" in message
    assert "ARIMA 单模型" in message
    assert "ARIMA-GARCH" not in message
    assert "GARCH 阶数:" not in message


def test_format_health_check_message_includes_garch_order() -> None:
    settings = _sample_settings(use_garch=True, garch_order=(1, 1))
    message = format_health_check_message(
        symbol=settings.symbol,
        interval=settings.interval,
        prediction_minutes=settings.prediction_minutes,
        confidence_threshold=settings.confidence_threshold,
        dry_run=True,
        train_window=settings.train_window,
        arima_order=settings.arima_order,
        use_garch=True,
        garch_order=settings.garch_order,
    )

    assert "GARCH 阶数:" in message
    assert "(1, 1)" in message


def test_format_alert_message_includes_exception_details() -> None:
    message = format_alert_message(
        "数据采集失败",
        "连续多次请求 Binance API 失败",
        source="live_collector",
        exception=RuntimeError("connection reset"),
    )

    assert "系统告警" in message
    assert "数据采集失败" in message
    assert "live_collector" in message
    assert "RuntimeError" in message
    assert "connection reset" in message


def test_validate_credentials_requires_token_and_chat_id() -> None:
    notifier = TelegramNotifier(bot_token=None, chat_id="123")

    with pytest.raises(TelegramConfigError, match="TELEGRAM_BOT_TOKEN"):
        notifier.validate_credentials()

    notifier = TelegramNotifier(bot_token="123:abc", chat_id=None)
    with pytest.raises(TelegramConfigError, match="TELEGRAM_CHAT_ID"):
        notifier.validate_credentials()


def test_send_message_dry_run_skips_http() -> None:
    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=True,
    )

    result = notifier.send_message("hello")

    assert result["ok"] is True
    assert result["dry_run"] is True


def test_send_message_posts_to_telegram_api() -> None:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"ok": True, "result": {"message_id": 1}}
    session.post.return_value = response

    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=False,
        max_retries=1,
        session=session,
    )

    result = notifier.send_message("<b>test</b>")

    assert result["ok"] is True
    session.post.assert_called_once()
    call_kwargs = session.post.call_args.kwargs
    assert call_kwargs["json"]["chat_id"] == "987654321"
    assert call_kwargs["json"]["text"] == "<b>test</b>"
    assert call_kwargs["json"]["parse_mode"] == "HTML"


def test_send_message_retries_and_raises_on_failure() -> None:
    session = MagicMock()
    session.post.side_effect = requests.Timeout("timed out")

    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=False,
        max_retries=2,
        retry_backoff=0.0,
        session=session,
    )

    with pytest.raises(TelegramAPIError, match="Failed to send Telegram message"):
        notifier.send_message("hello")

    assert session.post.call_count == 2


def test_notify_signal_skips_when_not_marked_for_push() -> None:
    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=True,
    )

    result = notifier.notify_signal(_sample_signal(should_push_telegram=False))

    assert result is None


def test_notify_signal_sends_when_marked_for_push() -> None:
    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=True,
    )

    result = notifier.notify_signal(_sample_signal(should_push_telegram=True))

    assert result == {"ok": True, "dry_run": True}


def test_from_settings_uses_env_credentials() -> None:
    settings = _sample_settings()
    notifier = TelegramNotifier.from_settings(settings)

    assert notifier.bot_token == "123456789:AAExampleTokenValue"
    assert notifier.chat_id == "987654321"
    assert notifier.dry_run is True


def test_send_health_check_uses_settings_context() -> None:
    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=True,
    )
    settings = _sample_settings()

    result = notifier.send_health_check(settings, extra_note="startup")

    assert result["dry_run"] is True


def test_send_health_check_reflects_arima_only_settings() -> None:
    notifier = TelegramNotifier(
        bot_token="123456789:AAExampleTokenValue",
        chat_id="987654321",
        dry_run=True,
    )
    settings = _sample_settings(use_garch=False)

    with patch.object(notifier, "send_message", wraps=notifier.send_message) as send_mock:
        notifier.send_health_check(settings)

    message = send_mock.call_args.args[0]
    assert "ARIMA 单模型" in message
    assert "GARCH 阶数:" not in message


def test_main_test_requires_credentials(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
SYMBOL=BTCUSDT
INTERVAL=1m
PREDICTION_MINUTES=10
ARIMA_ORDER=1,0,1
TRAIN_WINDOW=1440
CONFIDENCE_THRESHOLD=0.70
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DRY_RUN=false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with patch("src.notify.telegram.load_settings") as load_mock:
        settings = _sample_settings(
            telegram_bot_token=None,
            telegram_chat_id=None,
            dry_run=False,
            logs_dir=tmp_path / "logs",
        )
        load_mock.return_value = settings
        exit_code = main(["--test"])

    assert exit_code == 1


def test_main_test_dry_run_succeeds_without_http(monkeypatch) -> None:
    settings = _sample_settings(dry_run=True, logs_dir=Settings.from_environ().logs_dir)
    monkeypatch.setattr("src.notify.telegram.load_settings", lambda **_: settings)

    exit_code = main(["--test", "--dry-run"])

    assert exit_code == 0


def test_main_test_sends_message_when_not_dry_run(monkeypatch) -> None:
    settings = _sample_settings(dry_run=True, logs_dir=Settings.from_environ().logs_dir)
    monkeypatch.setattr("src.notify.telegram.load_settings", lambda **_: settings)

    with patch.object(TelegramNotifier, "send_health_check", return_value={"ok": True}) as send_mock:
        exit_code = main(["--test"])

    assert exit_code == 0
    send_mock.assert_called_once()
