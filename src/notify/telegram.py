"""Telegram Bot API notification delivery for trading signals and alerts."""

from __future__ import annotations

import argparse
import html
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

from src.signals.signal_engine import SIGNAL_DOWN, SIGNAL_HOLD, SIGNAL_UP, TradingSignal
from src.utils.config import Settings, load_settings

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 1.0
DEFAULT_REQUEST_TIMEOUT = 30.0

_DIRECTION_LABELS = {
    SIGNAL_UP: "涨 (UP)",
    SIGNAL_DOWN: "跌 (DOWN)",
    SIGNAL_HOLD: "观望 (HOLD)",
}


class TelegramConfigError(ValueError):
    """Raised when Telegram credentials are missing or invalid for sending."""


class TelegramAPIError(RuntimeError):
    """Raised when the Telegram Bot API returns an error after retries."""


def escape_html(text: str) -> str:
    """Escape user-controlled text for Telegram HTML parse mode."""
    return html.escape(text, quote=False)


def format_direction(direction: str) -> str:
    """Map internal direction codes to human-readable labels."""
    return _DIRECTION_LABELS.get(direction, direction)


def format_timestamp_ms(timestamp_ms: int) -> str:
    """Format epoch milliseconds as UTC timestamp."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_price(price: Optional[float]) -> str:
    """Format price with adaptive precision."""
    if price is None:
        return "N/A"
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    return f"{price:.8f}"


def format_signal_message(signal: TradingSignal) -> str:
    """Build a Telegram HTML message for a trading signal."""
    direction_label = escape_html(format_direction(signal.direction))
    symbol = escape_html(signal.symbol)
    current_price = escape_html(format_price(signal.current_price))
    signal_time = escape_html(format_timestamp_ms(signal.timestamp_ms))
    expiry_time = escape_html(format_timestamp_ms(signal.expiry_timestamp_ms))
    confidence = escape_html(f"{signal.confidence:.1%}")
    trigger_reason = escape_html(signal.trigger_summary)
    risk_note = escape_html(signal.risk_note)

    lines = [
        "<b>📊 事件合约预测信号</b>",
        "",
        f"<b>标的:</b> {symbol}",
        f"<b>方向:</b> {direction_label}",
        f"<b>当前价格:</b> {current_price}",
        f"<b>信号时间:</b> {signal_time}",
        f"<b>预测到期:</b> {expiry_time}",
        f"<b>置信度:</b> {confidence}",
        f"<b>触发原因:</b> {trigger_reason}",
    ]

    if signal.is_direction_reversal:
        lines.append("<b>备注:</b> 方向反转")

    if signal.predicted_cumulative_return is not None:
        pct = signal.predicted_cumulative_return * 100.0
        lines.append(f"<b>预测累计收益:</b> {escape_html(f'{pct:+.4f}')}%")

    lines.extend(
        [
            "",
            f"⚠️ {risk_note}",
            "本工具不自动下单，请人工确认后再参与事件合约。",
        ]
    )
    return "\n".join(lines)


def format_health_check_message(
    *,
    symbol: str,
    interval: str,
    prediction_minutes: int,
    confidence_threshold: float,
    dry_run: bool,
    train_window: int,
    arima_order: tuple[int, int, int],
    extra_note: Optional[str] = None,
) -> str:
    """Build a startup health-check message."""
    mode = "dry-run（仅日志，不推送信号）" if dry_run else "live（允许推送信号）"
    order_text = f"({arima_order[0]}, {arima_order[1]}, {arima_order[2]})"
    lines = [
        "<b>✅ ARIMA 预测工具已启动</b>",
        "",
        f"<b>标的:</b> {escape_html(symbol)}",
        f"<b>K 线周期:</b> {escape_html(interval)}",
        f"<b>预测窗口:</b> {prediction_minutes} 分钟",
        f"<b>训练窗口:</b> {train_window} 根 K 线",
        f"<b>ARIMA 阶数:</b> {escape_html(order_text)}",
        f"<b>置信度阈值:</b> {confidence_threshold:.1%}",
        f"<b>运行模式:</b> {escape_html(mode)}",
        "",
        "Telegram 通知通道正常，健康检查通过。",
    ]
    if extra_note:
        lines.insert(2, f"<b>说明:</b> {escape_html(extra_note)}")
    return "\n".join(lines)


def format_alert_message(
    title: str,
    message: str,
    *,
    source: Optional[str] = None,
    exception: Optional[BaseException] = None,
) -> str:
    """Build an exception or operational alert message."""
    lines = [
        "<b>🚨 系统告警</b>",
        "",
        f"<b>标题:</b> {escape_html(title)}",
        f"<b>详情:</b> {escape_html(message)}",
    ]
    if source:
        lines.append(f"<b>来源:</b> {escape_html(source)}")
    if exception is not None:
        lines.append(f"<b>异常:</b> {escape_html(type(exception).__name__)}: {escape_html(str(exception))}")
    lines.extend(
        [
            "",
            "请检查日志并人工确认系统状态。",
        ]
    )
    return "\n".join(lines)


class TelegramNotifier:
    """Send Telegram notifications via the Bot API."""

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        *,
        dry_run: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.request_timeout = request_timeout
        self._session = session or requests.Session()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dry_run: Optional[bool] = None,
        session: Optional[requests.Session] = None,
    ) -> TelegramNotifier:
        """Create a notifier from application settings."""
        return cls(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            dry_run=settings.dry_run if dry_run is None else dry_run,
            session=session,
        )

    def validate_credentials(self) -> None:
        """Ensure bot token and chat id are configured before sending."""
        errors: list[str] = []
        if not self.bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is not configured")
        if not self.chat_id:
            errors.append("TELEGRAM_CHAT_ID is not configured")
        if errors:
            raise TelegramConfigError("; ".join(errors))

    @property
    def _send_message_url(self) -> str:
        if not self.bot_token:
            raise TelegramConfigError("TELEGRAM_BOT_TOKEN is not configured")
        return f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage"

    def send_message(
        self,
        text: str,
        *,
        disable_notification: bool = False,
        parse_mode: str = "HTML",
    ) -> dict[str, Any]:
        """
        Send a text message to the configured chat.

        In dry-run mode the message is logged and no HTTP request is made.
        """
        self.validate_credentials()

        if self.dry_run:
            logger.info("Telegram dry-run message:\n%s", text)
            return {"ok": True, "dry_run": True}

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    self._send_message_url,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    description = data.get("description", "unknown Telegram API error")
                    raise TelegramAPIError(description)
                logger.info("Telegram message sent successfully")
                return data
            except (requests.RequestException, TelegramAPIError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Telegram send failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)

        raise TelegramAPIError(
            f"Failed to send Telegram message after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def send_signal(self, signal: TradingSignal) -> dict[str, Any]:
        """Send a formatted trading signal message."""
        message = format_signal_message(signal)
        logger.info(
            "Sending Telegram signal %s %s confidence=%.3f",
            signal.symbol,
            signal.direction,
            signal.confidence,
        )
        return self.send_message(message)

    def notify_signal(self, signal: TradingSignal) -> Optional[dict[str, Any]]:
        """Send a signal only when the signal engine marked it for Telegram push."""
        if not signal.should_push_telegram:
            logger.debug(
                "Skipping Telegram push for %s %s (should_push_telegram=false)",
                signal.symbol,
                signal.direction,
            )
            return None
        return self.send_signal(signal)

    def send_health_check(
        self,
        settings: Settings,
        *,
        extra_note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a startup health-check message."""
        message = format_health_check_message(
            symbol=settings.symbol,
            interval=settings.interval,
            prediction_minutes=settings.prediction_minutes,
            confidence_threshold=settings.confidence_threshold,
            dry_run=self.dry_run,
            train_window=settings.train_window,
            arima_order=settings.arima_order,
            extra_note=extra_note,
        )
        logger.info("Sending Telegram health check for %s", settings.symbol)
        return self.send_message(message)

    def send_alert(
        self,
        title: str,
        message: str,
        *,
        source: Optional[str] = None,
        exception: Optional[BaseException] = None,
    ) -> dict[str, Any]:
        """Send an operational or exception alert."""
        alert_text = format_alert_message(
            title,
            message,
            source=source,
            exception=exception,
        )
        logger.warning("Sending Telegram alert: %s", title)
        return self.send_message(alert_text)


def setup_logging(logs_dir, level: str) -> None:
    """Configure console and file logging for Telegram CLI usage."""
    from pathlib import Path

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_handler = logging.FileHandler(logs_path / "telegram.log", encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Send Telegram notification test messages")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a health-check test message using .env credentials",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the message without calling Telegram API (overrides DRY_RUN from .env)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for Telegram connectivity tests."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.test:
        parser.error("No action specified. Use --test to send a health-check message.")

    settings = load_settings(validate=False)
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(settings.logs_dir, log_level)

    dry_run = settings.dry_run if not args.dry_run else True
    if args.test and not args.dry_run:
        dry_run = False

    try:
        settings.validate()
    except Exception as exc:
        logger.error("Configuration validation failed: %s", exc)
        return 1

    notifier = TelegramNotifier.from_settings(settings, dry_run=dry_run)

    try:
        notifier.validate_credentials()
    except TelegramConfigError as exc:
        logger.error("%s", exc)
        logger.error(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env, "
            "then send /start to your bot before retrying."
        )
        return 1

    try:
        notifier.send_health_check(
            settings,
            extra_note="这是一条 Telegram 连通性测试消息。",
        )
    except TelegramAPIError as exc:
        logger.error("Telegram API error: %s", exc)
        return 1

    if dry_run:
        logger.info("Dry-run complete. No message was sent to Telegram.")
    else:
        logger.info("Test message sent. Check your Telegram chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
