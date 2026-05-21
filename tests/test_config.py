"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.config import ConfigError, Settings, load_settings


def _write_env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def valid_env_content() -> str:
    return """
SYMBOL=BTCUSDT
INTERVAL=1m
PREDICTION_MINUTES=10
ARIMA_ORDER=1,0,1
TRAIN_WINDOW=1440
REFIT_INTERVAL_MINUTES=5
CONFIDENCE_THRESHOLD=0.70
SIGNAL_COOLDOWN_MINUTES=10
MAX_SPREAD_BPS=50
BINANCE_MARKET=spot
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=false
TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenValue
TELEGRAM_CHAT_ID=987654321
DRY_RUN=true
LOG_LEVEL=INFO
DATA_DIR=data
LOGS_DIR=logs
""".strip()


def test_load_settings_from_env_file(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content)

    settings = load_settings(env_file=env_file)

    assert settings.symbol == "BTCUSDT"
    assert settings.interval == "1m"
    assert settings.prediction_minutes == 10
    assert settings.arima_order == (1, 0, 1)
    assert settings.train_window == 1440
    assert settings.confidence_threshold == 0.70
    assert settings.binance_market == "spot"
    assert settings.binance_api_key is None
    assert settings.binance_api_secret is None
    assert settings.telegram_bot_token == "123456789:AAExampleTokenValue"
    assert settings.telegram_chat_id == "987654321"
    assert settings.dry_run is True


def test_settings_from_environ_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(
            (
                "SYMBOL",
                "INTERVAL",
                "PREDICTION_MINUTES",
                "ARIMA_ORDER",
                "TRAIN_WINDOW",
                "CONFIDENCE_THRESHOLD",
                "BINANCE_",
                "TELEGRAM_",
                "DRY_RUN",
                "LOG_LEVEL",
            )
        ):
            monkeypatch.delenv(key, raising=False)

    settings = Settings.from_environ()
    settings.validate()

    assert settings.symbol == "BTCUSDT"
    assert settings.interval == "1m"
    assert settings.prediction_minutes == 10
    assert settings.arima_order == (1, 0, 1)
    assert settings.confidence_threshold == 0.70
    assert settings.dry_run is True


def test_symbol_is_normalized_to_uppercase(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content.replace("BTCUSDT", "ethusdt"))

    settings = load_settings(env_file=env_file)

    assert settings.symbol == "ETHUSDT"


def test_invalid_symbol_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content.replace("BTCUSDT", "BTC"))

    with pytest.raises(ConfigError, match="SYMBOL"):
        load_settings(env_file=env_file)


def test_invalid_interval_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content.replace("INTERVAL=1m", "INTERVAL=2m"))

    with pytest.raises(ConfigError, match="INTERVAL"):
        load_settings(env_file=env_file)


def test_invalid_arima_order_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content.replace("ARIMA_ORDER=1,0,1", "ARIMA_ORDER=1,0"))

    with pytest.raises(ConfigError, match="ARIMA_ORDER"):
        load_settings(env_file=env_file)


def test_invalid_confidence_threshold_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(
        tmp_path / ".env", valid_env_content.replace("CONFIDENCE_THRESHOLD=0.70", "CONFIDENCE_THRESHOLD=1.5")
    )

    with pytest.raises(ConfigError, match="CONFIDENCE_THRESHOLD"):
        load_settings(env_file=env_file)


def test_train_window_must_cover_prediction_window(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        valid_env_content.replace("TRAIN_WINDOW=1440", "TRAIN_WINDOW=5"),
    )

    with pytest.raises(ConfigError, match="TRAIN_WINDOW"):
        load_settings(env_file=env_file)


def test_binance_api_key_and_secret_must_be_paired(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        valid_env_content.replace("BINANCE_API_KEY=", "BINANCE_API_KEY=test-key"),
    )

    with pytest.raises(ConfigError, match="BINANCE_API_KEY and BINANCE_API_SECRET"):
        load_settings(env_file=env_file)


def test_telegram_required_when_not_dry_run(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        valid_env_content.replace("DRY_RUN=true", "DRY_RUN=false").replace(
            "TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenValue", "TELEGRAM_BOT_TOKEN="
        ),
    )

    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=env_file)

    assert any("TELEGRAM_BOT_TOKEN" in error for error in exc_info.value.errors)


def test_invalid_telegram_token_format(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(
        tmp_path / ".env",
        valid_env_content.replace(
            "TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenValue", "TELEGRAM_BOT_TOKEN=invalid-token"
        ),
    )

    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        load_settings(env_file=env_file)


def test_sensitive_values_are_not_hardcoded(tmp_path: Path, valid_env_content: str) -> None:
    token = "777777777:ZZZSensitiveToken"
    chat_id = "-1001234567890"
    api_key = "sensitive-api-key"
    api_secret = "sensitive-api-secret"

    env_file = _write_env(
        tmp_path / ".env",
        valid_env_content.replace("DRY_RUN=true", "DRY_RUN=false")
        .replace("TELEGRAM_BOT_TOKEN=123456789:AAExampleTokenValue", f"TELEGRAM_BOT_TOKEN={token}")
        .replace("TELEGRAM_CHAT_ID=987654321", f"TELEGRAM_CHAT_ID={chat_id}")
        .replace("BINANCE_API_KEY=", f"BINANCE_API_KEY={api_key}")
        .replace("BINANCE_API_SECRET=", f"BINANCE_API_SECRET={api_secret}"),
    )

    settings = load_settings(env_file=env_file)

    assert settings.telegram_bot_token == token
    assert settings.telegram_chat_id == chat_id
    assert settings.binance_api_key == api_key
    assert settings.binance_api_secret == api_secret


def test_load_settings_can_skip_validation(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content.replace("BTCUSDT", "BTC"))

    settings = load_settings(env_file=env_file, validate=False)

    assert settings.symbol == "BTC"
