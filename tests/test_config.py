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
USE_GARCH=true
GARCH_ORDER=1,1
GARCH_MEAN=constant
GARCH_DIST=normal
GARCH_MIN_TRAIN_POINTS=100
GARCH_VOL_SCALE=1.0
GARCH_FAILURE_MODE=hold
AGGREGATION_MODE=volatility_adjusted_arima
AGGREGATION_MIN_SNR=0.8
GARCH_EXTREME_VOL_ACTION=hold
GARCH_VOL_WEIGHT=0.35
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
    assert settings.use_garch is True
    assert settings.garch_failure_mode == "hold"
    assert settings.garch_extreme_vol_action == "hold"
    assert settings.aggregation_min_snr == 0.8


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
                "USE_GARCH",
                "GARCH_",
                "AGGREGATION_",
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
    assert settings.use_garch is True
    assert settings.garch_failure_mode == "hold"
    assert settings.garch_extreme_vol_action == "hold"
    assert settings.aggregation_min_snr == 0.8
    assert settings.use_garch is True
    assert settings.garch_order == (1, 1)
    assert settings.garch_mean == "constant"
    assert settings.garch_dist == "normal"
    assert settings.garch_min_train_points == 100
    assert settings.garch_vol_scale == 1.0
    assert settings.garch_failure_mode == "hold"
    assert settings.aggregation_mode == "volatility_adjusted_arima"
    assert settings.aggregation_min_snr == 0.8
    assert settings.garch_extreme_vol_action == "hold"
    assert settings.garch_vol_weight == 0.35


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


def test_garch_settings_from_env_file(tmp_path: Path, valid_env_content: str) -> None:
    garch_block = """
USE_GARCH=false
GARCH_ORDER=2,1
GARCH_MEAN=zero
GARCH_DIST=t
GARCH_MIN_TRAIN_POINTS=120
GARCH_VOL_SCALE=1.5
GARCH_FAILURE_MODE=fallback_to_arima
AGGREGATION_MODE=volatility_adjusted_arima
AGGREGATION_MIN_SNR=1.2
GARCH_EXTREME_VOL_ACTION=allow_with_penalty
GARCH_VOL_WEIGHT=0.5
"""
    env_file = _write_env(tmp_path / ".env", valid_env_content + garch_block)

    settings = load_settings(env_file=env_file)

    assert settings.use_garch is False
    assert settings.garch_order == (2, 1)
    assert settings.garch_mean == "zero"
    assert settings.garch_dist == "t"
    assert settings.garch_min_train_points == 120
    assert settings.garch_vol_scale == 1.5
    assert settings.garch_failure_mode == "fallback_to_arima"
    assert settings.aggregation_min_snr == 1.2
    assert settings.garch_extreme_vol_action == "allow_with_penalty"
    assert settings.garch_vol_weight == 0.5


def test_invalid_garch_order_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_ORDER=1,1,1")

    with pytest.raises(ConfigError, match="GARCH_ORDER"):
        load_settings(env_file=env_file)


def test_invalid_garch_mean_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_MEAN=ar")

    with pytest.raises(ConfigError, match="GARCH_MEAN"):
        load_settings(env_file=env_file)


def test_invalid_garch_dist_raises_config_error(tmp_path: Path, valid_env_content: str) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_DIST=skewt")

    with pytest.raises(ConfigError, match="GARCH_DIST"):
        load_settings(env_file=env_file)


def test_garch_min_train_points_below_minimum_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_MIN_TRAIN_POINTS=30")

    with pytest.raises(ConfigError, match="GARCH_MIN_TRAIN_POINTS"):
        load_settings(env_file=env_file)


def test_invalid_garch_vol_scale_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_VOL_SCALE=0")

    with pytest.raises(ConfigError, match="GARCH_VOL_SCALE"):
        load_settings(env_file=env_file)


def test_invalid_garch_failure_mode_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_FAILURE_MODE=ignore")

    with pytest.raises(ConfigError, match="GARCH_FAILURE_MODE"):
        load_settings(env_file=env_file)


def test_invalid_aggregation_mode_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nAGGREGATION_MODE=arima_only")

    with pytest.raises(ConfigError, match="AGGREGATION_MODE"):
        load_settings(env_file=env_file)


def test_negative_aggregation_min_snr_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nAGGREGATION_MIN_SNR=-0.1")

    with pytest.raises(ConfigError, match="AGGREGATION_MIN_SNR"):
        load_settings(env_file=env_file)


def test_invalid_garch_extreme_vol_action_raises_config_error(
    tmp_path: Path, valid_env_content: str
) -> None:
    env_file = _write_env(
        tmp_path / ".env", valid_env_content + "\nGARCH_EXTREME_VOL_ACTION=trade_anyway"
    )

    with pytest.raises(ConfigError, match="GARCH_EXTREME_VOL_ACTION"):
        load_settings(env_file=env_file)


def test_invalid_garch_vol_weight_raises_config_error(
    tmp_path: Path, valid_env_content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = _write_env(tmp_path / ".env", valid_env_content + "\nGARCH_VOL_WEIGHT=1.5")

    with pytest.raises(ConfigError, match="GARCH_VOL_WEIGHT"):
        load_settings(env_file=env_file)

    monkeypatch.delenv("GARCH_VOL_WEIGHT", raising=False)
