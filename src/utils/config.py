"""Load and validate application settings from environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)
ALLOWED_BINANCE_MARKETS = frozenset({"spot", "futures"})
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{4,20}$")
TELEGRAM_TOKEN_PATTERN = re.compile(r"^\d+:[A-Za-z0-9_-]+$")


class ConfigError(ValueError):
    """Raised when configuration values fail validation."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in self.errors)
        super().__init__(message)


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError([f"Invalid boolean value: {value!r}"])


def _parse_int(name: str, value: Optional[str], default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError([f"{name} must be an integer, got: {value!r}"]) from exc


def _parse_float(name: str, value: Optional[str], default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError([f"{name} must be a number, got: {value!r}"]) from exc


def _parse_arima_order(value: Optional[str], default: Tuple[int, int, int] = (1, 0, 1)) -> Tuple[int, int, int]:
    if value is None or value.strip() == "":
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ConfigError([f"ARIMA_ORDER must have 3 comma-separated integers, got: {value!r}"])
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigError([f"ARIMA_ORDER must contain integers only, got: {value!r}"]) from exc
    return order[0], order[1], order[2]


def _optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment."""

    symbol: str
    interval: str
    prediction_minutes: int
    arima_order: Tuple[int, int, int]
    train_window: int
    refit_interval_minutes: int
    confidence_threshold: float
    signal_cooldown_minutes: int
    max_spread_bps: float
    binance_market: str
    binance_api_key: Optional[str]
    binance_api_secret: Optional[str]
    binance_testnet: bool
    telegram_bot_token: Optional[str]
    telegram_chat_id: Optional[str]
    dry_run: bool
    log_level: str
    data_dir: Path
    logs_dir: Path
    project_root: Path

    @classmethod
    def from_environ(cls, project_root: Path = PROJECT_ROOT) -> Settings:
        """Build settings from the current process environment."""
        data_dir = project_root / os.getenv("DATA_DIR", "data")
        logs_dir = project_root / os.getenv("LOGS_DIR", "logs")

        return cls(
            symbol=os.getenv("SYMBOL", "BTCUSDT").upper(),
            interval=os.getenv("INTERVAL", "1m"),
            prediction_minutes=_parse_int("PREDICTION_MINUTES", os.getenv("PREDICTION_MINUTES"), 10),
            arima_order=_parse_arima_order(os.getenv("ARIMA_ORDER")),
            train_window=_parse_int("TRAIN_WINDOW", os.getenv("TRAIN_WINDOW"), 1440),
            refit_interval_minutes=_parse_int(
                "REFIT_INTERVAL_MINUTES", os.getenv("REFIT_INTERVAL_MINUTES"), 5
            ),
            confidence_threshold=_parse_float(
                "CONFIDENCE_THRESHOLD", os.getenv("CONFIDENCE_THRESHOLD"), 0.70
            ),
            signal_cooldown_minutes=_parse_int(
                "SIGNAL_COOLDOWN_MINUTES", os.getenv("SIGNAL_COOLDOWN_MINUTES"), 10
            ),
            max_spread_bps=_parse_float("MAX_SPREAD_BPS", os.getenv("MAX_SPREAD_BPS"), 50.0),
            binance_market=os.getenv("BINANCE_MARKET", "spot").strip().lower(),
            binance_api_key=_optional_str(os.getenv("BINANCE_API_KEY")),
            binance_api_secret=_optional_str(os.getenv("BINANCE_API_SECRET")),
            binance_testnet=_parse_bool(os.getenv("BINANCE_TESTNET"), False),
            telegram_bot_token=_optional_str(os.getenv("TELEGRAM_BOT_TOKEN")),
            telegram_chat_id=_optional_str(os.getenv("TELEGRAM_CHAT_ID")),
            dry_run=_parse_bool(os.getenv("DRY_RUN"), True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            data_dir=data_dir,
            logs_dir=logs_dir,
            project_root=project_root,
        )

    def validate(self) -> None:
        """Validate configuration values and raise ConfigError on failure."""
        errors: list[str] = []

        if not SYMBOL_PATTERN.match(self.symbol):
            errors.append(f"SYMBOL must be 4-20 uppercase alphanumeric characters, got: {self.symbol!r}")

        if self.interval not in ALLOWED_INTERVALS:
            errors.append(
                f"INTERVAL must be one of {sorted(ALLOWED_INTERVALS)}, got: {self.interval!r}"
            )

        if self.prediction_minutes < 1 or self.prediction_minutes > 60:
            errors.append(f"PREDICTION_MINUTES must be between 1 and 60, got: {self.prediction_minutes}")

        p, d, q = self.arima_order
        if min(p, d, q) < 0:
            errors.append(f"ARIMA_ORDER values must be non-negative, got: {self.arima_order}")
        if p + q > 10:
            errors.append(f"ARIMA_ORDER p + q should not exceed 10, got: {self.arima_order}")

        if self.train_window < self.prediction_minutes:
            errors.append(
                "TRAIN_WINDOW must be greater than or equal to PREDICTION_MINUTES "
                f"({self.train_window} < {self.prediction_minutes})"
            )
        if self.train_window < 60:
            errors.append(f"TRAIN_WINDOW must be at least 60, got: {self.train_window}")

        if self.refit_interval_minutes < 1:
            errors.append(f"REFIT_INTERVAL_MINUTES must be at least 1, got: {self.refit_interval_minutes}")

        if not 0.0 < self.confidence_threshold <= 1.0:
            errors.append(
                f"CONFIDENCE_THRESHOLD must be in (0.0, 1.0], got: {self.confidence_threshold}"
            )

        if self.signal_cooldown_minutes < 0:
            errors.append(
                f"SIGNAL_COOLDOWN_MINUTES must be non-negative, got: {self.signal_cooldown_minutes}"
            )

        if self.max_spread_bps <= 0:
            errors.append(f"MAX_SPREAD_BPS must be positive, got: {self.max_spread_bps}")

        if self.binance_market not in ALLOWED_BINANCE_MARKETS:
            errors.append(
                f"BINANCE_MARKET must be one of {sorted(ALLOWED_BINANCE_MARKETS)}, "
                f"got: {self.binance_market!r}"
            )

        has_key = self.binance_api_key is not None
        has_secret = self.binance_api_secret is not None
        if has_key ^ has_secret:
            errors.append("BINANCE_API_KEY and BINANCE_API_SECRET must both be set or both be empty")

        if self.log_level not in ALLOWED_LOG_LEVELS:
            errors.append(
                f"LOG_LEVEL must be one of {sorted(ALLOWED_LOG_LEVELS)}, got: {self.log_level!r}"
            )

        if self.telegram_bot_token and not TELEGRAM_TOKEN_PATTERN.match(self.telegram_bot_token):
            errors.append("TELEGRAM_BOT_TOKEN format is invalid")

        if self.telegram_chat_id and not re.fullmatch(r"-?\d+", self.telegram_chat_id):
            errors.append(f"TELEGRAM_CHAT_ID must be numeric, got: {self.telegram_chat_id!r}")

        if not self.dry_run:
            if not self.telegram_bot_token:
                errors.append("TELEGRAM_BOT_TOKEN is required when DRY_RUN=false")
            if not self.telegram_chat_id:
                errors.append("TELEGRAM_CHAT_ID is required when DRY_RUN=false")

        if errors:
            raise ConfigError(errors)


def load_settings(env_file: Optional[Path] = None, *, validate: bool = True) -> Settings:
    """Load settings from .env and process environment variables."""
    env_path = env_file or PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        load_dotenv()

    settings = Settings.from_environ()
    if validate:
        settings.validate()
    return settings
