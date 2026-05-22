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
ALLOWED_ARIMA_SERIES_TYPES = frozenset({"log_return", "price_diff"})
ALLOWED_GARCH_MEANS = frozenset({"constant", "zero"})
ALLOWED_GARCH_DISTS = frozenset({"normal", "t"})
ALLOWED_GARCH_FAILURE_MODES = frozenset({"hold", "fallback_to_arima"})
ALLOWED_AGGREGATION_MODES = frozenset({"volatility_adjusted_arima"})
ALLOWED_GARCH_EXTREME_VOL_ACTIONS = frozenset({"hold", "allow_with_penalty"})
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


def _parse_garch_order(value: Optional[str], default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    if value is None or value.strip() == "":
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ConfigError([f"GARCH_ORDER must have 2 comma-separated integers, got: {value!r}"])
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigError([f"GARCH_ORDER must contain integers only, got: {value!r}"]) from exc
    return order[0], order[1]


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
    arima_series_type: str
    use_auto_arima: bool
    auto_arima_max_p: int
    auto_arima_max_q: int
    auto_arima_max_d: int
    direction_threshold: float
    train_window: int
    refit_interval_minutes: int
    use_garch: bool
    garch_order: Tuple[int, int]
    garch_mean: str
    garch_dist: str
    garch_min_train_points: int
    garch_vol_scale: float
    garch_failure_mode: str
    aggregation_mode: str
    aggregation_min_snr: float
    garch_extreme_vol_action: str
    garch_vol_weight: float
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
    live_poll_interval_seconds: float
    live_kline_limit: int
    live_max_retries: int
    live_retry_backoff: float
    live_max_consecutive_errors: int
    live_error_retry_delay_seconds: float
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
            arima_series_type=os.getenv("ARIMA_SERIES_TYPE", "log_return").strip().lower(),
            use_auto_arima=_parse_bool(os.getenv("USE_AUTO_ARIMA"), False),
            auto_arima_max_p=_parse_int("AUTO_ARIMA_MAX_P", os.getenv("AUTO_ARIMA_MAX_P"), 5),
            auto_arima_max_q=_parse_int("AUTO_ARIMA_MAX_Q", os.getenv("AUTO_ARIMA_MAX_Q"), 5),
            auto_arima_max_d=_parse_int("AUTO_ARIMA_MAX_D", os.getenv("AUTO_ARIMA_MAX_D"), 2),
            direction_threshold=_parse_float(
                "DIRECTION_THRESHOLD", os.getenv("DIRECTION_THRESHOLD"), 0.0
            ),
            train_window=_parse_int("TRAIN_WINDOW", os.getenv("TRAIN_WINDOW"), 1440),
            refit_interval_minutes=_parse_int(
                "REFIT_INTERVAL_MINUTES", os.getenv("REFIT_INTERVAL_MINUTES"), 5
            ),
            use_garch=_parse_bool(os.getenv("USE_GARCH"), True),
            garch_order=_parse_garch_order(os.getenv("GARCH_ORDER")),
            garch_mean=os.getenv("GARCH_MEAN", "constant").strip().lower(),
            garch_dist=os.getenv("GARCH_DIST", "normal").strip().lower(),
            garch_min_train_points=_parse_int(
                "GARCH_MIN_TRAIN_POINTS", os.getenv("GARCH_MIN_TRAIN_POINTS"), 100
            ),
            garch_vol_scale=_parse_float("GARCH_VOL_SCALE", os.getenv("GARCH_VOL_SCALE"), 1.0),
            garch_failure_mode=os.getenv("GARCH_FAILURE_MODE", "hold").strip().lower(),
            aggregation_mode=os.getenv("AGGREGATION_MODE", "volatility_adjusted_arima")
            .strip()
            .lower(),
            aggregation_min_snr=_parse_float(
                "AGGREGATION_MIN_SNR", os.getenv("AGGREGATION_MIN_SNR"), 0.8
            ),
            garch_extreme_vol_action=os.getenv("GARCH_EXTREME_VOL_ACTION", "hold")
            .strip()
            .lower(),
            garch_vol_weight=_parse_float("GARCH_VOL_WEIGHT", os.getenv("GARCH_VOL_WEIGHT"), 0.35),
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
            live_poll_interval_seconds=_parse_float(
                "LIVE_POLL_INTERVAL_SECONDS", os.getenv("LIVE_POLL_INTERVAL_SECONDS"), 10.0
            ),
            live_kline_limit=_parse_int("LIVE_KLINE_LIMIT", os.getenv("LIVE_KLINE_LIMIT"), 2),
            live_max_retries=_parse_int("LIVE_MAX_RETRIES", os.getenv("LIVE_MAX_RETRIES"), 5),
            live_retry_backoff=_parse_float(
                "LIVE_RETRY_BACKOFF", os.getenv("LIVE_RETRY_BACKOFF"), 1.0
            ),
            live_max_consecutive_errors=_parse_int(
                "LIVE_MAX_CONSECUTIVE_ERRORS", os.getenv("LIVE_MAX_CONSECUTIVE_ERRORS"), 10
            ),
            live_error_retry_delay_seconds=_parse_float(
                "LIVE_ERROR_RETRY_DELAY_SECONDS",
                os.getenv("LIVE_ERROR_RETRY_DELAY_SECONDS"),
                5.0,
            ),
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

        if self.arima_series_type not in ALLOWED_ARIMA_SERIES_TYPES:
            errors.append(
                f"ARIMA_SERIES_TYPE must be one of {sorted(ALLOWED_ARIMA_SERIES_TYPES)}, "
                f"got: {self.arima_series_type!r}"
            )

        if self.auto_arima_max_p < 0 or self.auto_arima_max_q < 0 or self.auto_arima_max_d < 0:
            errors.append(
                "AUTO_ARIMA_MAX_P, AUTO_ARIMA_MAX_Q and AUTO_ARIMA_MAX_D must be non-negative"
            )

        if self.direction_threshold < 0:
            errors.append(
                f"DIRECTION_THRESHOLD must be non-negative, got: {self.direction_threshold}"
            )

        if self.train_window < self.prediction_minutes:
            errors.append(
                "TRAIN_WINDOW must be greater than or equal to PREDICTION_MINUTES "
                f"({self.train_window} < {self.prediction_minutes})"
            )
        if self.train_window < 60:
            errors.append(f"TRAIN_WINDOW must be at least 60, got: {self.train_window}")

        if self.refit_interval_minutes < 1:
            errors.append(f"REFIT_INTERVAL_MINUTES must be at least 1, got: {self.refit_interval_minutes}")

        p_garch, q_garch = self.garch_order
        if min(p_garch, q_garch) < 0:
            errors.append(f"GARCH_ORDER values must be non-negative, got: {self.garch_order}")

        if self.garch_mean not in ALLOWED_GARCH_MEANS:
            errors.append(
                f"GARCH_MEAN must be one of {sorted(ALLOWED_GARCH_MEANS)}, got: {self.garch_mean!r}"
            )

        if self.garch_dist not in ALLOWED_GARCH_DISTS:
            errors.append(
                f"GARCH_DIST must be one of {sorted(ALLOWED_GARCH_DISTS)}, got: {self.garch_dist!r}"
            )

        if self.garch_min_train_points < 50:
            errors.append(
                f"GARCH_MIN_TRAIN_POINTS must be at least 50, got: {self.garch_min_train_points}"
            )

        if self.garch_vol_scale <= 0:
            errors.append(f"GARCH_VOL_SCALE must be positive, got: {self.garch_vol_scale}")

        if self.garch_failure_mode not in ALLOWED_GARCH_FAILURE_MODES:
            errors.append(
                f"GARCH_FAILURE_MODE must be one of {sorted(ALLOWED_GARCH_FAILURE_MODES)}, "
                f"got: {self.garch_failure_mode!r}"
            )

        if self.aggregation_mode not in ALLOWED_AGGREGATION_MODES:
            errors.append(
                f"AGGREGATION_MODE must be one of {sorted(ALLOWED_AGGREGATION_MODES)}, "
                f"got: {self.aggregation_mode!r}"
            )

        if self.aggregation_min_snr < 0:
            errors.append(
                f"AGGREGATION_MIN_SNR must be non-negative, got: {self.aggregation_min_snr}"
            )

        if self.garch_extreme_vol_action not in ALLOWED_GARCH_EXTREME_VOL_ACTIONS:
            errors.append(
                f"GARCH_EXTREME_VOL_ACTION must be one of "
                f"{sorted(ALLOWED_GARCH_EXTREME_VOL_ACTIONS)}, "
                f"got: {self.garch_extreme_vol_action!r}"
            )

        if not 0.0 <= self.garch_vol_weight <= 1.0:
            errors.append(
                f"GARCH_VOL_WEIGHT must be in [0.0, 1.0], got: {self.garch_vol_weight}"
            )

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

        if self.live_poll_interval_seconds < 1.0:
            errors.append(
                "LIVE_POLL_INTERVAL_SECONDS must be at least 1.0, "
                f"got: {self.live_poll_interval_seconds}"
            )

        if self.live_kline_limit < 1 or self.live_kline_limit > 1000:
            errors.append(
                "LIVE_KLINE_LIMIT must be between 1 and 1000, "
                f"got: {self.live_kline_limit}"
            )

        if self.live_max_retries < 1:
            errors.append(f"LIVE_MAX_RETRIES must be at least 1, got: {self.live_max_retries}")

        if self.live_retry_backoff <= 0:
            errors.append(
                f"LIVE_RETRY_BACKOFF must be positive, got: {self.live_retry_backoff}"
            )

        if self.live_max_consecutive_errors < 1:
            errors.append(
                "LIVE_MAX_CONSECUTIVE_ERRORS must be at least 1, "
                f"got: {self.live_max_consecutive_errors}"
            )

        if self.live_error_retry_delay_seconds <= 0:
            errors.append(
                "LIVE_ERROR_RETRY_DELAY_SECONDS must be positive, "
                f"got: {self.live_error_retry_delay_seconds}"
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
