"""Tests for the live application entry point."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.app import build_parser, main, resolve_dry_run, run_live


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.mode == "live"
    assert args.dry_run is False
    assert args.no_dry_run is False
    assert args.once is False


def test_resolve_dry_run_prefers_cli_flags() -> None:
    args = argparse.Namespace(dry_run=True, no_dry_run=False)
    assert resolve_dry_run(args, settings_dry_run=False) is True

    args = argparse.Namespace(dry_run=False, no_dry_run=True)
    assert resolve_dry_run(args, settings_dry_run=True) is False

    args = argparse.Namespace(dry_run=False, no_dry_run=False)
    assert resolve_dry_run(args, settings_dry_run=True) is True


def test_resolve_dry_run_rejects_conflicting_flags() -> None:
    args = argparse.Namespace(dry_run=True, no_dry_run=True)

    with pytest.raises(ValueError, match="both --dry-run and --no-dry-run"):
        resolve_dry_run(args, settings_dry_run=False)


def test_main_rejects_unsupported_mode() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--mode", "backtest"])

    assert exc_info.value.code == 2


def test_run_live_once_dry_run(monkeypatch) -> None:
    from src.utils.config import Settings

    base = Settings.from_environ()
    settings = Settings(
        symbol="BTCUSDT",
        interval="1m",
        prediction_minutes=10,
        arima_order=(1, 0, 1),
        arima_series_type="log_return",
        use_auto_arima=False,
        auto_arima_max_p=5,
        auto_arima_max_q=5,
        auto_arima_max_d=2,
        direction_threshold=0.0,
        train_window=1440,
        refit_interval_minutes=5,
        use_garch=False,
        garch_order=(1, 1),
        garch_mean="constant",
        garch_dist="normal",
        garch_min_train_points=100,
        garch_vol_scale=1.0,
        garch_failure_mode="hold",
        aggregation_mode="volatility_adjusted_arima",
        aggregation_min_snr=0.8,
        garch_extreme_vol_action="hold",
        garch_vol_weight=0.35,
        confidence_threshold=0.70,
        signal_cooldown_minutes=10,
        max_spread_bps=50.0,
        binance_market="spot",
        binance_api_key=None,
        binance_api_secret=None,
        binance_testnet=False,
        telegram_bot_token=None,
        telegram_chat_id=None,
        dry_run=True,
        log_level="INFO",
        live_poll_interval_seconds=10.0,
        live_kline_limit=2,
        live_max_retries=5,
        live_retry_backoff=1.0,
        live_max_consecutive_errors=10,
        live_error_retry_delay_seconds=5.0,
        data_dir=base.data_dir,
        logs_dir=base.logs_dir,
        project_root=base.project_root,
    )

    cycle_result = MagicMock()
    cycle_result.signal.direction = "HOLD"
    cycle_result.signal.confidence = 0.0
    cycle_result.refit_performed = False
    cycle_result.telegram_sent = False

    runner = MagicMock()
    runner.run_cycle.return_value = cycle_result

    monkeypatch.setattr("src.app.load_settings", lambda **_: settings)
    monkeypatch.setattr("src.app.LiveTradingRunner", lambda *a, **k: runner)
    monkeypatch.setattr("src.app.RestMarketDataSource", lambda **_: MagicMock())
    monkeypatch.setattr("src.app.MarketDataStorage", lambda _: MagicMock())
    monkeypatch.setattr("src.app.setup_logging", lambda *_a, **_k: None)

    args = build_parser().parse_args(["--mode", "live", "--dry-run", "--once", "--no-health-check"])
    exit_code = run_live(args)

    assert exit_code == 0
    runner.run_cycle.assert_called_once()
    runner.send_startup_health_check.assert_not_called()
