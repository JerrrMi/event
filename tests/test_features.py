"""Tests for feature engineering and backtest label generation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_UP,
    FeatureConfig,
    assert_no_feature_leakage,
    build_feature_frame,
    compute_backtest_labels,
    compute_kline_features,
    merge_orderbook_features,
    model_input_columns,
)


def _make_klines(
    closes: list[float],
    *,
    start_ts: int = 1_000_000,
    step_ms: int = 60_000,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    if volumes is None:
        volumes = [10.0 + index for index in range(len(closes))]

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


def _make_orderbook(
    timestamps: list[int],
    *,
    bid: float = 100.0,
    ask: float = 100.1,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
) -> pd.DataFrame:
    spread = ask - bid
    mid = (ask + bid) / 2.0
    imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "best_bid_price": bid,
            "best_bid_qty": bid_qty,
            "best_ask_price": ask,
            "best_ask_qty": ask_qty,
            "spread": spread,
            "mid_price": mid,
            "book_imbalance": imbalance,
        }
    )


def test_compute_kline_features_basic_values() -> None:
    klines = _make_klines([100.0, 110.0, 99.0], volumes=[10.0, 20.0, 5.0])
    features = compute_kline_features(klines, volatility_window=2)

    assert pd.isna(features.loc[0, "log_return"])
    assert features.loc[1, "log_return"] == pytest.approx(np.log(110.0 / 100.0))
    assert features.loc[2, "price_diff"] == pytest.approx(-11.0)
    assert features.loc[1, "volume_change_rate"] == pytest.approx(1.0)
    assert features.loc[2, "volume_change_rate"] == pytest.approx(-0.75)


def test_rolling_volatility_requires_full_window() -> None:
    klines = _make_klines([100.0, 101.0, 102.0, 103.0])
    features = compute_kline_features(klines, volatility_window=3)

    assert pd.isna(features.loc[1, "rolling_volatility"])
    assert pd.isna(features.loc[2, "rolling_volatility"])
    assert not pd.isna(features.loc[3, "rolling_volatility"])
    assert features.loc[3, "rolling_volatility"] >= 0.0


def test_volume_change_rate_nan_when_previous_volume_zero() -> None:
    klines = _make_klines([100.0, 101.0], volumes=[0.0, 10.0])
    features = compute_kline_features(klines)

    assert pd.isna(features.loc[1, "volume_change_rate"])


def test_merge_orderbook_uses_latest_past_snapshot_only() -> None:
    klines = _make_klines([100.0, 101.0, 102.0], start_ts=60_000, step_ms=60_000)
    orderbook = _make_orderbook(
        [30_000, 90_000, 150_000],
        bid=100.0,
        ask=100.2,
        bid_qty=3.0,
        ask_qty=1.0,
    )

    merged = merge_orderbook_features(klines, orderbook)

    assert merged.loc[0, "spread"] == pytest.approx(0.2)
    assert merged.loc[0, "book_imbalance"] == pytest.approx(0.5)
    assert merged.loc[1, "spread"] == pytest.approx(0.2)
    assert merged.loc[2, "spread"] == pytest.approx(0.2)


def test_future_orderbook_snapshot_does_not_affect_current_kline() -> None:
    klines = _make_klines([100.0], start_ts=60_000, step_ms=60_000)
    orderbook_before = _make_orderbook([30_000], bid=100.0, ask=100.1)
    orderbook_after = pd.concat(
        [
            _make_orderbook([30_000], bid=100.0, ask=100.1),
            _make_orderbook([120_000], bid=100.0, ask=100.9),
        ],
        ignore_index=True,
    )

    merged_before = merge_orderbook_features(klines, orderbook_before)
    merged_after = merge_orderbook_features(klines, orderbook_after)

    assert merged_before.loc[0, "spread"] == pytest.approx(0.1)
    assert merged_after.loc[0, "spread"] == pytest.approx(0.1)


def test_backtest_labels_align_by_timestamp_not_row_shift() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 120.0]
    klines = _make_klines(closes, start_ts=0, step_ms=60_000)
    klines = klines.drop(index=5).reset_index(drop=True)

    labeled = compute_backtest_labels(klines, prediction_minutes=10)

    first_ts = klines.loc[0, "timestamp"]
    expected_future = klines.loc[klines["timestamp"] == first_ts + 10 * 60_000, "close"]
    assert len(expected_future) == 1
    assert labeled.loc[0, "future_close"] == pytest.approx(expected_future.iloc[0])
    assert labeled.loc[0, "label_direction"] == LABEL_UP


def test_backtest_labels_last_horizon_rows_are_nan() -> None:
    klines = _make_klines([100.0 + index for index in range(15)])
    labeled = compute_backtest_labels(klines, prediction_minutes=10)

    assert labeled.iloc[-10:]["future_close"].isna().all()
    assert labeled.iloc[-10:]["label_direction"].isna().all()
    assert not pd.isna(labeled.iloc[0]["future_close"])


def test_backtest_label_direction_respects_threshold() -> None:
    klines = _make_klines([100.0] * 20)
    klines.loc[0, "close"] = 100.0
    klines.loc[10, "close"] = 100.2
    klines.loc[1, "close"] = 100.0
    klines.loc[11, "close"] = 99.7
    klines.loc[2, "close"] = 100.0
    klines.loc[12, "close"] = 100.05

    labeled = compute_backtest_labels(klines, prediction_minutes=10, label_threshold=0.001)

    assert labeled.loc[0, "label_direction"] == LABEL_UP
    assert labeled.loc[1, "label_direction"] == LABEL_DOWN
    assert labeled.loc[2, "label_direction"] == LABEL_FLAT


def test_build_feature_frame_without_orderbook_fills_nan() -> None:
    klines = _make_klines([100.0, 101.0, 102.0])
    frame = build_feature_frame(klines, include_labels=True)

    assert frame["spread"].isna().all()
    assert frame["book_imbalance"].isna().all()
    assert "future_close" in frame.columns
    assert "label_direction" in frame.columns


def test_model_input_columns_exclude_labels() -> None:
    assert "label_direction" not in model_input_columns()
    assert "future_log_return" not in model_input_columns()


def test_assert_no_feature_leakage_on_klines_only() -> None:
    klines = _make_klines([100.0 + index for index in range(30)])
    assert_no_feature_leakage(klines, config=FeatureConfig(volatility_window=5))


def test_assert_no_feature_leakage_with_orderbook() -> None:
    klines = _make_klines([100.0 + index for index in range(20)], start_ts=0)
    orderbook = _make_orderbook(
        [index * 45_000 for index in range(40)],
        bid=100.0,
        ask=100.5,
    )
    assert_no_feature_leakage(klines, orderbook, config=FeatureConfig(volatility_window=4))


def test_assert_no_feature_leakage_detects_forward_looking_feature() -> None:
    klines = _make_klines([100.0, 101.0, 102.0, 103.0])

    def leaky_compute(klines: pd.DataFrame, *, volatility_window: int = 20) -> pd.DataFrame:
        frame = compute_kline_features(klines, volatility_window=volatility_window)
        frame["log_return"] = np.log(frame["close"].shift(-1) / frame["close"])
        return frame

    with patch("src.features.engineering.compute_kline_features", leaky_compute):
        with pytest.raises(AssertionError, match="Feature leakage detected"):
            assert_no_feature_leakage(klines, feature_columns_to_check=["log_return"])


def test_empty_klines_return_empty_frame_with_columns() -> None:
    empty = pd.DataFrame(columns=_make_klines([]).columns)
    frame = build_feature_frame(empty, include_labels=True)

    assert frame.empty
    assert "log_return" in frame.columns
    assert "label_direction" in frame.columns


def test_unsorted_and_duplicate_timestamps_are_normalized() -> None:
    klines = _make_klines([100.0, 101.0, 102.0], start_ts=0)
    shuffled = pd.concat([klines.iloc[[2, 0]], klines.iloc[[1, 1]]], ignore_index=True)

    features = compute_kline_features(shuffled)

    assert features["timestamp"].is_monotonic_increasing
    assert len(features) == 3
    assert features.loc[2, "close"] == pytest.approx(102.0)


def test_invalid_feature_config() -> None:
    with pytest.raises(ValueError, match="prediction_minutes"):
        FeatureConfig(prediction_minutes=0)
    with pytest.raises(ValueError, match="volatility_window"):
        FeatureConfig(volatility_window=1)
    with pytest.raises(ValueError, match="label_threshold"):
        FeatureConfig(label_threshold=-0.1)


def test_missing_required_kline_columns_raises() -> None:
    bad = _make_klines([100.0]).drop(columns=["close"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute_kline_features(bad)
