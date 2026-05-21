"""Feature engineering from 1-minute klines and order book snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.kline_schema import KLINE_COLUMNS, interval_to_ms
from src.data.order_book_schema import ORDER_BOOK_COLUMNS

KLINE_FEATURE_COLUMNS: list[str] = [
    "log_return",
    "price_diff",
    "rolling_volatility",
    "volume_change_rate",
]

ORDERBOOK_FEATURE_COLUMNS: list[str] = [
    "spread",
    "spread_bps",
    "book_imbalance",
    "mid_price",
]

LABEL_COLUMNS: list[str] = [
    "future_close",
    "future_log_return",
    "label_direction",
]

FEATURE_COLUMNS: list[str] = KLINE_FEATURE_COLUMNS + ORDERBOOK_FEATURE_COLUMNS

LABEL_UP = "UP"
LABEL_DOWN = "DOWN"
LABEL_FLAT = "FLAT"


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters for feature and label generation."""

    interval: str = "1m"
    prediction_minutes: int = 10
    volatility_window: int = 20
    label_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.prediction_minutes < 1:
            raise ValueError(f"prediction_minutes must be >= 1, got {self.prediction_minutes}")
        if self.volatility_window < 2:
            raise ValueError(f"volatility_window must be >= 2, got {self.volatility_window}")
        if self.label_threshold < 0:
            raise ValueError(f"label_threshold must be non-negative, got {self.label_threshold}")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _prepare_klines(klines: pd.DataFrame) -> pd.DataFrame:
    if klines.empty:
        return pd.DataFrame(columns=list(KLINE_COLUMNS))

    _require_columns(klines, KLINE_COLUMNS, name="klines")
    frame = klines.copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    for column in KLINE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def _prepare_orderbook(orderbook: pd.DataFrame) -> pd.DataFrame:
    if orderbook.empty:
        return pd.DataFrame(columns=list(ORDER_BOOK_COLUMNS))

    _require_columns(orderbook, ORDER_BOOK_COLUMNS, name="orderbook")
    frame = orderbook.copy()
    frame["timestamp"] = frame["timestamp"].astype("int64")
    for column in ORDER_BOOK_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def compute_kline_features(
    klines: pd.DataFrame,
    *,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Compute kline-derived features using only current and past bars."""
    frame = _prepare_klines(klines)
    if frame.empty:
        for column in KLINE_FEATURE_COLUMNS:
            frame[column] = pd.Series(dtype="float64")
        return frame

    close = frame["close"]
    volume = frame["volume"]

    frame["log_return"] = np.log(close / close.shift(1))
    frame["price_diff"] = close.diff()
    frame["rolling_volatility"] = frame["log_return"].rolling(
        window=volatility_window,
        min_periods=volatility_window,
    ).std()
    previous_volume = volume.shift(1)
    frame["volume_change_rate"] = np.where(
        previous_volume > 0,
        (volume - previous_volume) / previous_volume,
        np.nan,
    )

    return frame


def merge_orderbook_features(klines: pd.DataFrame, orderbook: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest known order book snapshot at or before each kline timestamp."""
    kline_frame = _prepare_klines(klines)
    if kline_frame.empty:
        merged = kline_frame.copy()
        for column in ORDERBOOK_FEATURE_COLUMNS:
            merged[column] = pd.Series(dtype="float64")
        return merged

    book_frame = _prepare_orderbook(orderbook)
    if book_frame.empty:
        merged = kline_frame.copy()
        for column in ORDERBOOK_FEATURE_COLUMNS:
            merged[column] = np.nan
        return merged

    merged = pd.merge_asof(
        kline_frame,
        book_frame[
            [
                "timestamp",
                "spread",
                "book_imbalance",
                "mid_price",
            ]
        ],
        on="timestamp",
        direction="backward",
    )
    merged["spread_bps"] = np.where(
        merged["mid_price"] > 0,
        merged["spread"] / merged["mid_price"] * 10_000.0,
        np.nan,
    )
    return merged


def compute_backtest_labels(
    klines: pd.DataFrame,
    *,
    interval: str = "1m",
    prediction_minutes: int = 10,
    label_threshold: float = 0.0,
) -> pd.DataFrame:
    """Compute future direction labels for backtesting only (uses future prices)."""
    frame = _prepare_klines(klines)
    if frame.empty:
        labeled = frame.copy()
        for column in LABEL_COLUMNS:
            labeled[column] = pd.Series(dtype="object" if column == "label_direction" else "float64")
        return labeled

    horizon_ms = prediction_minutes * interval_to_ms(interval)
    lookup = frame[["timestamp", "close"]].rename(
        columns={"timestamp": "target_timestamp", "close": "future_close"}
    )
    labeled = frame.copy()
    labeled["target_timestamp"] = labeled["timestamp"] + horizon_ms
    labeled = labeled.merge(lookup, on="target_timestamp", how="left")
    labeled = labeled.drop(columns=["target_timestamp"])

    labeled["future_log_return"] = np.log(labeled["future_close"] / labeled["close"])
    labeled["label_direction"] = LABEL_FLAT
    labeled.loc[labeled["future_log_return"] > label_threshold, "label_direction"] = LABEL_UP
    labeled.loc[labeled["future_log_return"] < -label_threshold, "label_direction"] = LABEL_DOWN
    labeled.loc[labeled["future_close"].isna(), "label_direction"] = np.nan

    return labeled


def build_feature_frame(
    klines: pd.DataFrame,
    orderbook: Optional[pd.DataFrame] = None,
    *,
    config: Optional[FeatureConfig] = None,
    include_labels: bool = False,
) -> pd.DataFrame:
    """Build the full feature matrix, optionally with backtest labels."""
    cfg = config or FeatureConfig()
    with_kline_features = compute_kline_features(
        klines,
        volatility_window=cfg.volatility_window,
    )

    if orderbook is not None:
        frame = merge_orderbook_features(with_kline_features, orderbook)
    else:
        frame = with_kline_features.copy()
        for column in ORDERBOOK_FEATURE_COLUMNS:
            frame[column] = np.nan

    if include_labels:
        labels = compute_backtest_labels(
            klines,
            interval=cfg.interval,
            prediction_minutes=cfg.prediction_minutes,
            label_threshold=cfg.label_threshold,
        )
        for column in LABEL_COLUMNS:
            frame[column] = labels[column].values

    return frame.reset_index(drop=True)


def feature_columns(include_orderbook: bool = True, include_labels: bool = False) -> list[str]:
    """Return feature column names for model input or export."""
    columns = list(KLINE_FEATURE_COLUMNS)
    if include_orderbook:
        columns.extend(ORDERBOOK_FEATURE_COLUMNS)
    if include_labels:
        columns.extend(LABEL_COLUMNS)
    return columns


def model_input_columns(include_orderbook: bool = True) -> list[str]:
    """Columns safe for live prediction (never includes backtest labels)."""
    return feature_columns(include_orderbook=include_orderbook, include_labels=False)


def assert_no_feature_leakage(
    klines: pd.DataFrame,
    orderbook: Optional[pd.DataFrame] = None,
    *,
    config: Optional[FeatureConfig] = None,
    feature_columns_to_check: Optional[Iterable[str]] = None,
) -> None:
    """Verify feature values at each row depend only on data up to that timestamp."""
    cfg = config or FeatureConfig()
    columns = list(feature_columns_to_check or model_input_columns(include_orderbook=orderbook is not None))
    full_frame = build_feature_frame(klines, orderbook, config=cfg, include_labels=False)

    prepared_klines = _prepare_klines(klines)
    prepared_orderbook = _prepare_orderbook(orderbook) if orderbook is not None else None

    for idx in range(len(prepared_klines)):
        cutoff = prepared_klines.loc[idx, "timestamp"]
        truncated_klines = prepared_klines.iloc[: idx + 1]
        truncated_orderbook = None
        if prepared_orderbook is not None and not prepared_orderbook.empty:
            truncated_orderbook = prepared_orderbook[prepared_orderbook["timestamp"] <= cutoff]

        partial = build_feature_frame(
            truncated_klines,
            truncated_orderbook,
            config=cfg,
            include_labels=False,
        )
        if partial.empty:
            continue

        for column in columns:
            if column not in partial.columns:
                continue
            expected = partial.iloc[-1][column]
            actual = full_frame.iloc[idx][column]
            if pd.isna(expected) and pd.isna(actual):
                continue
            if not np.isclose(expected, actual, rtol=1e-9, atol=1e-12, equal_nan=True):
                raise AssertionError(
                    f"Feature leakage detected at index {idx}, timestamp {cutoff}, "
                    f"column {column!r}: expected {expected!r}, got {actual!r}"
                )
