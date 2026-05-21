"""Duplicate removal and time-continuity checks for K-line data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from src.data.kline_schema import KLINE_COLUMNS


@dataclass(frozen=True)
class QualityReport:
    """Summary of K-line data quality after download."""

    row_count: int
    duplicate_count: int
    gap_count: int
    gaps: List[Tuple[int, int, int]]  # (prev_ts, next_ts, missing_bars)
    start_timestamp: Optional[int]
    end_timestamp: Optional[int]
    is_continuous: bool

    def __str__(self) -> str:
        if self.row_count == 0:
            return "QualityReport: empty dataset"
        gap_detail = f", gaps={self.gap_count}" if self.gap_count else ", continuous"
        dup_detail = f", duplicates_removed={self.duplicate_count}" if self.duplicate_count else ""
        return (
            f"QualityReport: rows={self.row_count}{dup_detail}{gap_detail}, "
            f"range=[{self.start_timestamp}, {self.end_timestamp}]"
        )


def dedupe_by_timestamp(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate timestamps, keeping the last row for each timestamp."""
    if frame.empty:
        return frame.copy(), 0
    duplicate_mask = frame.duplicated(subset=["timestamp"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        return frame.copy(), 0
    # Keep last occurrence so later pages win on overlap
    deduped = frame.drop_duplicates(subset=["timestamp"], keep="last")
    removed = len(frame) - len(deduped)
    return deduped.sort_values("timestamp").reset_index(drop=True), removed


def find_time_gaps(
    timestamps: pd.Series, interval_ms: int
) -> List[Tuple[int, int, int]]:
    """Return gaps where consecutive candles are more than one interval apart."""
    if len(timestamps) < 2:
        return []
    gaps: List[Tuple[int, int, int]] = []
    values = timestamps.astype("int64").tolist()
    for prev_ts, next_ts in zip(values, values[1:]):
        delta = next_ts - prev_ts
        if delta > interval_ms:
            missing = int(delta // interval_ms) - 1
            gaps.append((prev_ts, next_ts, missing))
    return gaps


def assess_quality(frame: pd.DataFrame, interval_ms: int) -> QualityReport:
    """Build a quality report for a K-line DataFrame."""
    if frame.empty:
        return QualityReport(
            row_count=0,
            duplicate_count=0,
            gap_count=0,
            gaps=[],
            start_timestamp=None,
            end_timestamp=None,
            is_continuous=True,
        )
    gaps = find_time_gaps(frame["timestamp"], interval_ms)
    return QualityReport(
        row_count=len(frame),
        duplicate_count=0,
        gap_count=len(gaps),
        gaps=gaps,
        start_timestamp=int(frame["timestamp"].iloc[0]),
        end_timestamp=int(frame["timestamp"].iloc[-1]),
        is_continuous=len(gaps) == 0,
    )


def prepare_klines(frame: pd.DataFrame, interval_ms: int) -> tuple[pd.DataFrame, QualityReport]:
    """Deduplicate, sort, and assess continuity."""
    working = frame[KLINE_COLUMNS].copy() if not frame.empty else pd.DataFrame(columns=KLINE_COLUMNS)
    deduped, duplicate_count = dedupe_by_timestamp(working)
    gaps = find_time_gaps(deduped["timestamp"], interval_ms) if not deduped.empty else []
    report = QualityReport(
        row_count=len(deduped),
        duplicate_count=duplicate_count,
        gap_count=len(gaps),
        gaps=gaps,
        start_timestamp=int(deduped["timestamp"].iloc[0]) if not deduped.empty else None,
        end_timestamp=int(deduped["timestamp"].iloc[-1]) if not deduped.empty else None,
        is_continuous=len(gaps) == 0,
    )
    return deduped, report
