from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .config import ScoringConfig
from .srt_io import Segment


@dataclass(frozen=True)
class PauseMetrics:
    assistant_silence_total_sec: float
    assistant_long_silence_total_sec: float
    long_silence_sec_per_min: float
    silence_gaps: Tuple[float, ...]


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _interval_coverage(interval: Tuple[float, float], cover: List[Tuple[float, float]]) -> float:
    a, b = interval
    if b <= a:
        return 0.0

    clipped: List[Tuple[float, float]] = []
    for s, e in cover:
        if e <= a:
            continue
        if s >= b:
            break
        clipped.append((max(s, a), min(e, b)))

    merged = _merge_intervals(clipped)
    return sum(e - s for s, e in merged)


def compute_turn_aware_pause_metrics(
    assistant_segments: List[Segment],
    patient_segments: List[Segment],
    max_seconds: float,
    cfg: ScoringConfig,
) -> PauseMetrics:
    if len(assistant_segments) < 2:
        minutes = max(max_seconds / 60.0, 1e-9)
        return PauseMetrics(0.0, 0.0, 0.0, tuple())

    patient_intervals = _merge_intervals([(s.start, s.end) for s in patient_segments])

    gaps: List[float] = []
    total = 0.0
    total_long = 0.0

    for a, b in zip(assistant_segments, assistant_segments[1:]):
        gap_start = max(0.0, min(a.end, max_seconds))
        gap_end = max(0.0, min(b.start, max_seconds))
        if gap_end <= gap_start:
            continue

        covered = _interval_coverage((gap_start, gap_end), patient_intervals)
        silence = max(0.0, (gap_end - gap_start) - covered)

        gaps.append(silence)
        total += silence
        if silence >= cfg.long_pause_threshold_sec:
            total_long += silence

    minutes = max(max_seconds / 60.0, 1e-9)
    return PauseMetrics(
        assistant_silence_total_sec=total,
        assistant_long_silence_total_sec=total_long,
        long_silence_sec_per_min=total_long / minutes,
        silence_gaps=tuple(gaps),
    )
