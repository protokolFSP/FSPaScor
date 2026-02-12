from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import srt


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def load_srt_segments(path: Path) -> List[Segment]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    segments: List[Segment] = []
    for cue in srt.parse(raw):
        start = cue.start.total_seconds()
        end = cue.end.total_seconds()
        if end <= start:
            continue
        text = _normalize_text(cue.content)
        if not text:
            continue
        segments.append(Segment(start=float(start), end=float(end), text=text))
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def clip_segments(segments: List[Segment], start_sec: float, end_sec: float) -> List[Segment]:
    clipped: List[Segment] = []
    for s in segments:
        if s.start >= end_sec:
            break
        if s.end <= start_sec:
            continue
        clipped.append(Segment(start=max(s.start, start_sec), end=min(s.end, end_sec), text=s.text))
    return clipped
