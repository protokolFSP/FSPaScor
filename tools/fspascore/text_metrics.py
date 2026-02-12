from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

from .config import ScoringConfig

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:[-'][A-Za-zÄÖÜäöüß]+)?", re.UNICODE)
_WEIRD_TOKEN_RE = re.compile(r"(�|_|\d|[^\s]{30,}|(.)\2\2\2)", re.UNICODE)


@dataclass(frozen=True)
class TextMetrics:
    word_count: int
    filler_count: int
    repetition_count: int
    weird_token_count: int

    filler_per_100w: float
    repetition_per_100w: float
    weird_per_100w: float

    clarity_score: float
    fluency_score: float


def _tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _per_100(count: int, words: int) -> float:
    if words <= 0:
        return 0.0
    return (count / words) * 100.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _count_fillers(words: List[str], cfg: ScoringConfig) -> int:
    fillers = {t.lower() for t in cfg.filler_tokens}
    return sum(1 for w in words if w in fillers)


def _count_repetitions(words: List[str]) -> int:
    rep = 0
    window = 3
    for i in range(1, len(words)):
        start = max(0, i - window)
        if words[i] in words[start:i]:
            rep += 1
    return rep


def _count_weird_tokens(raw_text: str) -> int:
    return len(_WEIRD_TOKEN_RE.findall(raw_text))


def compute_text_metrics(assistant_text: str, cfg: ScoringConfig, long_silence_sec_per_min: float) -> TextMetrics:
    words = _tokenize_words(assistant_text)
    wc = len(words)

    filler = _count_fillers(words, cfg)
    rep = _count_repetitions(words)
    weird = _count_weird_tokens(assistant_text)

    filler100 = _per_100(filler, wc)
    rep100 = _per_100(rep, wc)
    weird100 = _per_100(weird, wc)

    clarity_pen = (
        cfg.clarity_w_filler * math.log1p(filler100)
        + cfg.clarity_w_repeat * math.log1p(rep100)
        + cfg.clarity_w_weird * math.log1p(weird100)
    )
    clarity = _clamp(100.0 - clarity_pen)

    flu_pen = (
        cfg.fluency_w_filler * math.log1p(filler100)
        + cfg.fluency_w_long_pause * math.log1p(max(0.0, long_silence_sec_per_min))
    )
    fluency = _clamp(100.0 - flu_pen)

    return TextMetrics(
        word_count=wc,
        filler_count=filler,
        repetition_count=rep,
        weird_token_count=weird,
        filler_per_100w=filler100,
        repetition_per_100w=rep100,
        weird_per_100w=weird100,
        clarity_score=clarity,
        fluency_score=fluency,
    )
