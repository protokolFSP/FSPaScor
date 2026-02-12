from __future__ import annotations

import re
from typing import Iterable

from .config import ScoringConfig


_SPEAKER_LABEL_RE = re.compile(
    r"^\s*[-–—]?\s*(arzt|ärztin|doktor|dr\.?|assistant|assistent|sprech(er|erin))\s*:\s*",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    t = _SPEAKER_LABEL_RE.sub("", text)
    t = " ".join(t.split()).strip().lower()
    return t


def _has_any_prefix(text: str, prefixes: Iterable[str]) -> bool:
    return any(text.startswith(p) for p in prefixes)


def _has_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(p in text for p in phrases)


def is_assistant_line(text: str, cfg: ScoringConfig) -> bool:
    """
    Heuristic speaker labeling:
    - Assistant lines are question-like (contains '?' OR starts with German interrogatives OR contains common doctor question phrases)
    - Patient lines are excluded; we also try to avoid common patient markers (Herr Doktor, etc.)
    """
    raw_lower = text.lower()
    if any(m in raw_lower for m in cfg.patient_markers):
        return False

    t = _norm(text)

    if "?" in text:
        # still guard against obvious patient address patterns
        return True

    if _has_any_prefix(t, cfg.assistant_question_starters):
        return True

    if _has_any_phrase(t, cfg.assistant_phrases):
        return True

    # Extra: leading inversion like "Haben Sie..." without '?'
    if re.match(r"^(haben|können|könnten|nehmen|sind|dürfen|würden|möchten)\s+sie\b", t):
        return True

    return False
