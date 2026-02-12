from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .config import ScoringConfig


Role = Literal["assistant", "patient"]

_SPEAKER_LABEL_RE = re.compile(
    r"^\s*[-–—]?\s*(arzt|ärztin|doktor|dr\.?|assistant|assistent|sprech(er|erin))\s*:\s*",
    re.IGNORECASE,
)

_SECOND_PERSON_RE = re.compile(r"\b(sie|ihnen|ihr|ihre|ihrer|ihrem|ihren)\b", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(
    r"\b(ich|mir|mich|mein|meine|meiner|meinem|meinen|wir|uns|unser|unsere|unserer|unserem|unseren)\b",
    re.IGNORECASE,
)

_MODAL_SIE_LEAD_RE = re.compile(
    r"^(haben|können|koennen|könnten|koennten|nehmen|sind|dürfen|duerfen|würden|wuerden|möchten|moechten)\s+sie\b",
    re.IGNORECASE,
)

_PATIENT_CUE_RE = re.compile(
    r"\b(bei mir|ich habe|mir ist|es tut weh|schmerzen|seit|gestern|heute|vorhin|ungefähr|ungefaehr|immer|manchmal)\b",
    re.IGNORECASE,
)

_PATIENT_ANSWER_START_RE = re.compile(
    r"^\s*(ja|nein|genau|okay|also|naja|hm|äh|ähm)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    t = _SPEAKER_LABEL_RE.sub("", text)
    return " ".join(t.split()).strip()


def _has_any_prefix(text: str, prefixes: Iterable[str]) -> bool:
    return any(text.startswith(p) for p in prefixes)


def _has_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(p in text for p in phrases)


def label_role(text: str, cfg: ScoringConfig) -> Role:
    """
    Conservative role labeling:
    - assistant: question-like doctor prompts
    - patient: first-person / symptom narration / answer starters dominate
    Patient overrides assistant when signals conflict (prevents pause inflation).
    """
    raw = (text or "").strip()
    if not raw:
        return "patient"

    raw_lower = raw.lower()
    if any(m in raw_lower for m in cfg.patient_markers):
        return "patient"

    t = _norm(raw)
    t_lower = t.lower()

    has_q = "?" in raw
    has_2p = bool(_SECOND_PERSON_RE.search(t))
    has_1p = bool(_FIRST_PERSON_RE.search(t))
    starts_q = _has_any_prefix(t_lower, cfg.assistant_question_starters)
    modal_sie = bool(_MODAL_SIE_LEAD_RE.match(t_lower))
    has_doc_phrase = _has_any_phrase(t_lower, cfg.assistant_phrases)

    patient_cue = bool(_PATIENT_CUE_RE.search(t))
    patient_answer_start = bool(_PATIENT_ANSWER_START_RE.match(t))

    assistant_score = 0
    if starts_q:
        assistant_score += 3
    if modal_sie:
        assistant_score += 3
    if has_doc_phrase:
        assistant_score += 2
    if has_q:
        assistant_score += 1
    if has_2p:
        assistant_score += 1

    patient_score = 0
    if patient_answer_start:
        patient_score += 2
    if patient_cue:
        patient_score += 2
    if has_1p:
        patient_score += 2
    if has_q:
        patient_score += 1
    if has_doc_phrase or starts_q or modal_sie:
        patient_score -= 2  # doctor-ish structure reduces patient likelihood

    # Patient-first override when close/ambiguous
    if patient_score >= 3 and patient_score >= assistant_score - 1:
        return "patient"

    return "assistant" if assistant_score >= 4 and assistant_score > patient_score else "patient"


def is_assistant_line(text: str, cfg: ScoringConfig) -> bool:
    return label_role(text, cfg) == "assistant"
