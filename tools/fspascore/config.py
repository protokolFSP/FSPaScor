from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class ScoringConfig:
    """
    Tunables for scoring.

    Notes:
    - language_quality uses a soft log1p mapping from grammar_per_100w -> 0..100
    - clarity/fluency use soft log1p penalties (not too harsh)
    - turn-aware long pause uses "seconds of long pauses per minute" for smoothness
    """

    # LanguageTool
    lt_language: str = "de-DE"
    lt_ignore_category_ids: FrozenSet[str] = frozenset(
        {"STYLE", "PUNCTUATION", "WHITESPACE", "CASING"}
    )
    lt_ignore_issue_types: FrozenSet[str] = frozenset(
        {
            "style",
            "whitespace",
            "typographical",
        }
    )

    # Heuristics
    assistant_question_starters: Tuple[str, ...] = (
        "wie ",
        "was ",
        "wann ",
        "wo ",
        "warum ",
        "wieso ",
        "welche ",
        "welcher ",
        "welches ",
        "wieviel ",
        "wie viel ",
        "wie lange ",
        "seit wann ",
    )
    assistant_phrases: Tuple[str, ...] = (
        "können sie",
        "könnten sie",
        "haben sie",
        "nehmen sie",
        "ist ihnen",
        "sind sie",
        "dürfen sie",
        "würden sie",
        "möchten sie",
        "darf ich",
        "kann ich",
        "erzählen sie",
        "beschreiben sie",
        "zeigen sie",
        "haben sie schon",
        "haben sie allergien",
        "haben sie schmerzen",
        "seit wann",
        "wie lange",
        "gibt es",
        "haben sie jemals",
        "haben sie in letzter zeit",
        "haben sie fieber",
    )
    patient_markers: Tuple[str, ...] = (
        "herr doktor",
        "frau doktor",
        "doktor,",
        "doktor ",
    )

    # Text quality
    filler_tokens: Tuple[str, ...] = (
        "äh",
        "ähm",
        "hm",
        "hmm",
        "also",
        "naja",
        "eigentlich",
        "quasi",
        "sozusagen",
        "halt",
        "irgendwie",
    )

    # Pause metrics
    long_pause_threshold_sec: float = 2.0

    # Score mappings
    language_quality_alpha: float = 0.22  # smaller => less harsh

    clarity_w_filler: float = 11.0
    clarity_w_repeat: float = 10.0
    clarity_w_weird: float = 7.0

    fluency_w_filler: float = 7.0
    fluency_w_long_pause: float = 10.0
