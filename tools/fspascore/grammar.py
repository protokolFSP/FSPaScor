from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import language_tool_python

from .config import ScoringConfig


@dataclass(frozen=True)
class GrammarResult:
    error_count: int
    grammar_per_100w: float
    ignored_count: int


def _word_count_simple(text: str) -> int:
    import re

    words = re.findall(r"[A-Za-zÄÖÜäöüß]+(?:[-'][A-Za-zÄÖÜäöüß]+)?", text, flags=re.UNICODE)
    return len(words)


def _is_ignored_match(match, cfg: ScoringConfig) -> bool:
    try:
        cat_id = (match.rule.category.id or "").upper()
    except Exception:
        cat_id = ""
    try:
        issue_type = (match.rule.issueType or "").lower()
    except Exception:
        issue_type = ""

    if cat_id in cfg.lt_ignore_category_ids:
        return True
    if issue_type in cfg.lt_ignore_issue_types:
        return True
    return False


class LanguageToolClient:
    def __init__(self, cfg: ScoringConfig):
        self.cfg = cfg
        self._tool: Optional[language_tool_python.LanguageTool] = None

    def __enter__(self) -> "LanguageToolClient":
        self._tool = language_tool_python.LanguageTool(self.cfg.lt_language)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tool is not None:
            try:
                self._tool.close()
            finally:
                self._tool = None

    def check(self, text: str) -> List:
        if self._tool is None:
            raise RuntimeError("LanguageToolClient not initialized")
        return self._tool.check(text)


def compute_grammar_metrics(text: str, cfg: ScoringConfig, lt: LanguageToolClient) -> GrammarResult:
    wc = _word_count_simple(text)
    if wc <= 0:
        return GrammarResult(error_count=0, grammar_per_100w=0.0, ignored_count=0)

    matches = lt.check(text)
    ignored = 0
    kept = 0
    for m in matches:
        if _is_ignored_match(m, cfg):
            ignored += 1
        else:
            kept += 1

    return GrammarResult(error_count=kept, grammar_per_100w=(kept / wc) * 100.0, ignored_count=ignored)
