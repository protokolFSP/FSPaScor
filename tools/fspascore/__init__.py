"""
fspascore

Scoring pipeline for FSP transcripts (SRT):
- Evaluate only first N seconds (default 1120)
- Score only assistant lines via heuristics (question-like)
- Grammar via LanguageTool (de-DE) with ignored categories
- Turn-aware pause metrics (assistant-assistant only; patient speech is not counted as assistant silence)
"""
