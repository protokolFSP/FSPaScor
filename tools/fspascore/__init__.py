"""
fspascore

Audio-first scoring for FSP transcripts:
- Score only first N seconds
- SRT is used only as a guide to identify assistant time intervals (heuristic, no diarization)
- Text, pause, fluency are computed from audio via faster-whisper (de)
- Grammar via LanguageTool (de-DE) ignoring STYLE/PUNCTUATION/WHITESPACE/CASING
"""
