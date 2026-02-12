from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .audio_transcribe import transcribe_audio_for_title
from .config import ScoringConfig
from .grammar import LanguageToolClient, compute_grammar_metrics
from .heuristics import label_role
from .pause import compute_turn_aware_pause_metrics
from .srt_io import Segment, clip_segments, load_srt_segments
from .text_metrics import compute_text_metrics


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _jsonify(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonify(x) for x in obj)
    if isinstance(obj, tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    return str(obj)


def language_quality_from_grammar(grammar_per_100w: float, cfg: ScoringConfig) -> float:
    score = 100.0 / (1.0 + cfg.language_quality_alpha * math.log1p(max(0.0, grammar_per_100w)))
    return _clamp(score)


def _list_srt_files(transcripts_dir: Path) -> List[Path]:
    return sorted([p for p in transcripts_dir.rglob("*.srt") if p.is_file()])


def _file_id_from_path(srt_path: Path) -> str:
    return srt_path.stem


def _load_existing_full(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 3, "generated_at_utc": None, "config": {}, "items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" not in data:
        data["items"] = []
    return data


def _existing_ids(full: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for it in full.get("items", []):
        fid = it.get("file_id")
        if isinstance(fid, str) and fid:
            out.add(fid)
    return out


def _assistant_patient_split_srt(segments: List[Segment], cfg: ScoringConfig) -> tuple[List[Segment], List[Segment]]:
    assistant: List[Segment] = []
    patient: List[Segment] = []
    for s in segments:
        role = label_role(s.text, cfg)
        (assistant if role == "assistant" else patient).append(s)
    return assistant, patient


def _concat_text(segments: List[Segment]) -> str:
    return " ".join(s.text for s in segments).strip()


def score_one(
    srt_path: Path,
    transcripts_root: Path,
    max_seconds: float,
    cfg: ScoringConfig,
    lt: LanguageToolClient,
) -> Dict[str, Any]:
    raw_segments = load_srt_segments(srt_path)
    srt_segments = clip_segments(raw_segments, 0.0, max_seconds)

    assistant_guide_srt, patient_srt = _assistant_patient_split_srt(srt_segments, cfg)

    whisper_model = os.getenv("WHISPER_MODEL", "medium")
    whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cache_dir = Path(".cache/fspascore/audio")

    audio = transcribe_audio_for_title(
        title=_file_id_from_path(srt_path),
        assistant_guide_segments=assistant_guide_srt,
        max_seconds=max_seconds,
        cache_dir=cache_dir,
        whisper_model=whisper_model,
        whisper_compute_type=whisper_compute_type,
    )

    if audio and audio.assistant_text and len(audio.assistant_text.split()) >= 20:
        assistant_segments = audio.assistant_segments
        patient_segments = audio.patient_segments
        assistant_text = audio.assistant_text
        transcript_source = "audio_whisper_srt_guided"
        audio_identifier = audio.audio_ref.identifier
        audio_filename = audio.audio_ref.filename
        audio_url = audio.audio_ref.download_url
    else:
        assistant_segments = assistant_guide_srt
        patient_segments = patient_srt
        assistant_text = _concat_text(assistant_guide_srt)
        transcript_source = "srt_fallback"
        audio_identifier = None
        audio_filename = None
        audio_url = None

    pause = compute_turn_aware_pause_metrics(
        assistant_segments=assistant_segments,
        patient_segments=patient_segments,
        max_seconds=max_seconds,
        cfg=cfg,
    )

    grammar = compute_grammar_metrics(assistant_text, cfg, lt)
    language_quality = language_quality_from_grammar(grammar.grammar_per_100w, cfg)

    textm = compute_text_metrics(
        assistant_text=assistant_text,
        cfg=cfg,
        long_silence_sec_per_min=pause.long_silence_sec_per_min,
    )

    overall = _clamp(0.70 * language_quality + 0.20 * textm.clarity_score + 0.10 * textm.fluency_score)

    rel_path = str(srt_path.relative_to(transcripts_root)).replace("\\", "/")
    file_id = _file_id_from_path(srt_path)

    return {
        "file_id": file_id,
        "transcript_path": rel_path,
        "max_seconds": round(max_seconds, 3),
        "processed_at_utc": _utc_now_iso(),
        "transcript_source": transcript_source,
        "audio_identifier": audio_identifier,
        "audio_filename": audio_filename,
        "audio_url": audio_url,
        "assistant_line_count": len(assistant_segments),
        "patient_line_count": len(patient_segments),
        "assistant_text_word_count": textm.word_count,
        "grammar_error_count": grammar.error_count,
        "grammar_ignored_count": grammar.ignored_count,
        "grammar_per_100w": round(grammar.grammar_per_100w, 6),
        "language_quality": round(language_quality, 6),
        "filler_per_100w": round(textm.filler_per_100w, 6),
        "repetition_per_100w": round(textm.repetition_per_100w, 6),
        "weird_token_per_100w": round(textm.weird_per_100w, 6),
        "clarity": round(textm.clarity_score, 6),
        "assistant_silence_total_sec": round(pause.assistant_silence_total_sec, 6),
        "assistant_long_silence_total_sec": round(pause.assistant_long_silence_total_sec, 6),
        "long_silence_sec_per_min": round(pause.long_silence_sec_per_min, 6),
        "fluency": round(textm.fluency_score, 6),
        "overall": round(overall, 6),
    }


def _write_scores_json(public_dir: Path, items: List[Dict[str, Any]]) -> None:
    minimal = [
        {
            "file_id": it["file_id"],
            "transcript_path": it["transcript_path"],
            "overall": it["overall"],
            "language_quality": it["language_quality"],
            "clarity": it["clarity"],
            "fluency": it["fluency"],
            "transcript_source": it.get("transcript_source"),
        }
        for it in items
    ]
    (public_dir / "scores_1120.json").write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_full_json(public_dir: Path, cfg: ScoringConfig, items: List[Dict[str, Any]]) -> None:
    payload = {
        "version": 3,
        "generated_at_utc": _utc_now_iso(),
        "config": _jsonify(asdict(cfg)),
        "items": _jsonify(items),
    }
    (public_dir / "scores_1120.full.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_metrics_csv(public_dir: Path, items: List[Dict[str, Any]]) -> None:
    out = public_dir / "metrics_1120.csv"
    fieldnames = [
        "file_id",
        "transcript_path",
        "max_seconds",
        "processed_at_utc",
        "transcript_source",
        "audio_identifier",
        "audio_filename",
        "assistant_line_count",
        "patient_line_count",
        "assistant_text_word_count",
        "grammar_error_count",
        "grammar_ignored_count",
        "grammar_per_100w",
        "language_quality",
        "filler_per_100w",
        "repetition_per_100w",
        "weird_token_per_100w",
        "clarity",
        "assistant_silence_total_sec",
        "assistant_long_silence_total_sec",
        "long_silence_sec_per_min",
        "fluency",
        "overall",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in items:
            w.writerow({k: it.get(k, "") for k in fieldnames})


def run_pipeline(transcripts_dir: Path, public_dir: Path, max_seconds: float, max_new_files: int) -> None:
    cfg = ScoringConfig()
    public_dir.mkdir(parents=True, exist_ok=True)

    full_path = public_dir / "scores_1120.full.json"
    existing_full = _load_existing_full(full_path)
    existing_items: List[Dict[str, Any]] = list(existing_full.get("items", []))
    done = _existing_ids(existing_full)

    srt_files = _list_srt_files(transcripts_dir)
    new_files = [p for p in srt_files if _file_id_from_path(p) not in done]
    new_files = sorted(new_files, key=lambda p: str(p).lower())
    if max_new_files > 0:
        new_files = new_files[:max_new_files]

    transcripts_root = transcripts_dir

    with LanguageToolClient(cfg) as lt:
        for p in new_files:
            existing_items.append(
                score_one(
                    srt_path=p,
                    transcripts_root=transcripts_root,
                    max_seconds=max_seconds,
                    cfg=cfg,
                    lt=lt,
                )
            )

    items_sorted = sorted(existing_items, key=lambda it: (it.get("transcript_path", ""), it.get("file_id", "")))
    _write_full_json(public_dir, cfg, items_sorted)
    _write_scores_json(public_dir, items_sorted)
    _write_metrics_csv(public_dir, items_sorted)
