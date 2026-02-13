from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .audio_transcribe import AudioNotFoundError, transcribe_audio_for_title
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


def _parse_iso(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def language_quality_from_grammar(grammar_per_100w: float, cfg: ScoringConfig) -> float:
    score = 100.0 / (1.0 + cfg.language_quality_alpha * math.log1p(max(0.0, grammar_per_100w)))
    return _clamp(score)


def _list_srt_files(transcripts_dir: Path) -> List[Path]:
    return sorted([p for p in transcripts_dir.rglob("*.srt") if p.is_file()])


def _file_id_from_path(srt_path: Path) -> str:
    return srt_path.stem


def _load_existing_full(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 6, "generated_at_utc": None, "config": {}, "items": [], "skipped": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("items", [])
    data.setdefault("skipped", [])
    return data


def _assistant_patient_split_srt(segments: List[Segment], cfg: ScoringConfig) -> Tuple[List[Segment], List[Segment]]:
    assistant: List[Segment] = []
    patient: List[Segment] = []
    for s in segments:
        role = label_role(s.text, cfg)
        (assistant if role == "assistant" else patient).append(s)
    return assistant, patient


def _concat_text(segments: List[Segment]) -> str:
    return " ".join(s.text for s in segments).strip()


def _audio_missing_policy() -> str:
    v = (os.getenv("AUDIO_MISSING_POLICY", "skip") or "skip").strip().lower()
    return v if v in {"skip", "fail", "srt_fallback"} else "skip"


def _reprocess_fallback_enabled() -> bool:
    return (os.getenv("REPROCESS_FALLBACK", "1") or "1").strip() == "1"


def _source_priority(src: Optional[str]) -> int:
    # audio > srt_fallback > unknown
    if src == "audio_whisper_srt_guided":
        return 2
    if src == "srt_fallback":
        return 1
    return 0


def _item_rank(item: Dict[str, Any]) -> Tuple[int, datetime]:
    return (_source_priority(item.get("transcript_source")), _parse_iso(item.get("processed_at_utc")))


def _dedupe_items(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Collapse duplicates by file_id:
    - prefer audio over srt_fallback
    - if same source, prefer latest processed_at_utc
    """
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        fid = it.get("file_id")
        if not isinstance(fid, str) or not fid:
            continue
        prev = out.get(fid)
        if prev is None or _item_rank(it) > _item_rank(prev):
            out[fid] = it
    return out


def score_one(
    srt_path: Path,
    transcripts_root: Path,
    max_seconds: float,
    cfg: ScoringConfig,
    lt: LanguageToolClient,
) -> Dict[str, Any] | None:
    raw_segments = load_srt_segments(srt_path)
    srt_segments = clip_segments(raw_segments, 0.0, max_seconds)

    assistant_guide_srt, patient_srt = _assistant_patient_split_srt(srt_segments, cfg)

    whisper_model = os.getenv("WHISPER_MODEL", "medium")
    whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    cache_dir = Path(".cache/fspascore/audio")

    policy = _audio_missing_policy()

    try:
        audio = transcribe_audio_for_title(
            title=_file_id_from_path(srt_path),
            assistant_guide_segments=assistant_guide_srt,
            max_seconds=max_seconds,
            cache_dir=cache_dir,
            whisper_model=whisper_model,
            whisper_compute_type=whisper_compute_type,
        )
        assistant_segments = audio.assistant_segments
        patient_segments = audio.patient_segments
        assistant_text = audio.assistant_text
        transcript_source = "audio_whisper_srt_guided"
        audio_identifier = audio.audio_ref.identifier
        audio_filename = audio.audio_ref.filename
    except AudioNotFoundError as e:
        if policy == "fail":
            raise
        if policy == "skip":
            print(f"[audio] SKIP: {srt_path.name} => {e}")
            return None
        # srt_fallback
        assistant_segments = assistant_guide_srt
        patient_segments = patient_srt
        assistant_text = _concat_text(assistant_guide_srt)
        transcript_source = "srt_fallback"
        audio_identifier = None
        audio_filename = None

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


def _write_full_json(public_dir: Path, cfg: ScoringConfig, items: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> None:
    payload = {
        "version": 6,
        "generated_at_utc": _utc_now_iso(),
        "config": _jsonify(asdict(cfg)),
        "items": _jsonify(items),
        "skipped": _jsonify(skipped),
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

    # 1) dedupe existing items (fix your current duplicates)
    existing_items_raw: List[Dict[str, Any]] = list(existing_full.get("items", []))
    items_by_id = _dedupe_items(existing_items_raw)

    skipped: List[Dict[str, Any]] = list(existing_full.get("skipped", []))

    # 2) build candidates: new OR retry srt_fallback (upgrade path)
    reprocess_fallback = _reprocess_fallback_enabled()
    srt_files = _list_srt_files(transcripts_dir)

    retry: List[Path] = []
    new: List[Path] = []

    for p in srt_files:
        fid = _file_id_from_path(p)
        existing = items_by_id.get(fid)
        if existing is None:
            new.append(p)
            continue
        if reprocess_fallback and existing.get("transcript_source") == "srt_fallback":
            retry.append(p)

    # prioritize retry to convert srt_fallback -> audio when audio appears later
    retry = sorted(retry, key=lambda p: str(p).lower())
    new = sorted(new, key=lambda p: str(p).lower())
    candidates = retry + new

    if max_new_files > 0:
        candidates = candidates[:max_new_files]

    transcripts_root = transcripts_dir

    with LanguageToolClient(cfg) as lt:
        for p in candidates:
            fid = _file_id_from_path(p)
            print(f"[progress] scoring file_id='{fid}'")
            try:
                item = score_one(
                    srt_path=p,
                    transcripts_root=transcripts_root,
                    max_seconds=max_seconds,
                    cfg=cfg,
                    lt=lt,
                )
                if item is not None:
                    items_by_id[fid] = item  # replace (no duplicates)
                else:
                    skipped.append(
                        {
                            "file_id": fid,
                            "transcript_path": str(p.relative_to(transcripts_root)).replace("\\", "/"),
                            "reason": "no_audio_match",
                            "at_utc": _utc_now_iso(),
                        }
                    )
            except AudioNotFoundError as e:
                skipped.append(
                    {
                        "file_id": fid,
                        "transcript_path": str(p.relative_to(transcripts_root)).replace("\\", "/"),
                        "reason": "no_audio_match_fail",
                        "error": str(e),
                        "at_utc": _utc_now_iso(),
                    }
                )
                raise

    # 3) write outputs from deduped map
    items_sorted = sorted(items_by_id.values(), key=lambda it: (it.get("transcript_path", ""), it.get("file_id", "")))
    _write_full_json(public_dir, cfg, items_sorted, skipped)
    _write_scores_json(public_dir, items_sorted)
    _write_metrics_csv(public_dir, items_sorted)
