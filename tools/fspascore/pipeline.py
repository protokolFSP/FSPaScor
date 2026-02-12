from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import ScoringConfig
from .grammar import LanguageToolClient, compute_grammar_metrics
from .heuristics import is_assistant_line
from .pause import compute_turn_aware_pause_metrics
from .srt_io import clip_segments, load_srt_segments
from .text_metrics import compute_text_metrics


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_round(x: Any, nd: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _jsonify(obj: Any) -> Any:
    """
    Make obj JSON-serializable (handles frozenset/set/tuple/dataclasses-like dicts).
    """
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


def _load_existing_full(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "generated_at_utc": None, "config": {}, "items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" not in data:
        data["items"] = []
    return data


def _existing_ids(full: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for it in full.get("items", []):
        fid = it.get("file_id")
        if isinstance(fid, str) and fid:
            ids.add(fid)
    return ids


def _file_id_from_path(srt_path: Path) -> str:
    return srt_path.stem


def _assistant_patient_split(segments, cfg: ScoringConfig):
    assistant = []
    patient = []
    for s in segments:
        if is_assistant_line(s.text, cfg):
            assistant.append(s)
        else:
            patient.append(s)
    return assistant, patient


def _concat_text(segments) -> str:
    return " ".join(s.text for s in segments).strip()


def score_one_srt(
    srt_path: Path,
    max_seconds: float,
    cfg: ScoringConfig,
    lt: LanguageToolClient,
    transcripts_root: Path,
) -> Dict[str, Any]:
    raw_segments = load_srt_segments(srt_path)
    segments = clip_segments(raw_segments, 0.0, max_seconds)

    assistant_segments, patient_segments = _assistant_patient_split(segments, cfg)
    assistant_text = _concat_text(assistant_segments)

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

    overall = (
        0.70 * language_quality
        + 0.20 * textm.clarity_score
        + 0.10 * textm.fluency_score
    )
    overall = _clamp(overall)

    rel_path = str(srt_path.relative_to(transcripts_root)).replace("\\", "/")
    file_id = _file_id_from_path(srt_path)

    item: Dict[str, Any] = {
        "file_id": file_id,
        "transcript_path": rel_path,
        "max_seconds": _safe_round(max_seconds, 3),
        "processed_at_utc": _utc_now_iso(),
        "assistant_line_count": len(assistant_segments),
        "patient_line_count": len(patient_segments),
        "assistant_text_word_count": textm.word_count,
        "grammar_error_count": grammar.error_count,
        "grammar_ignored_count": grammar.ignored_count,
        "grammar_per_100w": _safe_round(grammar.grammar_per_100w, 6),
        "language_quality": _safe_round(language_quality, 6),
        "filler_per_100w": _safe_round(textm.filler_per_100w, 6),
        "repetition_per_100w": _safe_round(textm.repetition_per_100w, 6),
        "weird_token_per_100w": _safe_round(textm.weird_per_100w, 6),
        "clarity": _safe_round(textm.clarity_score, 6),
        "fluency": _safe_round(textm.fluency_score, 6),
        "assistant_silence_total_sec": _safe_round(pause.assistant_silence_total_sec, 6),
        "assistant_long_silence_total_sec": _safe_round(pause.assistant_long_silence_total_sec, 6),
        "long_silence_sec_per_min": _safe_round(pause.long_silence_sec_per_min, 6),
        "overall": _safe_round(overall, 6),
        "silence_gaps": tuple(_safe_round(x, 6) for x in pause.silence_gaps),
    }
    return item


def _write_scores_json(public_dir: Path, items: List[Dict[str, Any]]) -> None:
    minimal = [
        {
            "file_id": it["file_id"],
            "transcript_path": it["transcript_path"],
            "overall": it["overall"],
            "language_quality": it["language_quality"],
            "clarity": it["clarity"],
            "fluency": it["fluency"],
        }
        for it in items
    ]
    out = public_dir / "scores_1120.json"
    out.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_full_json(public_dir: Path, cfg: ScoringConfig, items: List[Dict[str, Any]]) -> None:
    out = public_dir / "scores_1120.full.json"
    payload = {
        "version": 1,
        "generated_at_utc": _utc_now_iso(),
        "config": _jsonify(asdict(cfg)),
        "items": _jsonify(items),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_metrics_csv(public_dir: Path, items: List[Dict[str, Any]]) -> None:
    out = public_dir / "metrics_1120.csv"
    fieldnames = [
        "file_id",
        "transcript_path",
        "max_seconds",
        "processed_at_utc",
        "assistant_line_count",
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
            row = {k: it.get(k, "") for k in fieldnames}
            w.writerow(row)


def run_pipeline(
    transcripts_dir: Path,
    public_dir: Path,
    max_seconds: float,
    max_new_files: int,
) -> None:
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

    if not new_files:
        items_sorted = sorted(
            existing_items,
            key=lambda it: (it.get("transcript_path", ""), it.get("file_id", "")),
        )
        _write_full_json(public_dir, cfg, items_sorted)
        _write_scores_json(public_dir, items_sorted)
        _write_metrics_csv(public_dir, items_sorted)
        return

    with LanguageToolClient(cfg) as lt:
        for p in new_files:
            item = score_one_srt(
                srt_path=p,
                max_seconds=max_seconds,
                cfg=cfg,
                lt=lt,
                transcripts_root=transcripts_root,
            )
            existing_items.append(item)

    items_sorted = sorted(
        existing_items,
        key=lambda it: (it.get("transcript_path", ""), it.get("file_id", "")),
    )

    _write_full_json(public_dir, cfg, items_sorted)
    _write_scores_json(public_dir, items_sorted)
    _write_metrics_csv(public_dir, items_sorted)
