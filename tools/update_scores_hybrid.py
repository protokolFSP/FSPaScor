"""
Hybrid scoring:
- Download audio from Internet Archive
- Compute audio metrics (LUFS/SNR/clipping/VAD pauses)
- Load matching SRT (same filename + .srt) and compute assistant language metrics
- Produce public/scores.csv and public/scores.json

Ranking:
overall_score = 0.85*assistant_overall_score + 0.15*audio_quality_score
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import language_tool_python
import pandas as pd
import requests

# IMPORTANT: script is run as `python tools/update_scores_hybrid.py`
# so imports must be local (no "tools.")
from audio_features import extract_audio_metrics
from srt_assistant_scoring import score_assistant_from_srt


CSV_COLUMNS = [
    "filename",
    "duration_s",
    "clipping_ratio",
    "lufs",
    "snr_db_est",
    "speech_ratio",
    "pause_ratio",
    "long_pauses_0p5s",
    "long_pauses_per_min",
    "audio_quality_score",
    "assistant_overall_score",
    "overall_score",
    "anamnesis_end_s",
    "presentation_start_s",
    "feedback_start_s",
    "p1_overall",
    "p1_duration_s",
    "p1_grammar_per_100w",
    "p1_language_quality",
    "p1_clarity",
    "p1_fluency",
    "p1_filler_per_100w",
    "p1_long_pauses_per_min",
    "p2_overall",
    "p2_duration_s",
    "p2_grammar_per_100w",
    "p2_language_quality",
    "p2_clarity",
    "p2_fluency",
    "p2_filler_per_100w",
    "p2_long_pauses_per_min",
    "p2_excl_short_q_count",
    "p2_excl_short_q_s",
    "schema_version",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--transcripts_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_new_files", type=int, default=10)
    ap.add_argument("--anamnesis_end_s", type=float, default=1200.0)
    ap.add_argument("--audio_exts", default="m4a,mp3,wav,flac,ogg")
    return ap.parse_args()


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CSV_COLUMNS]


def df_to_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for _, r in df.iterrows():
        obj = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        out.append(obj)
    return out


def ia_metadata(identifier: str) -> Dict[str, Any]:
    url = f"https://archive.org/metadata/{quote(identifier)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def list_audio_files(meta: Dict[str, Any], exts: Set[str]) -> List[Dict[str, Any]]:
    files = meta.get("files") or []
    out = []
    for f in files:
        name = (f.get("name") or "").strip()
        if not name:
            continue
        suf = Path(name).suffix.lower().lstrip(".")
        if suf in exts:
            out.append(f)
    out.sort(key=lambda x: str(x.get("name", "")))
    return out


def download_ia_file(identifier: str, filename: str, dst: Path) -> None:
    url = f"https://archive.org/download/{quote(identifier)}/{quote(filename)}"
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as w:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    w.write(chunk)


def find_srt(transcripts_dir: Path, audio_filename: str) -> Optional[Path]:
    target = audio_filename + ".srt"
    p = transcripts_dir / target
    if p.exists():
        return p
    hits = list(transcripts_dir.rglob(target))
    return hits[0] if hits else None


def compute_overall_score(assistant_overall: float, audio_quality: float) -> float:
    return 0.85 * float(assistant_overall) + 0.15 * float(audio_quality)


def main() -> int:
    args = parse_args()

    transcripts_dir = Path(args.transcripts_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    if not transcripts_dir.exists():
        raise SystemExit(f"Missing transcripts_dir: {transcripts_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = load_existing_csv(out_csv)
    existing = set(str(x) for x in df["filename"].dropna().tolist())

    exts = {x.strip().lower() for x in str(args.audio_exts).split(",") if x.strip()}
    meta = ia_metadata(args.identifier)
    audio_files = list_audio_files(meta, exts=exts)

    candidates: List[str] = []
    for f in audio_files:
        name = str(f.get("name", "")).strip()
        if name and name not in existing:
            candidates.append(name)

    candidates = candidates[: max(0, int(args.max_new_files))]
    if not candidates:
        out_json.write_text(json.dumps(df_to_json_records(df), ensure_ascii=False, indent=2), encoding="utf-8")
        print("No new audio files.")
        return 0

    tool = language_tool_python.LanguageTool("de-DE")

    new_rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)

        for audio_name in candidates:
            try:
                audio_path = tmpdir_p / audio_name
                download_ia_file(args.identifier, audio_name, audio_path)

                audio_m = extract_audio_metrics(audio_path)
                srt_path = find_srt(transcripts_dir, audio_name)

                if not srt_path:
                    row = {
                        "filename": audio_name,
                        "duration_s": audio_m.duration_s,
                        "clipping_ratio": audio_m.clipping_ratio,
                        "lufs": audio_m.lufs,
                        "snr_db_est": audio_m.snr_db_est,
                        "speech_ratio": audio_m.speech_ratio,
                        "pause_ratio": audio_m.pause_ratio,
                        "long_pauses_0p5s": audio_m.long_pauses_0p5s,
                        "long_pauses_per_min": audio_m.long_pauses_per_min,
                        "audio_quality_score": audio_m.audio_quality_score,
                        "assistant_overall_score": 0.0,
                        "overall_score": 0.15 * audio_m.audio_quality_score,
                        "schema_version": 8,
                    }
                    new_rows.append(row)
                    print(f"[WARN] SRT missing for {audio_name} -> audio only")
                    continue

                ass = score_assistant_from_srt(
                    srt_path=srt_path,
                    tool=tool,
                    anamnesis_end_s=float(args.anamnesis_end_s),
                )

                overall = compute_overall_score(
                    assistant_overall=float(ass["assistant_overall_score"]),
                    audio_quality=float(audio_m.audio_quality_score),
                )

                row = {
                    "filename": audio_name,
                    "duration_s": audio_m.duration_s,
                    "clipping_ratio": audio_m.clipping_ratio,
                    "lufs": audio_m.lufs,
                    "snr_db_est": audio_m.snr_db_est,
                    "speech_ratio": audio_m.speech_ratio,
                    "pause_ratio": audio_m.pause_ratio,
                    "long_pauses_0p5s": audio_m.long_pauses_0p5s,
                    "long_pauses_per_min": audio_m.long_pauses_per_min,
                    "audio_quality_score": audio_m.audio_quality_score,
                    "assistant_overall_score": float(ass["assistant_overall_score"]),
                    "overall_score": float(overall),
                    "schema_version": 8,
                    **ass,
                }
                new_rows.append(row)
                print(f"[OK] {audio_name}")

            except Exception as e:
                print(f"[FAIL] {audio_name}: {e}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        for c in CSV_COLUMNS:
            if c not in df_new.columns:
                df_new[c] = None
        df_new = df_new[CSV_COLUMNS]
        df = pd.concat([df, df_new], ignore_index=True)

    df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")
    df = df.sort_values(["overall_score", "filename"], ascending=[False, True], kind="mergesort")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(df_to_json_records(df), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv} + {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
