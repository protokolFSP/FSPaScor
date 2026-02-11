"""
Update public/assistant_scores.csv and public/assistant_scores.json using SRT files
from protokolFSP/FSPtranskript.

Assumption (confirmed by you):
- SRT filename is exactly audio filename + ".srt"
  Example: "A ....m4a.srt" -> filename column becomes "A ....m4a"

Only processes max_new SRTs that are not already present in the CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from tools.srt_assistant_scoring import score_from_srt


CSV_COLUMNS = [
    "filename",
    "assistant_overall_score",
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
    ap.add_argument("--transcripts_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_new", type=int, default=10)
    ap.add_argument("--anamnesis_end_s", type=float, default=1200.0)
    return ap.parse_args()


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[CSV_COLUMNS]


def list_srt_files(transcripts_dir: Path) -> List[Path]:
    return sorted(transcripts_dir.rglob("*.srt"))


def to_filename_from_srt(srt_path: Path) -> str:
    name = srt_path.name
    if name.lower().endswith(".srt"):
        return name[:-4]  # keep original extension like .m4a
    return name


def df_to_json_records(df: pd.DataFrame) -> List[Dict]:
    out = []
    for _, r in df.iterrows():
        obj = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        out.append(obj)
    return out


def main() -> int:
    args = parse_args()
    transcripts_dir = Path(args.transcripts_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    transcripts_dir_exists = transcripts_dir.exists()
    if not transcripts_dir_exists:
        raise SystemExit(f"Missing transcripts_dir: {transcripts_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = load_existing_csv(out_csv)
    existing = set(str(x) for x in df["filename"].dropna().tolist())

    srts = list_srt_files(transcripts_dir)
    candidates = []
    for srt in srts:
        fn = to_filename_from_srt(srt)
        if fn not in existing:
            candidates.append((fn, srt))

    candidates = candidates[: max(0, int(args.max_new))]
    if not candidates:
        # Still write JSON to keep in sync
        out_json.write_text(json.dumps(df_to_json_records(df), ensure_ascii=False, indent=2), encoding="utf-8")
        print("No new SRTs to process.")
        return 0

    new_rows: List[Dict] = []
    for fn, srt_path in candidates:
        try:
            m = score_from_srt(srt_path, anamnesis_end_s=float(args.anamnesis_end_s))
            row = {"filename": fn, **m}
            new_rows.append(row)
            print(f"[OK] {fn}")
        except Exception as e:
            print(f"[WARN] {fn} failed: {e}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        for c in CSV_COLUMNS:
            if c not in df_new.columns:
                df_new[c] = None
        df_new = df_new[CSV_COLUMNS]
        df = pd.concat([df, df_new], ignore_index=True)

    # sort: best assistant first (for debug); site can re-sort anyway
    df["assistant_overall_score"] = pd.to_numeric(df["assistant_overall_score"], errors="coerce")
    df = df.sort_values(["assistant_overall_score", "filename"], ascending=[False, True], kind="mergesort")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(df_to_json_records(df), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
