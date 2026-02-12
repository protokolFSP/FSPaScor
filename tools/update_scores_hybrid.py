"""
tools/update_scores_hybrid.py

Hybrid assistant scoring using SRT subtitles:
- Uses SRT timestamps to estimate pauses/fluency.
- Uses LanguageTool (German) to estimate grammar quality.
- Heuristically extracts assistant utterances in:
  - Phase 1 (Anamnese): mostly assistant questions
  - Phase 2 (Presentation): assistant monologue; excludes short Oberarzt questions

Outputs:
- public/metrics_hybrid.csv
- public/scores_hybrid.json (minimal)
- public/scores_hybrid.full.json (full per-file record)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

import language_tool_python


SCHEMA_VERSION = 7

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", re.UNICODE)
WEIRD_TOKEN_RE = re.compile(r"[^\wÄÖÜäöüß\-']", re.UNICODE)

SRT_TIME_RE = re.compile(
    r"(?P<sh>\d{2}):(?P<sm>\d{2}):(?P<ss>\d{2}),(?P<sms>\d{3})\s*-->\s*"
    r"(?P<eh>\d{2}):(?P<em>\d{2}):(?P<es>\d{2}),(?P<ems>\d{3})"
)

DEFAULT_FILLERS = [
    "äh", "ähm", "hm", "hmm", "also", "so", "sozusagen", "quasi", "halt", "naja", "tja",
    "okay", "ok", "mhm", "ähh", "ähhh"
]

PRESENTATION_PATTERNS = [
    r"\bich habe (einen|eine|ein)\b.*\bpatient(in)?\b",
    r"\bich (möchte|würde|wollte) (Ihnen|dir) (kurz )?(den|das|einen|eine)\b.*\bfall\b.*\bvorstellen\b",
    r"\bdarf ich\b.*\bfall\b.*\bvorstellen\b",
    r"\bhätten sie\b.*\bkurz zeit\b",
    r"\bich stelle\b.*\bfall\b.*\bvor\b",
]

FEEDBACK_PATTERNS = [
    r"\brückmeldung\b",
    r"\bfeedback\b",
    r"\breflexion\b",
    r"\bwas war gut\b",
    r"\bwas war schlecht\b",
    r"\bverbesser(n|ung)\b",
]

QUESTION_STARTERS = [
    "wie", "was", "wann", "wo", "wieviel", "welche", "welcher", "welches",
    "haben", "hatten", "nehmen", "können", "könnten", "dürfen", "möchten",
    "sind", "ist", "war", "waren", "wurde", "würde",
]

DOCTOR_CUES = [
    "können sie", "könnten sie", "haben sie", "hatten sie", "nehmen sie",
    "seit wann", "wie lange", "wo genau", "können sie bitte", "würden sie",
    "ich möchte", "ich würde", "darf ich", "bitte", "erzählen sie",
]

PATIENT_CUES = [
    "ich", "mir", "mich", "mein", "meine", "meinen", "meiner", "meinem",
    "seit", "gestern", "heute", "vorhin", "schmerzen", "mir ist", "ich habe",
]


@dataclass(frozen=True)
class SrtBlock:
    start_s: float
    end_s: float
    text: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sec_from_match(m: re.Match) -> Tuple[float, float]:
    sh, sm, ss, sms = int(m["sh"]), int(m["sm"]), int(m["ss"]), int(m["sms"])
    eh, em, es, ems = int(m["eh"]), int(m["em"]), int(m["es"]), int(m["ems"])
    start = sh * 3600 + sm * 60 + ss + sms / 1000.0
    end = eh * 3600 + em * 60 + es + ems / 1000.0
    return start, end


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\.(m4a|mp3|wav|ogg|flac|srt)$", "", s)
    s = s.replace("ü", "u").replace("ö", "o").replace("ä", "a").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def filler_count(words: List[str], fillers: List[str]) -> int:
    filler_set = {f.lower() for f in fillers}
    return sum(1 for w in words if w.lower() in filler_set)


def weird_token_rate(text: str, words: List[str]) -> float:
    if not words:
        return 0.0
    weird = WEIRD_TOKEN_RE.findall(text)
    # rough: weird char count / word count
    return clamp(len(weird) / len(words), 0.0, 1.0)


def repetition_rate(words: List[str]) -> float:
    if len(words) < 6:
        return 0.0
    reps = 0
    for i in range(2, len(words)):
        if words[i].lower() == words[i - 1].lower() == words[i - 2].lower():
            reps += 1
    return reps / len(words)


def parse_srt(path: Path) -> List[SrtBlock]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in raw.splitlines()]

    blocks: List[SrtBlock] = []
    i = 0
    while i < len(lines):
        # skip empty
        if not lines[i].strip():
            i += 1
            continue

        # optional index line
        if lines[i].strip().isdigit():
            i += 1
            if i >= len(lines):
                break

        # time line
        m = SRT_TIME_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start_s, end_s = sec_from_match(m)
        i += 1

        # text lines until blank
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        if text:
            blocks.append(SrtBlock(start_s=start_s, end_s=end_s, text=text))
    return blocks


def find_first_time(blocks: List[SrtBlock], patterns: List[str], start_after_s: float = 0.0) -> Optional[float]:
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    for b in blocks:
        if b.start_s < start_after_s:
            continue
        t = b.text
        for r in regs:
            if r.search(t):
                return b.start_s
    return None


def is_short_question(text: str, duration_s: float, max_s: float, max_words: int) -> bool:
    ws = tokenize_words(text)
    t = text.strip().lower()
    qmark = "?" in text
    starts_like_q = any(t.startswith(q + " ") for q in QUESTION_STARTERS)
    if duration_s <= max_s and len(ws) <= max_words and (qmark or starts_like_q):
        return True
    if duration_s <= max_s and len(ws) <= max_words and "sie" in t and ("?" in text):
        return True
    return False


def looks_like_assistant_p1(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False

    # strong cues
    if "?" in text:
        return True
    if any(t.startswith(q + " ") for q in QUESTION_STARTERS):
        return True
    if any(cue in t for cue in DOCTOR_CUES):
        return True

    # weak heuristic: short + addressing "Sie"
    ws = tokenize_words(text)
    if "sie" in t and len(ws) <= 18 and not t.startswith("ich "):
        return True

    # avoid likely patient monologue
    if any(t.startswith(c + " ") for c in PATIENT_CUES) and len(ws) >= 10:
        return False

    return False


def blocks_in_range(blocks: List[SrtBlock], start_s: float, end_s: float) -> List[SrtBlock]:
    out: List[SrtBlock] = []
    for b in blocks:
        if b.end_s <= start_s:
            continue
        if b.start_s >= end_s:
            break
        out.append(b)
    return out


def compute_pause_stats(blocks: List[SrtBlock], pause_threshold_s: float, window_duration_s: float) -> Tuple[int, float]:
    if window_duration_s <= 1.0:
        return 0, 0.0
    if len(blocks) < 2:
        return 0, 0.0
    blocks_sorted = sorted(blocks, key=lambda x: x.start_s)
    long_pauses = 0
    for prev, cur in zip(blocks_sorted, blocks_sorted[1:]):
        gap = cur.start_s - prev.end_s
        if gap >= pause_threshold_s:
            long_pauses += 1
    per_min = long_pauses / (window_duration_s / 60.0)
    return long_pauses, per_min


def chunk_text(text: str, max_chars: int = 4500) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    cur = []
    cur_len = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent:
            continue
        if cur_len + len(sent) + 1 > max_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks


def grammar_errors_per_100w(tool: language_tool_python.LanguageTool, text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    total_matches = 0
    for chunk in chunk_text(text):
        matches = tool.check(chunk)
        total_matches += len(matches)
    return (total_matches / len(words)) * 100.0


def score_components(
    tool: language_tool_python.LanguageTool,
    blocks: List[SrtBlock],
    window_duration_s: float,
    fillers: List[str],
    pause_threshold_s: float,
) -> Dict[str, float]:
    text = " ".join(b.text for b in blocks).strip()
    words = tokenize_words(text)
    wc = len(words)

    fcount = filler_count(words, fillers)
    f_per_100w = (fcount / wc) * 100.0 if wc else 0.0

    weird = weird_token_rate(text, words)
    rep = repetition_rate(words)

    long_pause_count, long_pause_per_min = compute_pause_stats(blocks, pause_threshold_s, window_duration_s)

    g_per_100w = grammar_errors_per_100w(tool, text) if text else 0.0

    # language_quality: mostly grammar-driven
    language_quality = clamp(100.0 - (g_per_100w * 8.0), 0.0, 100.0)

    # clarity: penalize fillers + repetitions + weird tokens
    clarity_pen = clamp(f_per_100w * 3.0, 0.0, 45.0) + clamp(rep * 100.0 * 0.8, 0.0, 35.0) + clamp(weird * 100.0 * 0.6, 0.0, 30.0)
    clarity = clamp(100.0 - clarity_pen, 0.0, 100.0)

    # fluency: penalize pauses + fillers
    fluency_pen = clamp(long_pause_per_min * 6.5, 0.0, 70.0) + clamp(f_per_100w * 1.8, 0.0, 40.0)
    fluency = clamp(100.0 - fluency_pen, 0.0, 100.0)

    overall = clamp(0.50 * language_quality + 0.25 * clarity + 0.25 * fluency, 0.0, 100.0)

    return {
        "word_count": float(wc),
        "filler_count": float(fcount),
        "filler_per_100w": float(f_per_100w),
        "weird_token_rate": float(weird),
        "repetition_rate": float(rep),
        "long_pauses_count": float(long_pause_count),
        "long_pauses_per_min": float(long_pause_per_min),
        "grammar_per_100w": float(g_per_100w),
        "language_quality": float(language_quality),
        "clarity": float(clarity),
        "fluency": float(fluency),
        "overall": float(overall),
    }


def ensure_clone(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def build_srt_index(transcripts_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in transcripts_dir.rglob("*.srt"):
        idx[normalize_name(p.name)] = p
    return idx


def find_srt_for_audio(audio_filename: str, srt_index: Dict[str, Path]) -> Optional[Path]:
    key = normalize_name(audio_filename)
    if key in srt_index:
        return srt_index[key]

    # fuzzy fallback
    choices = list(srt_index.keys())
    if not choices:
        return None
    best = process.extractOne(key, choices, scorer=fuzz.WRatio)
    if not best:
        return None
    match_key, score, _ = best
    if score < 86:
        return None
    return srt_index.get(match_key)


def load_existing_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def choose_audio_metrics_csv(public_dir: Path) -> Optional[Path]:
    # prefer explicit known names if present
    candidates = [
        public_dir / "metrics3.csv",
        public_dir / "metrics.csv",
        public_dir / "metricsv2.csv",
        public_dir / "metricsv1.csv",
        public_dir / "metrics_alt.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: any metrics*.csv excluding hybrid
    all_csv = sorted(public_dir.glob("metrics*.csv"))
    all_csv = [p for p in all_csv if "hybrid" not in p.name.lower()]
    return all_csv[0] if all_csv else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts_repo", default="https://github.com/protokolFSP/FSPtranskript.git")
    ap.add_argument("--transcripts_dir", default="data/FSPtranskript/transcripts")
    ap.add_argument("--clone_dir", default="data/FSPtranskript")
    ap.add_argument("--public_dir", default="public")

    ap.add_argument("--audio_metrics_csv", default="")
    ap.add_argument("--max_new_files", type=int, default=10)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--anamnesis_seconds", type=float, default=1200.0)
    ap.add_argument("--pause_threshold_s", type=float, default=0.5)

    ap.add_argument("--p2_short_q_max_s", type=float, default=6.0)
    ap.add_argument("--p2_short_q_max_words", type=int, default=10)

    ap.add_argument("--out_metrics_csv", default="public/metrics_hybrid.csv")
    ap.add_argument("--out_scores_json", default="public/scores_hybrid.json")
    ap.add_argument("--out_scores_full_json", default="public/scores_hybrid.full.json")

    ap.add_argument("--p1_weight", type=float, default=0.65)
    ap.add_argument("--p2_weight", type=float, default=0.35)

    ap.add_argument("--fillers", default=",".join(DEFAULT_FILLERS))
    args = ap.parse_args()

    public_dir = Path(args.public_dir)
    public_dir.mkdir(parents=True, exist_ok=True)

    out_metrics_csv = Path(args.out_metrics_csv)
    out_scores_json = Path(args.out_scores_json)
    out_scores_full_json = Path(args.out_scores_full_json)

    clone_dir = Path(args.clone_dir)
    transcripts_dir = Path(args.transcripts_dir)

    # Clone transcripts if missing
    if not transcripts_dir.exists():
        ensure_clone(args.transcripts_repo, clone_dir)

    if not transcripts_dir.exists():
        raise SystemExit(f"transcripts_dir not found: {transcripts_dir}")

    srt_index = build_srt_index(transcripts_dir)
    if not srt_index:
        raise SystemExit(f"No .srt files found in {transcripts_dir}")

    audio_metrics_csv = Path(args.audio_metrics_csv) if args.audio_metrics_csv else None
    if not audio_metrics_csv:
        audio_metrics_csv = choose_audio_metrics_csv(public_dir)

    audio_df = pd.DataFrame()
    if audio_metrics_csv and audio_metrics_csv.exists():
        audio_df = pd.read_csv(audio_metrics_csv)
        if "filename" not in audio_df.columns:
            audio_df = pd.DataFrame()

    existing_df = load_existing_metrics(out_metrics_csv)
    already = set(existing_df["filename"].astype(str).tolist()) if (not existing_df.empty and "filename" in existing_df.columns) else set()

    # Candidate filenames: prefer audio metrics list (so we score only audios we already know)
    if not audio_df.empty:
        filenames = audio_df["filename"].astype(str).tolist()
    else:
        filenames = [p.name.replace(".srt", ".m4a") for p in srt_index.values()]

    if not args.force:
        filenames = [f for f in filenames if f not in already]

    filenames = filenames[: max(0, int(args.max_new_files))]
    if not filenames:
        print("No new files to score.")
        return 0

    fillers = [x.strip() for x in args.fillers.split(",") if x.strip()]

    tool = language_tool_python.LanguageTool("de-DE")

    rows: List[Dict[str, float | str]] = []

    for fn in tqdm(filenames, desc="Hybrid scoring"):
        srt_path = find_srt_for_audio(fn, srt_index)
        if not srt_path:
            continue

        blocks = parse_srt(srt_path)
        if not blocks:
            continue

        total_end = max(b.end_s for b in blocks)
        anam_end = min(float(args.anamnesis_seconds), float(total_end))

        pres_start = find_first_time(blocks, PRESENTATION_PATTERNS, start_after_s=0.0)
        if pres_start is None:
            pres_start = anam_end

        fb_start = find_first_time(blocks, FEEDBACK_PATTERNS, start_after_s=float(pres_start))
        if fb_start is None:
            fb_start = float(total_end)

        # Phase 1: [0, pres_start) but cap to anamnesis window
        p1_end = min(float(pres_start), anam_end)
        p1_blocks_all = blocks_in_range(blocks, 0.0, p1_end)
        p1_blocks = [b for b in p1_blocks_all if looks_like_assistant_p1(b.text)]
        p1_dur = max(0.0, p1_end - 0.0)

        # Phase 2: [pres_start, fb_start)
        p2_blocks_all = blocks_in_range(blocks, float(pres_start), float(fb_start))
        p2_dur = max(0.0, float(fb_start) - float(pres_start))

        excl_q_count = 0
        excl_q_s = 0.0
        p2_blocks: List[SrtBlock] = []
        for b in p2_blocks_all:
            dur = max(0.0, b.end_s - b.start_s)
            if is_short_question(b.text, dur, float(args.p2_short_q_max_s), int(args.p2_short_q_max_words)):
                excl_q_count += 1
                excl_q_s += dur
                continue
            p2_blocks.append(b)

        p1 = score_components(tool, p1_blocks, p1_dur, fillers, float(args.pause_threshold_s)) if p1_dur > 1 else {
            "grammar_per_100w": 0.0, "language_quality": 0.0, "clarity": 0.0, "fluency": 0.0,
            "filler_per_100w": 0.0, "long_pauses_per_min": 0.0, "overall": 0.0,
        }
        p2 = score_components(tool, p2_blocks, p2_dur, fillers, float(args.pause_threshold_s)) if p2_dur > 1 else {
            "grammar_per_100w": 0.0, "language_quality": 0.0, "clarity": 0.0, "fluency": 0.0,
            "filler_per_100w": 0.0, "long_pauses_per_min": 0.0, "overall": 0.0,
        }

        # duration-weighted assistant overall across phases
        w1 = float(args.p1_weight)
        w2 = float(args.p2_weight)
        if p1_dur < 30:
            w1 = 0.25
            w2 = 0.75
        if p2_dur < 30:
            w1 = 0.80
            w2 = 0.20
        denom = (w1 + w2) if (w1 + w2) > 0 else 1.0
        assistant_overall = (w1 * float(p1["overall"]) + w2 * float(p2["overall"])) / denom

        rows.append(
            {
                "filename": fn,
                "assistant_overall_score": float(assistant_overall),
                "anamnesis_end_s": float(args.anamnesis_seconds),
                "presentation_start_s": float(pres_start),
                "feedback_start_s": float(fb_start),
                "p1_overall": float(p1["overall"]),
                "p1_duration_s": float(p1_dur),
                "p1_grammar_per_100w": float(p1["grammar_per_100w"]),
                "p1_language_quality": float(p1["language_quality"]),
                "p1_clarity": float(p1["clarity"]),
                "p1_fluency": float(p1["fluency"]),
                "p1_filler_per_100w": float(p1["filler_per_100w"]),
                "p1_long_pauses_per_min": float(p1["long_pauses_per_min"]),
                "p2_overall": float(p2["overall"]),
                "p2_duration_s": float(p2_dur),
                "p2_grammar_per_100w": float(p2["grammar_per_100w"]),
                "p2_language_quality": float(p2["language_quality"]),
                "p2_clarity": float(p2["clarity"]),
                "p2_fluency": float(p2["fluency"]),
                "p2_filler_per_100w": float(p2["filler_per_100w"]),
                "p2_long_pauses_per_min": float(p2["long_pauses_per_min"]),
                "p2_excl_short_q_count": int(excl_q_count),
                "p2_excl_short_q_s": float(excl_q_s),
                "schema_version": int(SCHEMA_VERSION),
            }
        )

    if not rows:
        print("No rows produced (no matching SRTs).")
        return 0

    new_df = pd.DataFrame(rows)
    if not existing_df.empty:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df

    # stable ordering: best score first
    merged = merged.sort_values(by=["assistant_overall_score", "filename"], ascending=[False, True])

    out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_metrics_csv, index=False)

    # Minimal JSON for UI
    minimal = [
        {"filename": r["filename"], "assistant_overall_score": float(r["assistant_overall_score"])}
        for r in merged.to_dict(orient="records")
    ]
    out_scores_json.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")

    # Full JSON
    full = merged.to_dict(orient="records")
    out_scores_full_json.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_metrics_csv}")
    print(f"Wrote: {out_scores_json}")
    print(f"Wrote: {out_scores_full_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
