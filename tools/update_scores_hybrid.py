"""
tools/update_scores_hybrid.py

Hybrid assistant scoring using SRT subtitles (German medical dialogs).

Fixes:
- Speaker-turn aware pause counting (patient speech is NOT counted as assistant pauses)
- Robust presentation-start detection (regex + fuzzy keyword scoring)
- Grammar scoring calibrated + LanguageTool filtering (ignore style/punct/whitespace)
- Optional medical allowlist to ignore spelling errors on medical terms

Outputs:
- public/metrics_hybrid.csv
- public/scores_hybrid.json
- public/scores_hybrid.full.json
"""

from __future__ import annotations

import argparse
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


SCHEMA_VERSION = 8

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

QUESTION_STARTERS = [
    "wie", "was", "wann", "wo", "wieviel", "welche", "welcher", "welches",
    "haben", "hatten", "nehmen", "können", "könnten", "dürfen", "möchten",
    "sind", "ist", "war", "waren", "wurde", "würde",
]

DOCTOR_CUES = [
    "können sie", "könnten sie", "haben sie", "hatten sie", "nehmen sie",
    "seit wann", "wie lange", "wo genau", "können sie bitte", "würden sie",
    "darf ich", "erzählen sie", "ich möchte", "ich würde",
]

PRESENTATION_PATTERNS = [
    r"\bich habe\b.*\bpatient(in|en)?\b",
    r"\bich (möchte|würde|wollte)\b.*\bfall\b.*\bvorstell",
    r"\bdarf ich\b.*\bfall\b.*\bvorstell",
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

# LanguageTool filters: keep only these issue types (best-effort; LT may vary)
KEEP_ISSUE_TYPES = {"grammar", "misspelling", "typographical"}
IGNORE_CATEGORY_IDS = {"STYLE", "TYPOGRAPHY", "PUNCTUATION", "WHITESPACE", "CASING"}


@dataclass(frozen=True)
class SrtBlock:
    start_s: float
    end_s: float
    text: str


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\.(m4a|mp3|wav|ogg|flac|srt)$", "", s)
    s = s.replace("ü", "u").replace("ö", "o").replace("ä", "a").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sec_from_match(m: re.Match) -> Tuple[float, float]:
    sh, sm, ss, sms = int(m["sh"]), int(m["sm"]), int(m["ss"]), int(m["sms"])
    eh, em, es, ems = int(m["eh"]), int(m["em"]), int(m["es"]), int(m["ems"])
    start = sh * 3600 + sm * 60 + ss + sms / 1000.0
    end = eh * 3600 + em * 60 + es + ems / 1000.0
    return start, end


def parse_srt(path: Path) -> List[SrtBlock]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in raw.splitlines()]

    blocks: List[SrtBlock] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        if lines[i].strip().isdigit():
            i += 1
            if i >= len(lines):
                break

        m = SRT_TIME_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start_s, end_s = sec_from_match(m)
        i += 1

        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines).strip()
        if text:
            blocks.append(SrtBlock(start_s=start_s, end_s=end_s, text=text))
    return sorted(blocks, key=lambda b: b.start_s)


def blocks_in_range(blocks: List[SrtBlock], start_s: float, end_s: float) -> List[SrtBlock]:
    out: List[SrtBlock] = []
    for b in blocks:
        if b.end_s <= start_s:
            continue
        if b.start_s >= end_s:
            break
        out.append(b)
    return out


def looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if "?" in text:
        return True
    if any(t.startswith(q + " ") for q in QUESTION_STARTERS):
        return True
    if any(cue in t for cue in DOCTOR_CUES):
        return True
    return False


def is_short_question(text: str, duration_s: float, max_s: float, max_words: int) -> bool:
    ws = tokenize_words(text)
    t = text.strip().lower()
    if duration_s <= max_s and len(ws) <= max_words and looks_like_question(text):
        return True
    # extra: very short interrogatives even without '?'
    if duration_s <= max_s and len(ws) <= max_words and any(t.startswith(q + " ") for q in QUESTION_STARTERS):
        return True
    return False


def find_first_time_regex(blocks: List[SrtBlock], patterns: List[str], start_after_s: float = 0.0) -> Optional[float]:
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    for b in blocks:
        if b.start_s < start_after_s:
            continue
        for r in regs:
            if r.search(b.text):
                return b.start_s
    return None


def find_presentation_start(blocks: List[SrtBlock], fallback_s: float) -> float:
    # 1) regex
    t = find_first_time_regex(blocks, PRESENTATION_PATTERNS, start_after_s=0.0)
    if t is not None:
        return float(t)

    # 2) fuzzy keyword scoring: find earliest block with enough “presentation” keywords
    keywords = [
        ("vorstell", 2),
        ("fall", 2),
        ("patient", 2),
        ("patientin", 2),
        ("kurz zeit", 2),
        ("oberarzt", 1),
        ("darf ich", 2),
        ("ich habe", 1),
        ("ich möchte", 1),
        ("ich würde", 1),
    ]

    for b in blocks:
        txt = b.text.lower()
        score = 0
        for kw, w in keywords:
            if kw in txt:
                score += w
        if score >= 4:
            return float(b.start_s)

    return float(fallback_s)


def find_feedback_start(blocks: List[SrtBlock], start_after_s: float, fallback_s: float) -> float:
    t = find_first_time_regex(blocks, FEEDBACK_PATTERNS, start_after_s=start_after_s)
    if t is not None:
        return float(t)
    return float(fallback_s)


def filler_count(words: List[str], fillers: List[str]) -> int:
    filler_set = {f.lower() for f in fillers}
    return sum(1 for w in words if w.lower() in filler_set)


def weird_token_rate(text: str, words: List[str]) -> float:
    if not words:
        return 0.0
    weird = WEIRD_TOKEN_RE.findall(text)
    return clamp(len(weird) / len(words), 0.0, 1.0)


def repetition_rate(words: List[str]) -> float:
    if len(words) < 6:
        return 0.0
    reps = 0
    for i in range(2, len(words)):
        if words[i].lower() == words[i - 1].lower() == words[i - 2].lower():
            reps += 1
    return reps / len(words)


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


def load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    terms = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        terms.add(s.lower())
    return terms


def _lt_issue_type(match) -> str:
    # language_tool_python provides match.ruleIssueType
    it = getattr(match, "ruleIssueType", None)
    if it:
        return str(it).lower()
    return ""


def _lt_category_id(match) -> str:
    cat = getattr(match, "category", None)
    if isinstance(cat, dict) and "id" in cat:
        return str(cat["id"]).upper()
    # some versions use match.rule.category.id
    rule = getattr(match, "rule", None)
    if rule is not None:
        category = getattr(rule, "category", None)
        if category is not None:
            cid = getattr(category, "id", None)
            if cid:
                return str(cid).upper()
    return ""


def is_sentence_start(text: str, offset: int) -> bool:
    if offset <= 0:
        return True
    prefix = text[:offset]
    # last strong punctuation
    m = re.search(r"[.!?]\s*$", prefix)
    if m:
        return True
    # newline separation in some SRTs
    if prefix.endswith("\n"):
        return True
    return False


def grammar_errors_per_100w(tool: language_tool_python.LanguageTool, text: str, allowlist: set[str]) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0

    kept = 0
    for chunk in chunk_text(text):
        matches = tool.check(chunk)
        for m in matches:
            issue_type = _lt_issue_type(m)
            cat_id = _lt_category_id(m)

            if issue_type and issue_type not in KEEP_ISSUE_TYPES:
                continue
            if cat_id and cat_id in IGNORE_CATEGORY_IDS:
                continue

            off = int(getattr(m, "offset", 0))
            ln = int(getattr(m, "errorLength", 0))
            bad = chunk[off: off + ln].strip()
            bad_lc = bad.lower()

            # ignore tiny tokens
            if len(bad_lc) <= 2:
                continue

            # ignore allowlisted medical terms / abbreviations
            if bad_lc in allowlist:
                continue

            # ignore likely proper nouns (capitalized not at sentence start)
            if bad and bad[0].isupper() and not is_sentence_start(chunk, off):
                continue

            # ignore tokens that contain digits/hyphen (often dosages / abbreviations)
            if any(ch.isdigit() for ch in bad) or "-" in bad:
                continue

            kept += 1

    return (kept / len(words)) * 100.0


def compute_turn_pauses(
    blocks: List[SrtBlock],
    labels: List[str],
    pause_threshold_s: float,
    window_duration_s: float,
    target_label: str,
) -> Tuple[int, float]:
    """
    Count pauses ONLY within the same speaker runs:
    consecutive blocks with same label, with no other speaker between them.
    """
    if window_duration_s <= 1.0 or len(blocks) < 2:
        return 0, 0.0

    long_pauses = 0
    prev_end = None
    prev_label = None

    for b, lab in zip(blocks, labels):
        if prev_end is None:
            prev_end = b.end_s
            prev_label = lab
            continue

        gap = b.start_s - prev_end
        if lab == target_label and prev_label == target_label:
            if gap >= pause_threshold_s:
                long_pauses += 1

        prev_end = b.end_s
        prev_label = lab

    per_min = long_pauses / (window_duration_s / 60.0)
    return long_pauses, per_min


def score_components(
    tool: language_tool_python.LanguageTool,
    blocks: List[SrtBlock],
    labels: List[str],
    window_duration_s: float,
    fillers: List[str],
    pause_threshold_s: float,
    allowlist: set[str],
    target_label: str = "assistant",
) -> Dict[str, float]:
    # only target speaker text
    speaker_blocks = [b for b, lab in zip(blocks, labels) if lab == target_label]
    text = " ".join(b.text for b in speaker_blocks).strip()

    words = tokenize_words(text)
    wc = len(words)

    fcount = filler_count(words, fillers)
    f_per_100w = (fcount / wc) * 100.0 if wc else 0.0

    weird = weird_token_rate(text, words)
    rep = repetition_rate(words)

    # pauses computed within same-speaker runs
    long_pause_count, long_pause_per_min = compute_turn_pauses(
        blocks=blocks,
        labels=labels,
        pause_threshold_s=pause_threshold_s,
        window_duration_s=window_duration_s,
        target_label=target_label,
    )

    g_per_100w = grammar_errors_per_100w(tool, text, allowlist) if text else 0.0

    # calibrated mapping (much softer than linear)
    language_quality = clamp(100.0 - (18.0 * math.log1p(g_per_100w)), 0.0, 100.0)

    # clarity: soft penalties
    clarity_pen = clamp(f_per_100w * 2.2, 0.0, 35.0) + clamp(rep * 100.0 * 0.7, 0.0, 25.0) + clamp(weird * 100.0 * 0.4, 0.0, 20.0)
    clarity = clamp(100.0 - clarity_pen, 0.0, 100.0)

    # fluency: pauses + fillers (turn-based)
    fluency_pen = clamp(long_pause_per_min * 7.0, 0.0, 60.0) + clamp(f_per_100w * 1.2, 0.0, 25.0)
    fluency = clamp(100.0 - fluency_pen, 0.0, 100.0)

    overall = clamp(0.65 * language_quality + 0.25 * clarity + 0.10 * fluency, 0.0, 100.0)

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
    all_csv = sorted(public_dir.glob("metrics*.csv"))
    all_csv = [p for p in all_csv if "hybrid" not in p.name.lower()]
    return all_csv[0] if all_csv else None


def label_phase1(blocks: List[SrtBlock]) -> List[str]:
    """
    State machine:
    - When a question-like block appears => assistant
    - Following blocks => patient until next question-like block
    Also treat greeting/introduction as assistant if it contains doctor cues or station intro.
    """
    labels: List[str] = []
    patient_mode = True

    for b in blocks:
        t = b.text.strip().lower()

        intro = ("ich bin" in t and ("station" in t or "dienst" in t or "assist" in t))
        q = looks_like_question(b.text)

        if intro or q:
            labels.append("assistant")
            patient_mode = True  # next is likely patient answer
        else:
            if patient_mode:
                labels.append("patient")
            else:
                labels.append("assistant")

        # switch to patient after assistant question/intro
        if labels[-1] == "assistant":
            patient_mode = True

    return labels


def label_phase2(blocks: List[SrtBlock], short_q_max_s: float, short_q_max_words: int) -> Tuple[List[str], int, float]:
    """
    Phase2: assistant presents; oberarzt asks questions.
    Mark question-like blocks as oberarzt_question (excluded).
    """
    labels: List[str] = []
    excl_count = 0
    excl_s = 0.0

    for b in blocks:
        dur = max(0.0, b.end_s - b.start_s)
        if looks_like_question(b.text):
            # mark as oberarzt question (short or long)
            if is_short_question(b.text, dur, short_q_max_s, short_q_max_words):
                excl_count += 1
                excl_s += dur
                labels.append("oberarzt_question")
            else:
                # still exclude from assistant scoring, but track as "oberarzt_question"
                excl_count += 1
                excl_s += dur
                labels.append("oberarzt_question")
        else:
            labels.append("assistant")

    return labels, excl_count, excl_s


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

    ap.add_argument("--p1_weight", type=float, default=0.70)
    ap.add_argument("--p2_weight", type=float, default=0.30)

    ap.add_argument("--audio_weight", type=float, default=0.12)  # optional blend if audio metrics exist
    ap.add_argument("--fillers", default=",".join(DEFAULT_FILLERS))
    ap.add_argument("--medical_allowlist", default="public/medical_allowlist.auto.txt")

    args = ap.parse_args()

    public_dir = Path(args.public_dir)
    public_dir.mkdir(parents=True, exist_ok=True)

    out_metrics_csv = Path(args.out_metrics_csv)
    out_scores_json = Path(args.out_scores_json)
    out_scores_full_json = Path(args.out_scores_full_json)

    clone_dir = Path(args.clone_dir)
    transcripts_dir = Path(args.transcripts_dir)

    if not transcripts_dir.exists():
        ensure_clone(args.transcripts_repo, clone_dir)
    if not transcripts_dir.exists():
        raise SystemExit(f"transcripts_dir not found: {transcripts_dir}")

    srt_index = build_srt_index(transcripts_dir)
    if not srt_index:
        raise SystemExit(f"No .srt files found in {transcripts_dir}")

    fillers = [x.strip() for x in args.fillers.split(",") if x.strip()]
    allowlist = load_allowlist(Path(args.medical_allowlist))

    tool = language_tool_python.LanguageTool("de-DE")

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

    rows: List[Dict[str, float | str | int]] = []

    # helper map for optional audio blend
    audio_map: Dict[str, Dict[str, float]] = {}
    if not audio_df.empty:
        for r in audio_df.to_dict(orient="records"):
            fn = str(r.get("filename", ""))
            if not fn:
                continue
            aq = float(r.get("audio_quality_score", 0.0)) if "audio_quality_score" in r else 0.0
            fl = float(r.get("fluency_score", 0.0)) if "fluency_score" in r else 0.0
            audio_map[fn] = {"audio_quality_score": aq, "fluency_score": fl}

    for fn in tqdm(filenames, desc="Hybrid scoring v8"):
        srt_path = find_srt_for_audio(fn, srt_index)
        if not srt_path:
            continue

        blocks = parse_srt(srt_path)
        if not blocks:
            continue

        total_end = max(b.end_s for b in blocks)
        anam_end = min(float(args.anamnesis_seconds), float(total_end))

        pres_start = find_presentation_start(blocks, fallback_s=anam_end)
        fb_start = find_feedback_start(blocks, start_after_s=float(pres_start), fallback_s=float(total_end))

        # Phase 1 range
        p1_end = min(float(pres_start), anam_end)
        p1_blocks = blocks_in_range(blocks, 0.0, p1_end)
        p1_labels = label_phase1(p1_blocks)
        p1_dur = max(0.0, p1_end - 0.0)

        # Phase 2 range
        p2_blocks = blocks_in_range(blocks, float(pres_start), float(fb_start))
        p2_labels, excl_q_count, excl_q_s = label_phase2(
            p2_blocks, float(args.p2_short_q_max_s), int(args.p2_short_q_max_words)
        )
        p2_dur = max(0.0, float(fb_start) - float(pres_start))

        p1 = score_components(
            tool=tool,
            blocks=p1_blocks,
            labels=p1_labels,
            window_duration_s=p1_dur,
            fillers=fillers,
            pause_threshold_s=float(args.pause_threshold_s),
            allowlist=allowlist,
            target_label="assistant",
        ) if p1_dur > 1 else {
            "grammar_per_100w": 0.0, "language_quality": 0.0, "clarity": 0.0, "fluency": 0.0,
            "filler_per_100w": 0.0, "long_pauses_per_min": 0.0, "overall": 0.0,
        }

        p2 = score_components(
            tool=tool,
            blocks=p2_blocks,
            labels=p2_labels,
            window_duration_s=p2_dur,
            fillers=fillers,
            pause_threshold_s=float(args.pause_threshold_s),
            allowlist=allowlist,
            target_label="assistant",
        ) if p2_dur > 1 else {
            "grammar_per_100w": 0.0, "language_quality": 0.0, "clarity": 0.0, "fluency": 0.0,
            "filler_per_100w": 0.0, "long_pauses_per_min": 0.0, "overall": 0.0,
        }

        # phase blend
        w1 = float(args.p1_weight)
        w2 = float(args.p2_weight)
        denom = (w1 + w2) if (w1 + w2) > 0 else 1.0
        srt_assistant_overall = (w1 * float(p1["overall"]) + w2 * float(p2["overall"])) / denom

        # optional audio blend (if audio metrics exist)
        audio_weight = clamp(float(args.audio_weight), 0.0, 0.35)
        audio_component = None
        if fn in audio_map:
            aq = audio_map[fn].get("audio_quality_score", 0.0)
            fl = audio_map[fn].get("fluency_score", 0.0)
            audio_component = clamp(0.60 * aq + 0.40 * fl, 0.0, 100.0)

        assistant_overall = float(srt_assistant_overall)
        if audio_component is not None and audio_weight > 0:
            assistant_overall = (1.0 - audio_weight) * assistant_overall + audio_weight * float(audio_component)

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
        # de-dup by filename keep last (new)
        merged = merged.drop_duplicates(subset=["filename"], keep="last")
    else:
        merged = new_df

    merged = merged.sort_values(by=["assistant_overall_score", "filename"], ascending=[False, True])

    out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_metrics_csv, index=False)

    minimal = [
        {"filename": r["filename"], "assistant_overall_score": float(r["assistant_overall_score"])}
        for r in merged.to_dict(orient="records")
    ]
    Path(args.out_scores_json).write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_scores_full_json).write_text(json.dumps(merged.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {out_metrics_csv}")
    print(f"Wrote: {args.out_scores_json}")
    print(f"Wrote: {args.out_scores_full_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
