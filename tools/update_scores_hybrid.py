# tools/update_scores_hybrid.py
"""
SRT tabanlı "assistant performance" skorlayıcı.

Ne yapar?
- Internet Archive item içindeki audio dosyalarını listeler (identifier).
- FSPtranskript repo'sundaki .srt dosyalarıyla filename üzerinden eşleştirir.
- SRT metin + zaman damgalarından:
  - presentation başlangıcını (Ich habe eine(n) neue(n) Patient(in) ... / darf ich den Fall vorstellen ...)
  - feedback başlangıcını (Rückmeldung / Feedback ...)
  heuristikle bulur.
- Phase-1 (anamnez: 0..anamnesis_end_s) ve Phase-2 (presentation: presentation..feedback)
  için asistan konuşması olduğunu düşündüğü cümleleri seçer.
- LanguageTool ile (spell-check ağırlığını azaltıp) grammar hatalarını sayar.
- Dolgu (äh/ähm/...) + pause (SRT gap) metrikleri çıkarır.
- public/assistant_scores.csv ve public/assistant_scores.json üretir.

Not:
- Ses kalitesi (LUFS/SNR) burada yok; bu script asistanın dil performansına odaklıdır.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tqdm import tqdm

try:
    import language_tool_python
except Exception as exc:  # pragma: no cover
    language_tool_python = None  # type: ignore


SCHEMA_VERSION = 6

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".mp4"}
DEFAULT_ANAMNESIS_END_S = 1200.0

FILLERS = {
    "äh",
    "aeh",
    "ähm",
    "aehm",
    "hm",
    "hmm",
    "mmm",
    "also",
    "so",
    "naja",
    "quasi",
    "irgendwie",
    "halt",
    "ok",
    "okay",
    "ja",
    "genau",
}

PRESENTATION_PATTERNS = [
    r"\bich habe (?:einen|eine|nen|ne)\s+(?:neuen|neue|neu)\s+patient(?:in|en)?\b",
    r"\bdarf ich (?:ihnen|dir)\s+(?:kurz\s+)?den fall vorstellen\b",
    r"\bhaben sie kurz zeit\b",
    r"\bich (?:möchte|wollte)\s+(?:ihnen|dir)\s+den fall vorstellen\b",
    r"\bkurze fallvorstellung\b",
]

FEEDBACK_PATTERNS = [
    r"\brückmeldung\b",
    r"\bfeedback\b",
    r"\bwas war (?:gut|nicht gut)\b",
    r"\bverbesser(?:n|ung)\b",
    r"\bkritik\b",
    r"\bgut gemacht\b",
    r"\bbitte (?:noch )?mal\b.*\bzusammenfass\b",
]

# "Oberarzt soru soruyor" kısmında kısa soru cümlelerini p2'den atmak için:
SHORT_QUESTION_PATTERNS = [
    r"^\s*(und|wie|was|wann|wo|warum|wieso|weshalb|welche|welcher|welches)\b",
    r"\bhaben sie\b",
    r"\bhat (?:der|die|das)\b",
    r"\bgibt es\b",
    r"\bsind sie\b",
    r"\bkönnen sie\b",
    r"\bdürfen sie\b",
]


@dataclass(frozen=True)
class SrtCue:
    start_s: float
    end_s: float
    text: str

    @property
    def dur_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_umlauts(s: str) -> str:
    return (
        s.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "ae")
        .replace("Ö", "oe")
        .replace("Ü", "ue")
        .replace("ß", "ss")
    )


def norm_key(s: str) -> str:
    s = _normalize_umlauts(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ._-]+", "", s)
    return s


_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})"
)


def parse_srt_time(t: str) -> float:
    m = _TIME_RE.search(t.strip())
    if not m:
        raise ValueError(f"Bad SRT time: {t}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    se = int(m.group("s"))
    ms = int(m.group("ms"))
    return h * 3600 + mi * 60 + se + ms / 1000.0


def parse_srt(path: Path) -> list[SrtCue]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in raw.splitlines()]

    cues: list[SrtCue] = []
    i = 0
    while i < len(lines):
        # index line (optional)
        if lines[i].strip().isdigit():
            i += 1
        if i >= len(lines):
            break

        if "-->" not in lines[i]:
            i += 1
            continue

        time_line = lines[i].strip()
        i += 1
        try:
            left, right = [x.strip() for x in time_line.split("-->")]
            start_s = parse_srt_time(left)
            end_s = parse_srt_time(right.split()[0].strip())
        except Exception:
            continue

        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i].strip())
            i += 1
        i += 1  # blank

        text = " ".join(text_lines).strip()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            cues.append(SrtCue(start_s=start_s, end_s=end_s, text=text))

    cues.sort(key=lambda c: (c.start_s, c.end_s))
    return cues


def fetch_ia_audio_files(identifier: str, timeout_s: int = 30) -> list[dict[str, Any]]:
    url = f"https://archive.org/metadata/{identifier}"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    meta = r.json()

    files = meta.get("files", [])
    out: list[dict[str, Any]] = []
    for f in files:
        name = f.get("name")
        if not name or not isinstance(name, str):
            continue
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS:
            continue
        mtime = f.get("mtime")
        try:
            mtime_i = int(mtime) if mtime is not None else 0
        except Exception:
            mtime_i = 0
        out.append(
            {
                "name": name,
                "ext": ext,
                "mtime": mtime_i,
            }
        )

    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def build_srt_index(transcripts_dir: Path) -> dict[str, Path]:
    if not transcripts_dir.exists():
        _die(f"transcripts_dir not found: {transcripts_dir}")

    srts = list(transcripts_dir.rglob("*.srt"))
    idx: dict[str, Path] = {}
    for p in srts:
        idx[norm_key(p.name)] = p
        idx[norm_key(p.stem)] = p
    return idx


def match_srt_for_audio(
    audio_name: str, srt_index: dict[str, Path], min_score: int = 90
) -> Optional[Path]:
    # try exact keys
    candidates = [
        norm_key(audio_name),
        norm_key(Path(audio_name).stem),
        norm_key(Path(audio_name).name + ".srt"),
        norm_key(Path(audio_name).stem + ".srt"),
    ]
    for k in candidates:
        if k in srt_index:
            return srt_index[k]

    # fuzzy over index keys
    keys = list(srt_index.keys())
    q = norm_key(audio_name)
    best = process.extractOne(q, keys, scorer=fuzz.WRatio)
    if not best:
        return None
    key, score, _ = best
    if int(score) < min_score:
        return None
    return srt_index.get(key)


def detect_first_match_time(
    cues: list[SrtCue],
    patterns: list[str],
    min_time_s: float = 0.0,
    max_time_s: Optional[float] = None,
) -> Optional[float]:
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    for c in cues:
        if c.start_s < min_time_s:
            continue
        if max_time_s is not None and c.start_s > max_time_s:
            continue
        t = norm_key(c.text)
        for rg in regs:
            if rg.search(t):
                return c.start_s
    return None


def cue_in_window(c: SrtCue, start_s: float, end_s: float) -> bool:
    return (c.end_s > start_s) and (c.start_s < end_s)


_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+(?:'[A-Za-zÄÖÜäöüß]+)?")


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def count_fillers(words: list[str]) -> int:
    return sum(1 for w in words if norm_key(w) in FILLERS)


def weird_token_rate(words: list[str]) -> float:
    if not words:
        return 0.0
    weird = 0
    for w in words:
        w0 = w.strip()
        if re.fullmatch(r"[0-9]+", w0):
            continue
        if re.fullmatch(r"[A-Za-zÄÖÜäöüß]+", w0):
            continue
        weird += 1
    return weird / max(1, len(words))


def repetition_rate(words: list[str]) -> float:
    if len(words) < 6:
        return 0.0
    ws = [norm_key(w) for w in words]
    bigrams = list(zip(ws, ws[1:]))
    if not bigrams:
        return 0.0
    freq: dict[tuple[str, str], int] = {}
    for b in bigrams:
        freq[b] = freq.get(b, 0) + 1
    rep = sum(1 for v in freq.values() if v >= 3)
    return rep / max(1, len(freq))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_from_rate(rate: float, good: float, bad: float) -> float:
    # <=good -> 1, >=bad -> 0
    if bad <= good:
        return 1.0
    if rate <= good:
        return 1.0
    if rate >= bad:
        return 0.0
    return 1.0 - (rate - good) / (bad - good)


def srt_long_pauses_per_min(
    cues: list[SrtCue],
    window_start_s: float,
    window_end_s: float,
    pause_thresh_s: float = 0.5,
) -> float:
    in_cues = [c for c in cues if cue_in_window(c, window_start_s, window_end_s)]
    in_cues.sort(key=lambda c: (c.start_s, c.end_s))
    if len(in_cues) < 2:
        return 0.0

    long_pauses = 0
    for a, b in zip(in_cues, in_cues[1:]):
        gap = max(0.0, b.start_s - a.end_s)
        if gap >= pause_thresh_s:
            long_pauses += 1

    dur = max(1e-9, window_end_s - window_start_s)
    return long_pauses / (dur / 60.0)


def is_assistant_like_p1(text: str) -> bool:
    t = norm_key(text)
    # doctor-ish signals
    doctor_hits = 0
    if "?" in text:
        doctor_hits += 2
    if re.search(r"\b(k[oö]nnen|d[uü]rfen|m[oö]chten)\s+sie\b", t):
        doctor_hits += 2
    if re.search(r"\bwie\b|\bwas\b|\bwann\b|\bwo\b|\bwieviel\b|\bwelche\b", t):
        doctor_hits += 1
    if re.search(r"\bhaben sie\b|\bsind sie\b|\bnehmen sie\b", t):
        doctor_hits += 1
    if re.search(r"\bich bin\b.*\bassist", t):
        doctor_hits += 2

    # patient-ish signals
    patient_hits = 0
    if re.search(r"\bich\b|\bmir\b|\bmich\b|\bmein\b|\bmeine\b", t):
        patient_hits += 1
    if re.search(r"\bschmerz|\bueb(?:el|lig)|\berbroch|\bluftnot|\bschwindel", t):
        patient_hits += 1

    # short utterances lean assistant
    wc = len(tokenize_words(text))
    if wc <= 8:
        doctor_hits += 1

    return doctor_hits >= max(1, patient_hits)


def is_short_question_p2(text: str, max_words: int = 8) -> bool:
    wc = len(tokenize_words(text))
    if wc == 0:
        return False
    t = text.strip()
    tn = norm_key(t)
    if wc <= max_words and ("?" in t):
        return True
    if wc <= max_words:
        for p in SHORT_QUESTION_PATTERNS:
            if re.search(p, tn, flags=re.IGNORECASE):
                return True
    return False


def is_assistant_like_p2(text: str) -> bool:
    # Presentation/answer tends to be longer, not short question
    wc = len(tokenize_words(text))
    if wc >= 9 and not is_short_question_p2(text):
        return True
    # also allow key opening
    if detect_first_match_time([SrtCue(0, 0, text)], PRESENTATION_PATTERNS) is not None:
        return True
    return False


def language_tool_counts(
    tool: Any,
    text: str,
    ignore_spelling: bool = True,
) -> tuple[int, int]:
    """
    Returns (grammar_like_count, total_matches)
    ignore_spelling=True -> misspelling/typographical ağırlığını düşürür (medikal terimler için).
    """
    if not text.strip():
        return 0, 0

    # LanguageTool çok uzun metinde yavaşlayabiliyor: chunk
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sent:
            continue
        if cur_len + len(sent) > 2000 and cur:
            chunks.append(" ".join(cur))
            cur = [sent]
            cur_len = len(sent)
        else:
            cur.append(sent)
            cur_len += len(sent) + 1
    if cur:
        chunks.append(" ".join(cur))

    grammar = 0
    total = 0
    for ch in chunks:
        matches = tool.check(ch)
        total += len(matches)
        for m in matches:
            issue = getattr(m, "ruleIssueType", "") or ""
            rule_id = getattr(m, "ruleId", "") or ""
            # Why: medikal Fachbegriffe ve isimler spelling tetikler -> grammar skorunu bozmasın
            if ignore_spelling and issue in {"misspelling", "typographical"}:
                continue
            if ignore_spelling and "SPELLER" in rule_id.upper():
                continue
            if ignore_spelling and "MORFOLOGIK_RULE" in rule_id.upper():
                continue
            grammar += 1

    return grammar, total


def compute_phase_metrics(
    cues: list[SrtCue],
    window_start_s: float,
    window_end_s: float,
    assistant_selector,
    tool: Any,
    exclude_short_q: bool = False,
) -> dict[str, float]:
    in_cues = [c for c in cues if cue_in_window(c, window_start_s, window_end_s)]
    in_cues.sort(key=lambda c: (c.start_s, c.end_s))

    excl_q_count = 0
    excl_q_s = 0.0

    assistant_cues: list[SrtCue] = []
    for c in in_cues:
        if exclude_short_q and is_short_question_p2(c.text):
            excl_q_count += 1
            excl_q_s += c.dur_s
            continue
        if assistant_selector(c.text):
            assistant_cues.append(c)

    assistant_text = " ".join(c.text for c in assistant_cues).strip()
    words = tokenize_words(assistant_text)

    word_count = len(words)
    filler_count = count_fillers(words)
    filler_per_100w = (filler_count / max(1, word_count)) * 100.0

    long_pauses_per_min = srt_long_pauses_per_min(in_cues, window_start_s, window_end_s)

    rep_rate = repetition_rate(words)
    weird_rate = weird_token_rate(words)

    assistant_duration_s = sum(c.dur_s for c in assistant_cues)

    grammar_per_100w = 0.0
    if tool is not None and word_count >= 30:
        grammar_like, _total = language_tool_counts(tool, assistant_text, ignore_spelling=True)
        grammar_per_100w = (grammar_like / max(1, word_count)) * 100.0

    # --- scoring (0..100)
    grammar_score = score_from_rate(grammar_per_100w, good=0.5, bad=5.0)
    weird_score = score_from_rate(weird_rate, good=0.01, bad=0.06)
    language_quality = 100.0 * (0.7 * grammar_score + 0.3 * weird_score)

    filler_score = score_from_rate(filler_per_100w, good=1.0, bad=6.0)
    pause_score = score_from_rate(long_pauses_per_min, good=2.0, bad=10.0)
    fluency = 100.0 * (0.55 * filler_score + 0.45 * pause_score)

    rep_score = score_from_rate(rep_rate, good=0.02, bad=0.12)
    # Basit clarity: tekrar + weird + aşırı kısa/uzun konuşma dengesi
    dur_penalty = 0.0
    if assistant_duration_s < 10.0 and word_count < 30:
        dur_penalty = 0.25
    clarity = 100.0 * clamp01(0.55 * rep_score + 0.45 * weird_score - dur_penalty)

    overall = 0.45 * language_quality + 0.25 * clarity + 0.30 * fluency

    return {
        "overall": float(overall),
        "duration_s": float(assistant_duration_s),
        "grammar_per_100w": float(grammar_per_100w),
        "language_quality": float(language_quality),
        "clarity": float(clarity),
        "fluency": float(fluency),
        "filler_per_100w": float(filler_per_100w),
        "long_pauses_per_min": float(long_pauses_per_min),
        "excl_short_q_count": float(excl_q_count),
        "excl_short_q_s": float(excl_q_s),
    }


def load_existing_filenames(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            return set()
        return set(str(x) for x in df["filename"].dropna().tolist())
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True, help="Internet Archive identifier")
    ap.add_argument(
        "--transcripts_dir",
        default="transcripts_repo/transcripts",
        help="FSPtranskript repo transcripts path",
    )
    ap.add_argument("--anamnesis_end_s", type=float, default=DEFAULT_ANAMNESIS_END_S)
    ap.add_argument("--max_new_files", type=int, default=10)
    ap.add_argument("--timeout_s", type=int, default=30)
    ap.add_argument("--min_srt_match", type=int, default=90)
    ap.add_argument("--public_dir", default="public")
    ap.add_argument("--force_recompute", action="store_true")
    args = ap.parse_args()

    public_dir = Path(args.public_dir)
    out_csv = public_dir / "assistant_scores.csv"
    out_json = public_dir / "assistant_scores.json"

    transcripts_dir = Path(args.transcripts_dir)
    srt_index = build_srt_index(transcripts_dir)

    audio_files = fetch_ia_audio_files(args.identifier, timeout_s=args.timeout_s)
    if not audio_files:
        _die(f"No audio files found in IA item: {args.identifier}")

    existing = set()
    if not args.force_recompute:
        existing = load_existing_filenames(out_csv)

    todo: list[dict[str, Any]] = []
    for f in audio_files:
        if (not args.force_recompute) and (f["name"] in existing):
            continue
        todo.append(f)
        if len(todo) >= int(args.max_new_files):
            break

    # If nothing new, still ensure json exists (sorted copy of csv)
    if not todo and out_csv.exists():
        df = pd.read_csv(out_csv)
        df = df.sort_values("assistant_overall_score", ascending=False)
        df.to_csv(out_csv, index=False)
        _write_json(out_json, df.to_dict(orient="records"))
        print(f"[OK] No new files. Refreshed sort: {out_csv}")
        return 0

    # LanguageTool init
    tool = None
    if language_tool_python is None:
        print("[WARN] language_tool_python import failed; grammar metrics will be 0.")
    else:
        try:
            tool = language_tool_python.LanguageTool("de-DE")
        except Exception as exc:
            print(f"[WARN] LanguageTool init failed: {exc}. grammar metrics will be 0.")
            tool = None

    rows: list[dict[str, Any]] = []

    # include previous rows if exist
    if out_csv.exists() and not args.force_recompute:
        try:
            prev = pd.read_csv(out_csv).to_dict(orient="records")
            rows.extend(prev)
        except Exception:
            pass

    for f in tqdm(todo, desc="Scoring"):
        name = f["name"]
        srt_path = match_srt_for_audio(name, srt_index, min_score=int(args.min_srt_match))
        if not srt_path:
            print(f"[WARN] SRT not found for: {name}")
            continue

        cues = parse_srt(srt_path)
        if not cues:
            print(f"[WARN] Empty/invalid SRT for: {name} ({srt_path})")
            continue

        audio_dur_s = max(c.end_s for c in cues)

        presentation_start = detect_first_match_time(
            cues,
            PRESENTATION_PATTERNS,
            min_time_s=60.0,
            max_time_s=max(60.0, audio_dur_s),
        )
        presentation_start_s = float(presentation_start) if presentation_start is not None else float(args.anamnesis_end_s)

        feedback_start = detect_first_match_time(
            cues,
            FEEDBACK_PATTERNS,
            min_time_s=max(0.0, presentation_start_s + 15.0),
            max_time_s=max(0.0, audio_dur_s),
        )
        feedback_start_s = float(feedback_start) if feedback_start is not None else float(audio_dur_s)

        anamnesis_end_s = float(args.anamnesis_end_s)

        # windows
        p1_start = 0.0
        p1_end = min(anamnesis_end_s, max(0.0, presentation_start_s))
        if p1_end <= p1_start:
            p1_end = anamnesis_end_s

        p2_start = max(0.0, presentation_start_s)
        p2_end = max(p2_start, min(feedback_start_s, audio_dur_s))

        p1 = compute_phase_metrics(
            cues=cues,
            window_start_s=p1_start,
            window_end_s=p1_end,
            assistant_selector=is_assistant_like_p1,
            tool=tool,
            exclude_short_q=False,
        )

        p2 = compute_phase_metrics(
            cues=cues,
            window_start_s=p2_start,
            window_end_s=p2_end,
            assistant_selector=is_assistant_like_p2,
            tool=tool,
            exclude_short_q=True,
        )

        # combine: p1 ağırlık yüksek (anamnez standardı)
        if p2["duration_s"] <= 0.5:
            assistant_overall = p1["overall"]
        else:
            assistant_overall = 0.6 * p1["overall"] + 0.4 * p2["overall"]

        row = {
            "filename": name,
            "assistant_overall_score": float(assistant_overall),
            "anamnesis_end_s": float(anamnesis_end_s),
            "presentation_start_s": float(presentation_start_s),
            "feedback_start_s": float(feedback_start_s),
            "p1_overall": float(p1["overall"]),
            "p1_duration_s": float(p1["duration_s"]),
            "p1_grammar_per_100w": float(p1["grammar_per_100w"]),
            "p1_language_quality": float(p1["language_quality"]),
            "p1_clarity": float(p1["clarity"]),
            "p1_fluency": float(p1["fluency"]),
            "p1_filler_per_100w": float(p1["filler_per_100w"]),
            "p1_long_pauses_per_min": float(p1["long_pauses_per_min"]),
            "p2_overall": float(p2["overall"]),
            "p2_duration_s": float(p2["duration_s"]),
            "p2_grammar_per_100w": float(p2["grammar_per_100w"]),
            "p2_language_quality": float(p2["language_quality"]),
            "p2_clarity": float(p2["clarity"]),
            "p2_fluency": float(p2["fluency"]),
            "p2_filler_per_100w": float(p2["filler_per_100w"]),
            "p2_long_pauses_per_min": float(p2["long_pauses_per_min"]),
            "p2_excl_short_q_count": int(p2["excl_short_q_count"]),
            "p2_excl_short_q_s": float(p2["excl_short_q_s"]),
            "schema_version": int(SCHEMA_VERSION),
        }
        rows.append(row)

    if tool is not None:
        try:
            tool.close()
        except Exception:
            pass

    if not rows:
        _die("No rows produced (SRT matching failed?).")

    df = pd.DataFrame(rows)
    # de-dup by filename, keep best schema/newest row (last)
    df = df.drop_duplicates(subset=["filename"], keep="last")
    df = df.sort_values("assistant_overall_score", ascending=False)

    public_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    _write_json(out_json, df.to_dict(orient="records"))

    # Ayrıca site tarafı isterse diye "scores.json" aynası
    _write_json(public_dir / "scores.json", df[["filename", "assistant_overall_score"]].to_dict(orient="records"))

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
