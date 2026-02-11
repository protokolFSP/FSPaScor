"""
Assistant-only scoring for German FSP-style dialog recordings (medical).

What this version adds:
- LanguageTool grammar proxy (grammar_errors_per_100w) with medical allowlist filtering.
- Phase2 Oberarzt short-question filter (exclude short question-like segments from assistant Phase2).
- Final assistant score focuses on language performance:
  final_phase = 0.50*language_quality + 0.30*fluency + 0.20*clarity (defaults, configurable).
- Outputs:
  public/scores.json, public/scores.full.json, public/metrics.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import requests
import webrtcvad
from faster_whisper import WhisperModel
from tqdm import tqdm

EPS = 1e-12
SR = 16000


# -------------------------- helpers --------------------------


@dataclass(frozen=True)
class Scores:
    audio_quality: float
    fluency: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def ffmpeg_make_clip_wav(in_path: Path, out_wav: Path, clip_seconds: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(in_path),
        "-t",
        str(int(clip_seconds)),
        "-ac",
        "1",
        "-ar",
        str(SR),
        str(out_wav),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:800])


def ffmpeg_decode_pcm16_mono_16k(path: Path) -> bytes:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:800])
    return proc.stdout


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return x / 32768.0


def frame_generator(pcm16: bytes, sample_rate: int, frame_ms: int = 30) -> Iterable[bytes]:
    frame_len = int(sample_rate * frame_ms / 1000)
    nbytes = frame_len * 2
    for i in range(0, len(pcm16) - nbytes + 1, nbytes):
        yield pcm16[i : i + nbytes]


def vad_flags(pcm16: bytes, sample_rate: int = SR, aggressiveness: int = 3, frame_ms: int = 30) -> List[bool]:
    vad = webrtcvad.Vad(aggressiveness)
    return [vad.is_speech(fr, sample_rate) for fr in frame_generator(pcm16, sample_rate, frame_ms)]


def rle_segments(flags: List[bool], frame_s: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if not flags:
        return [], []
    speech, silence = [], []
    cur = flags[0]
    start = 0
    for i in range(1, len(flags)):
        if flags[i] != cur:
            seg = (start * frame_s, i * frame_s)
            (speech if cur else silence).append(seg)
            cur = flags[i]
            start = i
    seg = (start * frame_s, len(flags) * frame_s)
    (speech if cur else silence).append(seg)
    return speech, silence


def calc_lufs(x: np.ndarray, sr: int) -> Optional[float]:
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return None


def text_tokens(text: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+", (text or "").lower())


def repetition_rate(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    rep = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return rep / (len(tokens) - 1)


def weird_token_rate(text: str) -> float:
    raw = re.findall(r"\S+", (text or "").strip())
    if not raw:
        return 0.0
    weird = sum(1 for t in raw if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]", t) is None)
    return weird / len(raw)


def build_filler_regex(extra: Optional[str] = None) -> re.Pattern:
    fillers = [
        r"e+e+e+",
        r"a+a+a+",
        r"u+h+",
        r"u+m+",
        r"ä+h+",
        r"äh+m+",
        r"hm+",
        r"mhm+",
        r"mmm+",
        r"also",
        r"halt",
        r"eben",
        r"sozusagen",
        r"genau",
        r"okay",
        r"ok",
        r"quasi",
        r"irgendwie",
        r"naja",
        r"tja",
    ]
    if extra:
        for t in [x.strip().lower() for x in extra.split(",") if x.strip()]:
            fillers.append(re.escape(t))
    pat = r"\b(" + "|".join(fillers) + r")\b"
    return re.compile(pat, flags=re.IGNORECASE)


def loudness_score(lufs: Optional[float], target: float = -20.0) -> float:
    if lufs is None or not math.isfinite(lufs):
        return 0.5
    dist = abs(lufs - target)
    return 1.0 - clamp01(dist / 20.0)


def compute_base_scores(
    snr_db: Optional[float],
    clipping_ratio: float,
    lufs: Optional[float],
    pause_ratio: float,
    filler_per_100w: float,
    long_pauses_per_min: float,
    repetition: float,
    weird_rate: float,
    asr_conf: float,
    audio_weight: float,
    fluency_weight: float,
) -> Scores:
    snr_n = 0.0 if snr_db is None or not math.isfinite(snr_db) else clamp01(max(0.0, min(30.0, snr_db)) / 30.0)
    clip_n = 1.0 - clamp01(clipping_ratio / 0.01)
    loud_n = loudness_score(lufs)
    pause_n = 1.0 - clamp01(pause_ratio / 0.50)
    asr_n = clamp01(asr_conf)

    filler_n = 1.0 - clamp01(filler_per_100w / 10.0)
    longp_n = 1.0 - clamp01(long_pauses_per_min / 10.0)
    rep_n = 1.0 - clamp01(repetition / 0.20)
    weird_n = 1.0 - clamp01(weird_rate / 0.30)

    audio_quality = (0.35 * snr_n + 0.20 * clip_n + 0.15 * loud_n + 0.20 * pause_n + 0.10 * asr_n) * 100.0
    fluency = (0.40 * filler_n + 0.25 * longp_n + 0.20 * rep_n + 0.15 * weird_n) * 100.0

    audio_quality = audio_weight * audio_quality + (1.0 - audio_weight) * audio_quality
    fluency = fluency_weight * fluency + (1.0 - fluency_weight) * fluency
    return Scores(audio_quality=float(audio_quality), fluency=float(fluency))


# -------------------------- role patterns --------------------------


PHASE1_ASSISTANT_PATTERNS = [
    r"\baufnahme(gespräch|gespraech)\b",
    r"\bich bin\b",
    r"\bmein name ist\b",
    r"\bassistenz(arzt|ärztin)\b",
    r"\bich würde gerne\b",
    r"\bkönnten sie\b",
    r"\bkönnen sie\b",
    r"\bwie heißen sie\b",
    r"\bwie alt sind sie\b",
    r"\bwie groß sind sie\b",
    r"\bwie viel wiegen sie\b",
    r"\bwas führt sie zu uns\b",
    r"\bwas kann ich für sie tun\b",
    r"\bseit wann\b",
    r"\bhaben sie\b",
    r"\bnehmen sie\b",
    r"\bskala von\b",
    r"\bist das in ordnung\b",
]

PRESENTATION_START_PATTERNS = [
    r"\bich habe (einen|eine)\s+(neu(en|e)|neuen|neue|neuer|neues)\s+(patient(en)?|patientin)\b",
    r"\bich habe (einen|eine)\s+(patient(en)?|patientin)\b",
    r"\bhaben\s+sie\s+(kurz\s+)?zeit\b",
    r"\bhätten\s+sie\s+(kurz\s+)?zeit\b",
    r"\b(darf|kann)\s+ich\s+(kurz\s+)?(den\s+)?fall\s+vorstellen\b",
    r"\b(darf|kann)\s+ich\s+(ihnen|euch)\s+(kurz\s+)?(den\s+)?fall\s+vorstellen\b",
    r"\bich (möchte|würde)\s+(ihnen|euch)\s+(kurz\s+)?(den\s+)?fall\s+vorstellen\b",
    r"\bich stelle\s+(ihnen|euch)\s+(kurz\s+)?(einen|eine)\s+(patient(en)?|patientin)\s+vor\b",
]

PHASE2_REPORT_PATTERNS = [
    r"\bzusammenfassung\b",
    r"\banamnese\b",
    r"\bvorerkrankungen\b",
    r"\bmedikation\b",
    r"\ballergien\b",
    r"\bbefund\b",
    r"\bdiagnose\b",
    r"\bplan\b",
    r"\btherapie\b",
    r"\bvorstellung\b",
    r"\bder patient\b",
    r"\bdie patientin\b",
    r"\bwir haben\b",
    *PRESENTATION_START_PATTERNS,
]

FEEDBACK_STRONG_PATTERNS = [
    r"\brückmeldung\b",
    r"\bfeedback\b",
    r"\b(gesamt|kurz)\s+fazit\b",
    r"\bzusammengefasst\b",
    r"\bgut gemacht\b",
    r"\bsehr gut\b",
    r"\bprima\b",
]

FEEDBACK_MEDIUM_PATTERNS = [
    r"\bverbessern\b",
    r"\bverbesserung\b",
    r"\boptimieren\b",
    r"\bachten sie\b",
    r"\bin zukunft\b",
    r"\bsie hätten\b",
    r"\bsie sollten\b",
]

PATIENT_LIKE_PATTERNS = [
    r"\bich habe\b",
    r"\bmir ist\b",
    r"\bes tut\b",
    r"\bschmerzen\b",
    r"\bübel\b",
    r"\berbrochen\b",
    r"\bich bin\b.*\bgekommen\b",
]


STRUCTURE_MARKERS = [
    "zunächst",
    "anschließend",
    "außerdem",
    "jedoch",
    "hingegen",
    "daher",
    "deshalb",
    "somit",
    "zusammenfassend",
    "insgesamt",
    "aktuell",
]


QUESTION_HINTS = [
    r"\bwarum\b",
    r"\bwieso\b",
    r"\bworan\b",
    r"\bwie\b",
    r"\bwas\b",
    r"\bwelche\b",
    r"\bwelcher\b",
    r"\bwann\b",
    r"\bwo\b",
    r"\bkönnen sie\b",
    r"\bkönnten sie\b",
    r"\bhaben sie\b",
    r"\bdarf ich\b",
]


def _count_matches(text: str, patterns: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in patterns if re.search(p, t))


def _any_match(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


def is_question_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    return _count_matches(t, QUESTION_HINTS) > 0


def is_assistant_phase1(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    a = _count_matches(t, PHASE1_ASSISTANT_PATTERNS)
    p = _count_matches(t, PATIENT_LIKE_PATTERNS)
    if is_question_like(t):
        a += 2
    if a == 0 and p == 0 and len(t) <= 10:
        a = 1
    return a >= max(1, p)


def is_assistant_phase2_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    strong = 5 if _any_match(t, PRESENTATION_START_PATTERNS) else 0
    a = strong + _count_matches(t, PHASE2_REPORT_PATTERNS) * 2 + _count_matches(t, PHASE1_ASSISTANT_PATTERNS)
    p = _count_matches(t, PATIENT_LIKE_PATTERNS)
    return a >= max(1, p + 1)


# -------------------------- ASR --------------------------


def transcribe_segments(model: WhisperModel, path: Path, language: Optional[str]) -> Dict[str, Any]:
    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segs: List[Dict[str, Any]] = []
    texts: List[str] = []
    logps: List[float] = []

    for seg in segments_iter:
        txt = (seg.text or "").strip()
        segs.append(
            {
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": txt,
                "avg_logprob": float(getattr(seg, "avg_logprob", float("nan"))),
            }
        )
        if txt:
            texts.append(txt)
        if getattr(seg, "avg_logprob", None) is not None:
            logps.append(float(seg.avg_logprob))

    transcript = " ".join(texts).strip()
    avg_logprob_mean = float(np.mean(logps)) if logps else float("nan")
    asr_conf = float(math.exp(min(0.0, avg_logprob_mean))) if math.isfinite(avg_logprob_mean) else 0.0

    return {
        "segments": segs,
        "transcript": transcript,
        "asr_conf": asr_conf,
        "asr_language": getattr(info, "language", None),
        "asr_language_prob": getattr(info, "language_probability", None),
        "avg_logprob_mean": avg_logprob_mean,
    }


def merge_intervals(intervals: List[Tuple[float, float]], gap_s: float = 0.25, min_dur_s: float = 0.15) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + gap_s:
            cur_e = max(cur_e, e)
        else:
            if cur_e - cur_s >= min_dur_s:
                out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_e - cur_s >= min_dur_s:
        out.append((cur_s, cur_e))
    return out


def slice_concat_audio(x: np.ndarray, sr: int, intervals: List[Tuple[float, float]]) -> np.ndarray:
    if x.size == 0 or not intervals:
        return np.zeros(0, dtype=np.float32)
    parts = []
    n = x.shape[0]
    for s, e in intervals:
        a = max(0, min(n, int(s * sr)))
        b = max(0, min(n, int(e * sr)))
        if b > a:
            parts.append(x[a:b])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


# -------------------------- boundary detection --------------------------


def detect_presentation_start(
    segs: List[Dict[str, Any]],
    after_s: float,
    window_s: float,
    threshold: float,
) -> Optional[float]:
    for seg in segs:
        s = float(seg["start"])
        if s < after_s:
            continue
        t = str(seg.get("text") or "")
        if _any_match(t, PRESENTATION_START_PATTERNS):
            return s

    pts: List[Tuple[float, float]] = []
    for seg in segs:
        s = float(seg["start"])
        if s < after_s:
            continue
        t = str(seg.get("text") or "")
        score = float(_count_matches(t, PHASE2_REPORT_PATTERNS) * 2 + _count_matches(t, PHASE1_ASSISTANT_PATTERNS))
        if score <= 0:
            continue
        pts.append((s, score))

    if not pts:
        return None

    pts.sort()
    j = 0
    acc = 0.0
    for i in range(len(pts)):
        si, sc = pts[i]
        acc += sc
        while j <= i and pts[j][0] < si - window_s:
            acc -= pts[j][1]
            j += 1
        avg = acc / max(1.0, window_s)
        if avg >= threshold:
            return si
    return None


def detect_feedback_start(
    segs: List[Dict[str, Any]],
    audio_end_s: float,
    min_start_s: float,
    lookback_s: float,
    window_s: float,
    threshold: float,
    min_tail_ratio: float,
) -> Optional[float]:
    region_start = max(
        float(min_start_s),
        float(audio_end_s - lookback_s),
        float(audio_end_s * min_tail_ratio),
    )

    pts: List[Tuple[float, float]] = []
    for seg in segs:
        s = float(seg["start"])
        if s < region_start:
            continue
        t = str(seg.get("text") or "")
        strong = _count_matches(t, FEEDBACK_STRONG_PATTERNS) * 4
        medium = _count_matches(t, FEEDBACK_MEDIUM_PATTERNS) * 2
        score = float(strong + medium)
        if score <= 0:
            continue
        pts.append((s, score))

    if not pts:
        return None

    pts.sort()
    j = 0
    acc = 0.0
    for i in range(len(pts)):
        si, sc = pts[i]
        acc += sc
        while j <= i and pts[j][0] < si - window_s:
            acc -= pts[j][1]
            j += 1
        avg = acc / max(1.0, window_s)
        if avg >= threshold:
            return si
    return None


# -------------------------- IA fetch --------------------------


def download_ia_file(identifier: str, filename: str, out_path: Path) -> None:
    base = f"https://archive.org/download/{identifier}/"
    url = base + quote(filename, safe="/()[],'&+;=:@$-_.!~*")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def fetch_ia_files(identifier: str) -> List[Dict[str, Any]]:
    url = f"https://archive.org/metadata/{identifier}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", []) or []
    return [f for f in files if isinstance(f, dict) and f.get("name")]


def pick_audio_candidates(files: List[Dict[str, Any]]) -> List[str]:
    exts = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4")
    picked: List[str] = []
    for f in files:
        name = str(f.get("name"))
        if not name.lower().endswith(exts):
            continue
        src = str(f.get("source", "")).lower()
        if src and src != "original":
            continue
        picked.append(name)
    return sorted(set(picked))


# -------------------------- grammar (LanguageTool) --------------------------


_SPELLING_RULE_ID_HINTS = [
    "MORFOLOGIK",
    "HUNSPELL",
    "SPELLER",
    "SPELLING",
]


def load_allowlist(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    items = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        items.add(t.lower())
    return items


def looks_medical_token(tok: str, allowlist: Set[str]) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if t.lower() in allowlist:
        return True
    if any(ch.isdigit() for ch in t):
        return True
    if t.isupper() and 2 <= len(t) <= 8:
        return True
    if re.fullmatch(r"[a-zA-Z]{1,5}\.", t):
        return True
    if re.search(r"[-/]", t):
        return True
    if t.lower() in {"mg", "g", "ml", "mmol", "mmhg", "%"}:
        return True
    return False


def init_language_tool(enabled: bool) -> Optional[Any]:
    if not enabled:
        return None
    try:
        import language_tool_python  # type: ignore

        tool = language_tool_python.LanguageTool("de-DE")
        return tool
    except Exception as e:
        print(f"[WARN] LanguageTool init failed: {e}")
        return None


def grammar_errors_per_100w(
    tool: Optional[Any],
    text: str,
    allowlist: Set[str],
    max_chars: int,
) -> Tuple[float, int, int]:
    if tool is None:
        return float("nan"), 0, 0

    t = (text or "").strip()
    if not t:
        return float("nan"), 0, 0

    t = t[: max(0, int(max_chars))]
    words = text_tokens(t)
    wc = len(words)
    if wc < 5:
        return float("nan"), 0, len(t)

    try:
        matches = tool.check(t)
    except Exception as e:
        print(f"[WARN] LanguageTool check failed: {e}")
        return float("nan"), 0, len(t)

    err = 0
    for m in matches:
        rid = str(getattr(m, "ruleId", "") or "")
        context = str(getattr(m, "context", "") or "")
        offset = int(getattr(m, "offsetInContext", 0) or 0)
        length = int(getattr(m, "errorLength", 0) or 0)

        snippet = ""
        if context and length > 0:
            snippet = context[offset : offset + length].strip()

        if not snippet:
            continue

        if any(h in rid.upper() for h in _SPELLING_RULE_ID_HINTS):
            if looks_medical_token(snippet, allowlist):
                continue
            if snippet.isupper():
                continue
            if any(ch.isdigit() for ch in snippet):
                continue
            if not re.fullmatch(r"[a-zA-ZÀ-ÖØ-öø-ÿÄÖÜäöüß\-]+", snippet):
                continue

        err += 1

    rate = (err * 100.0) / max(1, wc)
    return float(rate), int(err), int(len(t))


# -------------------------- scoring --------------------------


def structure_markers_per_100w(text: str) -> float:
    toks = text_tokens(text)
    wc = len(toks)
    if wc == 0:
        return 0.0
    joined = " " + " ".join(toks) + " "
    count = 0
    for w in STRUCTURE_MARKERS:
        count += len(re.findall(rf"\b{re.escape(w)}\b", joined))
    return float(count * 100.0 / wc)


def clarity_score(asr_conf: float, audio_quality_score: float) -> float:
    aq = clamp01(audio_quality_score / 100.0)
    expected = 0.25 + 0.65 * aq
    gap = clamp01((expected - clamp01(asr_conf)) / 0.35)
    return float((1.0 - gap) * 100.0)


def language_quality_score(
    grammar_rate_per_100w: float,
    weird_rate: float,
    structure_per_100w: float,
) -> float:
    parts: List[Tuple[float, float]] = []

    if math.isfinite(grammar_rate_per_100w):
        grammar_n = 1.0 - clamp01(grammar_rate_per_100w / 10.0)
        parts.append((0.55, grammar_n))

    weird_n = 1.0 - clamp01(weird_rate / 0.30)
    parts.append((0.25, weird_n))

    struct_n = clamp01(structure_per_100w / 5.0)
    parts.append((0.20, struct_n))

    wsum = sum(w for w, _ in parts)
    if wsum <= 0:
        return 0.0
    score = sum(w * v for w, v in parts) / wsum
    return float(score * 100.0)


def final_phase_score(
    lang_score: float,
    fluency_score: float,
    clarity: float,
    w_lang: float,
    w_flu: float,
    w_cla: float,
) -> float:
    wsum = max(EPS, w_lang + w_flu + w_cla)
    return float((w_lang * lang_score + w_flu * fluency_score + w_cla * clarity) / wsum)


def phase_metrics(
    x_phase: np.ndarray,
    transcript: str,
    asr_conf: float,
    vad_aggr: int,
    filler_extra: Optional[str],
    pause_threshold_s: float,
    audio_weight: float,
    fluency_weight: float,
    grammar_tool: Optional[Any],
    med_allowlist: Set[str],
    grammar_max_chars: int,
    total_language_weight: float,
    total_fluency_weight: float,
    total_clarity_weight: float,
) -> Dict[str, Any]:
    if x_phase.size < SR * 3:
        return {
            "duration_s": float(x_phase.size / SR) if x_phase.size else 0.0,
            "overall_score": 0.0,
            "audio_quality_score": 0.0,
            "fluency_score": 0.0,
            "language_quality_score": 0.0,
            "clarity_score": 0.0,
            "grammar_errors_per_100w": float("nan"),
            "grammar_errors_count": 0,
            "grammar_chars_analyzed": 0,
            "structure_markers_per_100w": 0.0,
            "speech_rate_wpm": 0.0,
            "articulation_rate_wpm": 0.0,
        }

    clipping_ratio = float(np.mean(np.abs(x_phase) >= 0.999))
    lufs = calc_lufs(x_phase, SR)

    pcm16_p = (np.clip(x_phase, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    flags = vad_flags(pcm16_p, sample_rate=SR, aggressiveness=vad_aggr, frame_ms=30)
    frame_s = 0.03
    _, silence_segs = rle_segments(flags, frame_s=frame_s)
    speech_ratio = float(np.mean(flags)) if flags else 0.0
    pause_ratio = 1.0 - speech_ratio

    frame_len = int(SR * 30 / 1000)
    snr_db = float("nan")
    if len(flags) > 0 and x_phase.size >= frame_len:
        nframes = min(len(flags), x_phase.size // frame_len)
        xr = x_phase[: nframes * frame_len].reshape(nframes, frame_len)
        frame_rms = np.sqrt(np.mean(xr * xr, axis=1) + EPS)
        f = np.array(flags[:nframes], dtype=bool)
        if f.any() and (~f).any():
            speech_rms = float(np.mean(frame_rms[f]))
            noise_rms = float(np.mean(frame_rms[~f]))
            snr_db = 20.0 * math.log10(speech_rms / (noise_rms + EPS))

    pauses = [end - start for (start, end) in silence_segs]
    long_pauses = sum(1 for p in pauses if p >= pause_threshold_s)
    duration_s = float(x_phase.size / SR)
    long_pauses_per_min = float(long_pauses / max(1e-6, duration_s / 60.0))

    toks = text_tokens(transcript)
    word_count = int(len(toks))

    filler_re = build_filler_regex(extra=filler_extra)
    filler_count = int(len(filler_re.findall(transcript)))
    filler_per_100w = float(filler_count * 100.0 / max(1, word_count))

    rep = float(repetition_rate(toks))
    weird = float(weird_token_rate(transcript))

    base = compute_base_scores(
        snr_db=None if not math.isfinite(snr_db) else float(snr_db),
        clipping_ratio=clipping_ratio,
        lufs=lufs,
        pause_ratio=pause_ratio,
        filler_per_100w=filler_per_100w,
        long_pauses_per_min=long_pauses_per_min,
        repetition=rep,
        weird_rate=weird,
        asr_conf=asr_conf,
        audio_weight=audio_weight,
        fluency_weight=fluency_weight,
    )

    speech_duration_s = float(duration_s * max(0.0, min(1.0, speech_ratio)))
    speech_rate_wpm = float(word_count / max(EPS, duration_s / 60.0))
    articulation_rate_wpm = float(word_count / max(EPS, speech_duration_s / 60.0))

    struct_per_100w = structure_markers_per_100w(transcript)
    g_rate, g_count, g_chars = grammar_errors_per_100w(
        tool=grammar_tool,
        text=transcript,
        allowlist=med_allowlist,
        max_chars=int(grammar_max_chars),
    )

    lang_score = language_quality_score(
        grammar_rate_per_100w=g_rate,
        weird_rate=weird,
        structure_per_100w=struct_per_100w,
    )
    cla = clarity_score(asr_conf=asr_conf, audio_quality_score=base.audio_quality)

    overall = final_phase_score(
        lang_score=lang_score,
        fluency_score=base.fluency,
        clarity=cla,
        w_lang=total_language_weight,
        w_flu=total_fluency_weight,
        w_cla=total_clarity_weight,
    )

    return {
        "duration_s": duration_s,
        "clipping_ratio": clipping_ratio,
        "lufs": lufs,
        "snr_db_est": None if not math.isfinite(snr_db) else float(snr_db),
        "speech_ratio": speech_ratio,
        "pause_ratio": pause_ratio,
        "long_pauses": int(long_pauses),
        "long_pauses_per_min": long_pauses_per_min,
        "asr_conf": float(asr_conf),
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_per_100w": filler_per_100w,
        "repetition_rate": rep,
        "weird_token_rate": weird,
        "audio_quality_score": float(base.audio_quality),
        "fluency_score": float(base.fluency),
        "grammar_errors_per_100w": float(g_rate),
        "grammar_errors_count": int(g_count),
        "grammar_chars_analyzed": int(g_chars),
        "structure_markers_per_100w": float(struct_per_100w),
        "speech_rate_wpm": float(speech_rate_wpm),
        "articulation_rate_wpm": float(articulation_rate_wpm),
        "language_quality_score": float(lang_score),
        "clarity_score": float(cla),
        "overall_score": float(overall),
    }


# -------------------------- main analysis --------------------------


def analyze_assistant_dynamic(
    wav_path: Path,
    model: WhisperModel,
    filename: str,
    language: Optional[str],
    filler_extra: Optional[str],
    vad_aggr: int,
    anamnesis_end_s: float,
    pause_threshold_s: float,
    audio_weight: float,
    fluency_weight: float,
    phase1_weight: float,
    phase2_weight: float,
    schema_version: int,
    pres_win_s: float,
    pres_threshold: float,
    fb_lookback_s: float,
    fb_win_s: float,
    fb_threshold: float,
    fb_min_tail_ratio: float,
    p2_question_max_s: float,
    p2_min_keep_s: float,
    grammar_tool: Optional[Any],
    med_allowlist: Set[str],
    grammar_max_chars: int,
    total_language_weight: float,
    total_fluency_weight: float,
    total_clarity_weight: float,
) -> Dict[str, Any]:
    pcm16 = ffmpeg_decode_pcm16_mono_16k(wav_path)
    x = pcm16_to_float32(pcm16)
    audio_end_s = float(x.size / SR)

    tr = transcribe_segments(model, wav_path, language=language)
    segs = tr["segments"]

    presentation_start = detect_presentation_start(
        segs=segs,
        after_s=min(anamnesis_end_s, audio_end_s),
        window_s=pres_win_s,
        threshold=pres_threshold,
    )
    if presentation_start is None:
        presentation_start = min(anamnesis_end_s, audio_end_s)

    feedback_start = detect_feedback_start(
        segs=segs,
        audio_end_s=audio_end_s,
        min_start_s=min(audio_end_s, float(presentation_start) + 60.0),
        lookback_s=fb_lookback_s,
        window_s=fb_win_s,
        threshold=fb_threshold,
        min_tail_ratio=fb_min_tail_ratio,
    )
    if feedback_start is None:
        feedback_start = audio_end_s

    presentation_start = float(max(0.0, min(presentation_start, audio_end_s)))
    feedback_start = float(max(presentation_start, min(feedback_start, audio_end_s)))

    p1_intervals: List[Tuple[float, float]] = []
    p2_intervals: List[Tuple[float, float]] = []
    p1_texts: List[str] = []
    p2_texts: List[str] = []
    p1_logps: List[float] = []
    p2_logps: List[float] = []

    p2_excluded_short_q_count = 0
    p2_excluded_short_q_s = 0.0

    for seg in segs:
        s = float(seg["start"])
        e = float(seg["end"])
        dur = max(0.0, e - s)
        txt = str(seg.get("text") or "").strip()
        lp = seg.get("avg_logprob")

        if s < anamnesis_end_s:
            if is_assistant_phase1(txt):
                p1_intervals.append((s, e))
                if txt:
                    p1_texts.append(txt)
                if isinstance(lp, (int, float)) and math.isfinite(lp):
                    p1_logps.append(float(lp))
        elif presentation_start <= s < feedback_start:
            has_pres_cue = _any_match(txt, PRESENTATION_START_PATTERNS)
            if dur <= float(p2_question_max_s) and is_question_like(txt) and not has_pres_cue:
                p2_excluded_short_q_count += 1
                p2_excluded_short_q_s += dur
                continue
            if dur < float(p2_min_keep_s) and not has_pres_cue and _count_matches(txt, PHASE2_REPORT_PATTERNS) == 0:
                continue
            if is_assistant_phase2_text(txt):
                p2_intervals.append((s, e))
                if txt:
                    p2_texts.append(txt)
                if isinstance(lp, (int, float)) and math.isfinite(lp):
                    p2_logps.append(float(lp))

    p1_intervals = merge_intervals(p1_intervals)
    p2_intervals = merge_intervals(p2_intervals)

    x1 = slice_concat_audio(x, SR, p1_intervals)
    x2 = slice_concat_audio(x, SR, p2_intervals)

    t1 = " ".join(p1_texts).strip()
    t2 = " ".join(p2_texts).strip()

    def conf_from_logps(logps: List[float], fallback: float) -> float:
        if not logps:
            return fallback
        avg = float(np.mean(logps))
        return float(math.exp(min(0.0, avg))) if math.isfinite(avg) else fallback

    conf_global = float(tr["asr_conf"])
    conf1 = conf_from_logps(p1_logps, conf_global)
    conf2 = conf_from_logps(p2_logps, conf_global)

    m1 = phase_metrics(
        x_phase=x1,
        transcript=t1,
        asr_conf=conf1,
        vad_aggr=vad_aggr,
        filler_extra=filler_extra,
        pause_threshold_s=pause_threshold_s,
        audio_weight=audio_weight,
        fluency_weight=fluency_weight,
        grammar_tool=grammar_tool,
        med_allowlist=med_allowlist,
        grammar_max_chars=grammar_max_chars,
        total_language_weight=total_language_weight,
        total_fluency_weight=total_fluency_weight,
        total_clarity_weight=total_clarity_weight,
    )
    m2 = phase_metrics(
        x_phase=x2,
        transcript=t2,
        asr_conf=conf2,
        vad_aggr=vad_aggr,
        filler_extra=filler_extra,
        pause_threshold_s=pause_threshold_s,
        audio_weight=audio_weight,
        fluency_weight=fluency_weight,
        grammar_tool=grammar_tool,
        med_allowlist=med_allowlist,
        grammar_max_chars=grammar_max_chars,
        total_language_weight=total_language_weight,
        total_fluency_weight=total_fluency_weight,
        total_clarity_weight=total_clarity_weight,
    )

    overall = phase1_weight * float(m1.get("overall_score", 0.0)) + phase2_weight * float(m2.get("overall_score", 0.0))

    return {
        "schema_version": schema_version,
        "filename": filename,
        "audio_end_s": audio_end_s,
        "boundaries": {
            "anamnesis_end_s": float(min(anamnesis_end_s, audio_end_s)),
            "presentation_start_s": presentation_start,
            "feedback_start_s": feedback_start,
        },
        "weights": {
            "phase1_weight": phase1_weight,
            "phase2_weight": phase2_weight,
            "audio_weight": audio_weight,
            "fluency_weight": fluency_weight,
            "total_language_weight": total_language_weight,
            "total_fluency_weight": total_fluency_weight,
            "total_clarity_weight": total_clarity_weight,
        },
        "assistant_phase1": {**m1, "intervals": p1_intervals[:2000], "transcript": t1},
        "assistant_phase2": {
            **m2,
            "intervals": p2_intervals[:2000],
            "transcript": t2,
            "excluded_short_questions_count": int(p2_excluded_short_q_count),
            "excluded_short_questions_s": float(p2_excluded_short_q_s),
        },
        "assistant_overall_score": float(overall),
        "asr_language": tr.get("asr_language"),
        "asr_language_prob": tr.get("asr_language_prob"),
        "avg_logprob_mean": tr.get("avg_logprob_mean"),
        "transcript_full": tr.get("transcript"),
    }


# -------------------------- CLI --------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identifier", required=True)
    ap.add_argument("--out_dir", default="public")
    ap.add_argument("--model", default="base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute_type", default="int8")
    ap.add_argument("--language", default="de")
    ap.add_argument("--filler_extra", default="")
    ap.add_argument("--vad_aggr", type=int, default=3)

    ap.add_argument("--max_new_files", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--download_workers", type=int, default=6)

    ap.add_argument("--clip_seconds", type=int, default=4200)
    ap.add_argument("--anamnesis_seconds", type=int, default=1200)
    ap.add_argument("--pause_threshold_s", type=float, default=1.0)

    ap.add_argument("--audio_weight", type=float, default=0.45)
    ap.add_argument("--fluency_weight", type=float, default=0.55)

    ap.add_argument("--phase1_weight", type=float, default=0.5)
    ap.add_argument("--phase2_weight", type=float, default=0.5)

    ap.add_argument("--total_language_weight", type=float, default=0.50)
    ap.add_argument("--total_fluency_weight", type=float, default=0.30)
    ap.add_argument("--total_clarity_weight", type=float, default=0.20)

    ap.add_argument("--schema_version", type=int, default=6)

    ap.add_argument("--presentation_window_s", type=float, default=90.0)
    ap.add_argument("--presentation_threshold", type=float, default=0.06)

    ap.add_argument("--feedback_lookback_s", type=float, default=600.0)
    ap.add_argument("--feedback_window_s", type=float, default=60.0)
    ap.add_argument("--feedback_threshold", type=float, default=0.10)
    ap.add_argument("--feedback_min_tail_ratio", type=float, default=0.75)

    ap.add_argument("--p2_question_max_s", type=float, default=3.0)
    ap.add_argument("--p2_min_keep_s", type=float, default=1.2)

    ap.add_argument("--enable_grammar", type=int, default=1)
    ap.add_argument("--grammar_max_chars", type=int, default=12000)
    ap.add_argument("--med_allowlist_path", default="data/medical_allowlist.txt")

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "scores.full.json"
    scores_path = out_dir / "scores.json"
    metrics_path = out_dir / "metrics.csv"

    full: Dict[str, Dict[str, Any]] = {}
    if full_path.exists():
        full = json.loads(full_path.read_text(encoding="utf-8") or "{}")

    med_allowlist = load_allowlist(args.med_allowlist_path)
    grammar_tool = init_language_tool(enabled=bool(int(args.enable_grammar)))

    files = fetch_ia_files(args.identifier)
    candidates = pick_audio_candidates(files)

    def needs_rescore(fn: str) -> bool:
        item = full.get(fn)
        if not isinstance(item, dict):
            return True
        return int(item.get("schema_version", -1)) != int(args.schema_version)

    scoring_candidates = [f for f in candidates if needs_rescore(f)]
    batch_now = scoring_candidates[: max(0, int(args.max_new_files))]

    print(
        f"IA candidates={len(candidates)} scoring_candidates={len(scoring_candidates)} scoring_now={len(batch_now)} schema={args.schema_version}",
        flush=True,
    )

    cache_root = Path(".cache/whisper").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, download_root=str(cache_root))

    language = args.language.strip() or None
    filler_extra = args.filler_extra.strip() or None

    if batch_now:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            for i in range(0, len(batch_now), int(args.batch_size)):
                batch = batch_now[i : i + int(args.batch_size)]
                local_map: Dict[str, Path] = {}

                with ThreadPoolExecutor(max_workers=max(1, int(args.download_workers))) as ex:
                    futs = {}
                    for filename in batch:
                        local = tmpdir / Path(filename).name
                        local_map[filename] = local
                        futs[ex.submit(download_ia_file, args.identifier, filename, local)] = filename
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", leave=False):
                        fut.result()

                for filename in tqdm(batch, desc="Scoring (language-focused)", leave=False):
                    local = local_map[filename]
                    work_audio = local

                    if int(args.clip_seconds) > 0:
                        clip_wav = tmpdir / f"{local.stem}.clip.wav"
                        ffmpeg_make_clip_wav(local, clip_wav, int(args.clip_seconds))
                        work_audio = clip_wav

                    row = analyze_assistant_dynamic(
                        wav_path=work_audio,
                        model=model,
                        filename=filename,
                        language=language,
                        filler_extra=filler_extra,
                        vad_aggr=int(args.vad_aggr),
                        anamnesis_end_s=float(args.anamnesis_seconds),
                        pause_threshold_s=float(args.pause_threshold_s),
                        audio_weight=float(args.audio_weight),
                        fluency_weight=float(args.fluency_weight),
                        phase1_weight=float(args.phase1_weight),
                        phase2_weight=float(args.phase2_weight),
                        schema_version=int(args.schema_version),
                        pres_win_s=float(args.presentation_window_s),
                        pres_threshold=float(args.presentation_threshold),
                        fb_lookback_s=float(args.feedback_lookback_s),
                        fb_win_s=float(args.feedback_window_s),
                        fb_threshold=float(args.feedback_threshold),
                        fb_min_tail_ratio=float(args.feedback_min_tail_ratio),
                        p2_question_max_s=float(args.p2_question_max_s),
                        p2_min_keep_s=float(args.p2_min_keep_s),
                        grammar_tool=grammar_tool,
                        med_allowlist=med_allowlist,
                        grammar_max_chars=int(args.grammar_max_chars),
                        total_language_weight=float(args.total_language_weight),
                        total_fluency_weight=float(args.total_fluency_weight),
                        total_clarity_weight=float(args.total_clarity_weight),
                    )
                    full[filename] = row

                full_path.write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")

    scores = {k: round(float(v.get("assistant_overall_score", 0.0)), 1) for k, v in full.items()}
    scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    if full:
        rows = []
        for item in full.values():
            p1 = item.get("assistant_phase1", {}) or {}
            p2 = item.get("assistant_phase2", {}) or {}
            b = item.get("boundaries", {}) or {}
            rows.append(
                {
                    "filename": item.get("filename"),
                    "assistant_overall_score": item.get("assistant_overall_score"),
                    "anamnesis_end_s": b.get("anamnesis_end_s"),
                    "presentation_start_s": b.get("presentation_start_s"),
                    "feedback_start_s": b.get("feedback_start_s"),
                    "p1_overall": p1.get("overall_score"),
                    "p1_duration_s": p1.get("duration_s"),
                    "p1_grammar_per_100w": p1.get("grammar_errors_per_100w"),
                    "p1_language_quality": p1.get("language_quality_score"),
                    "p1_clarity": p1.get("clarity_score"),
                    "p1_fluency": p1.get("fluency_score"),
                    "p1_filler_per_100w": p1.get("filler_per_100w"),
                    "p1_long_pauses_per_min": p1.get("long_pauses_per_min"),
                    "p2_overall": p2.get("overall_score"),
                    "p2_duration_s": p2.get("duration_s"),
                    "p2_grammar_per_100w": p2.get("grammar_errors_per_100w"),
                    "p2_language_quality": p2.get("language_quality_score"),
                    "p2_clarity": p2.get("clarity_score"),
                    "p2_fluency": p2.get("fluency_score"),
                    "p2_filler_per_100w": p2.get("filler_per_100w"),
                    "p2_long_pauses_per_min": p2.get("long_pauses_per_min"),
                    "p2_excl_short_q_count": p2.get("excluded_short_questions_count"),
                    "p2_excl_short_q_s": p2.get("excluded_short_questions_s"),
                    "schema_version": item.get("schema_version"),
                }
            )
        df = pd.DataFrame(rows).sort_values("assistant_overall_score", ascending=False, kind="mergesort")
        df.to_csv(metrics_path, index=False, encoding="utf-8")
    else:
        metrics_path.write_text("", encoding="utf-8")

    try:
        if grammar_tool is not None and hasattr(grammar_tool, "close"):
            grammar_tool.close()
    except Exception:
        pass

    print(f"Wrote: {scores_path} | {full_path} | {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
