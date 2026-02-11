"""
Dynamic phase-aware assistant-only scoring for FSP dialog recordings (German).

Assumptions:
- Phase 1 (anamnesis/interview): first N seconds (default 1200s fixed).
- Phase 2 (presentation): starts after phase 1, varies; auto-detected via lexical cues.
- Phase 3 (feedback/rückmeldung): usually near end; auto-detected; excluded.

Outputs (in --out_dir):
- scores.json: filename -> assistant_overall_score
- scores.full.json: detailed metrics + boundaries + transcripts
- metrics.csv: sortable summary

Incremental:
- Rescore when schema_version changes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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


@dataclass(frozen=True)
class Scores:
    audio_quality: float
    fluency: float
    overall: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def ffmpeg_make_clip_wav(in_path: Path, out_wav: Path, clip_seconds: int) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-i", str(in_path),
        "-t", str(int(clip_seconds)),
        "-ac", "1", "-ar", str(SR),
        str(out_wav),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore")[:800])


def ffmpeg_decode_pcm16_mono_16k(path: Path) -> bytes:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(SR),
        "-f", "s16le", "pipe:1",
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
        yield pcm16[i: i + nbytes]


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
    return re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿÄÖÜäöüß]+", text.lower())


def repetition_rate(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    rep = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return rep / (len(tokens) - 1)


def weird_token_rate(text: str) -> float:
    raw = re.findall(r"\S+", text.strip())
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


def compute_scores(
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
    overall = audio_weight * audio_quality + fluency_weight * fluency
    return Scores(audio_quality=audio_quality, fluency=fluency, overall=overall)


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

PHASE2_REPORT_PATTERNS = [
    r"\bich stelle (ihnen|euch) (kurz )?vor\b",
    r"\bich berichte\b",
    r"\bzusammenfassung\b",
    r"\banamnese\b",
    r"\bvorerkrankungen\b",
    r"\bmedikation\b",
    r"\ballergien\b",
    r"\bbefund\b",
    r"\bdiagnose\b",
    r"\bplan\b",
    r"\bvorstellung\b",
    r"\bder patient\b",
    r"\bdie patientin\b",
    r"\bwir haben\b",
]

FEEDBACK_PATTERNS = [
    r"\brückmeldung\b",
    r"\bfeedback\b",
    r"\bgut gemacht\b",
    r"\bsehr gut\b",
    r"\bprima\b",
    r"\btop\b",
    r"\bverbessern\b",
    r"\bverbesserung\b",
    r"\bachten sie\b",
    r"\bsie sollten\b",
    r"\bsie könnten\b",
    r"\btipp\b",
    r"\bfazit\b",
    r"\bzusammengefasst\b",
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


def _count_matches(text: str, patterns: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in patterns if re.search(p, t))


def is_assistant_phase1(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    a = _count_matches(t, PHASE1_ASSISTANT_PATTERNS)
    p = _count_matches(t, PATIENT_LIKE_PATTERNS)
    if "?" in t:
        a += 2
    if a == 0 and p == 0 and len(t) <= 10:
        a = 1
    return a >= max(1, p)


def is_assistant_phase2(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    a = _count_matches(t, PHASE2_REPORT_PATTERNS) * 2 + _count_matches(t, PHASE1_ASSISTANT_PATTERNS)
    p = _count_matches(t, PATIENT_LIKE_PATTERNS)
    if "?" in t:
        a += 1
    if a == 0 and p == 0 and len(t) <= 10:
        a = 1
    return a >= max(1, p + 1)


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


def detect_presentation_start(
    segs: List[Dict[str, Any]],
    after_s: float,
    window_s: float,
    threshold: float,
) -> Optional[float]:
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
    lookback_s: float,
    window_s: float,
    threshold: float,
) -> Optional[float]:
    region_start = max(0.0, audio_end_s - lookback_s)
    pts: List[Tuple[float, float]] = []
    for seg in segs:
        s = float(seg["start"])
        if s < region_start:
            continue
        t = str(seg.get("text") or "")
        score = float(_count_matches(t, FEEDBACK_PATTERNS) * 2)
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


def download_ia_file(identifier: str, filename: str, out_path: Path) -> None:
    base = f"https://archive.org/download/{identifier}/"
    url = base + quote(filename, safe="/()[],'&+;=:@$-_.!~*")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
