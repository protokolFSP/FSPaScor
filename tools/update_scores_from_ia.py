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


def phase_metrics(
    x_phase: np.ndarray,
    transcript: str,
    asr_conf: float,
    vad_aggr: int,
    filler_extra: Optional[str],
    pause_threshold_s: float,
    audio_weight: float,
    fluency_weight: float,
) -> Dict[str, Any]:
    if x_phase.size < SR * 3:
        return {
            "duration_s": float(x_phase.size / SR) if x_phase.size else 0.0,
            "overall_score": 0.0,
            "audio_quality_score": 0.0,
            "fluency_score": 0.0,
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

    scores = compute_scores(
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

    return {
        "duration_s": duration_s,
        "clipping_ratio": clipping_ratio,
        "lufs": lufs,
        "snr_db_est": None if not math.isfinite(snr_db) else float(snr_db),
        "speech_ratio": speech_ratio,
        "pause_ratio": pause_ratio,
        "long_pauses": int(long_pauses),
        "long_pauses_per_min": long_pauses_per_min,
        "asr_conf": asr_conf,
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_per_100w": filler_per_100w,
        "repetition_rate": rep,
        "weird_token_rate": weird,
        "audio_quality_score": scores.audio_quality,
        "fluency_score": scores.fluency,
        "overall_score": scores.overall,
    }


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
        lookback_s=fb_lookback_s,
        window_s=fb_win_s,
        threshold=fb_threshold,
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

    for seg in segs:
        s = float(seg["start"])
        e = float(seg["end"])
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
            if is_assistant_phase2(txt):
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
        },
        "assistant_phase1": {**m1, "intervals": p1_intervals[:2000], "transcript": t1},
        "assistant_phase2": {**m2, "intervals": p2_intervals[:2000], "transcript": t2},
        "assistant_overall_score": float(overall),
        "asr_language": tr.get("asr_language"),
        "asr_language_prob": tr.get("asr_language_prob"),
        "avg_logprob_mean": tr.get("avg_logprob_mean"),
        "transcript_full": tr.get("transcript"),
    }


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

    ap.add_argument("--schema_version", type=int, default=3)

    ap.add_argument("--presentation_window_s", type=float, default=90.0)
    ap.add_argument("--presentation_threshold", type=float, default=0.06)

    ap.add_argument("--feedback_lookback_s", type=float, default=900.0)
    ap.add_argument("--feedback_window_s", type=float, default=90.0)
    ap.add_argument("--feedback_threshold", type=float, default=0.04)

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
                batch = batch_now[i: i + int(args.batch_size)]
                local_map: Dict[str, Path] = {}

                with ThreadPoolExecutor(max_workers=max(1, int(args.download_workers))) as ex:
                    futs = {}
                    for filename in batch:
                        local = tmpdir / Path(filename).name
                        local_map[filename] = local
                        futs[ex.submit(download_ia_file, args.identifier, filename, local)] = filename
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", leave=False):
                        fut.result()

                for filename in tqdm(batch, desc="Scoring (assistant dynamic)", leave=False):
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
                    "p1_filler_per_100w": p1.get("filler_per_100w"),
                    "p1_long_pauses_per_min": p1.get("long_pauses_per_min"),
                    "p2_overall": p2.get("overall_score"),
                    "p2_duration_s": p2.get("duration_s"),
                    "p2_filler_per_100w": p2.get("filler_per_100w"),
                    "p2_long_pauses_per_min": p2.get("long_pauses_per_min"),
                    "schema_version": item.get("schema_version"),
                }
            )
        df = pd.DataFrame(rows).sort_values("assistant_overall_score", ascending=False, kind="mergesort")
        df.to_csv(metrics_path, index=False, encoding="utf-8")
    else:
        metrics_path.write_text("", encoding="utf-8")

    print(f"Wrote: {scores_path} | {full_path} | {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
