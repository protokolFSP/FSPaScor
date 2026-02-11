# file: tools/update_scores_from_ia.py
"""
Incremental IA audio scoring (German) with:
- max files per run (avoid GH Actions time limits)
- optional clipping (score only first N seconds)
- batching + parallel download

Outputs (in --out_dir):
- scores.json         filename -> overall_score (0..100)
- scores.full.json    filename -> full metrics (state for incremental runs)
- metrics.csv         tabular view

Usage:
python tools/update_scores_from_ia.py --identifier <IA_ID> --out_dir public --language de
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
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import requests
import webrtcvad
from faster_whisper import WhisperModel
from tqdm import tqdm

EPS = 1e-12


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
        "16000",
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
        "16000",
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


def vad_flags(pcm16: bytes, sample_rate: int = 16000, aggressiveness: int = 3, frame_ms: int = 30) -> List[bool]:
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
        r"mmm+",
        r"also",
        r"halt",
        r"eben",
        r"sozusagen",
        r"genau",
        r"okay",
        r"ok",
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
    overall = 0.60 * audio_quality + 0.40 * fluency
    return Scores(audio_quality=audio_quality, fluency=fluency, overall=overall)


def transcribe(model: WhisperModel, path: Path, language: Optional[str]) -> Dict[str, object]:
    segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    texts: List[str] = []
    avg_logprobs: List[float] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            texts.append(t)
        if getattr(seg, "avg_logprob", None) is not None:
            avg_logprobs.append(float(seg.avg_logprob))

    transcript = " ".join(texts).strip()
    avg_logprob_mean = float(np.mean(avg_logprobs)) if avg_logprobs else float("nan")
    asr_conf = float(math.exp(min(0.0, avg_logprob_mean))) if math.isfinite(avg_logprob_mean) else 0.0

    return {
        "transcript": transcript,
        "asr_conf": asr_conf,
        "asr_language": getattr(info, "language", None),
        "asr_language_prob": getattr(info, "language_probability", None),
        "avg_logprob_mean": avg_logprob_mean,
    }


def download_ia_file(identifier: str, filename: str, out_path: Path) -> None:
    base = f"https://archive.org/download/{identifier}/"
    url = base + quote(filename, safe="/()[],'&+;=:@$-_.!~*")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def fetch_ia_files(identifier: str) -> List[Dict[str, object]]:
    url = f"https://archive.org/metadata/{identifier}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", []) or []
    return [f for f in files if isinstance(f, dict) and f.get("name")]


def pick_audio_candidates(files: List[Dict[str, object]]) -> List[str]:
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


def analyze_one(
    wav_or_audio_path: Path,
    model: WhisperModel,
    filename: str,
    language: Optional[str],
    filler_extra: Optional[str],
    vad_aggr: int = 3,
) -> Dict[str, object]:
    pcm16 = ffmpeg_decode_pcm16_mono_16k(wav_or_audio_path)
    x = pcm16_to_float32(pcm16)
    sr = 16000

    clipping_ratio = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0
    lufs = calc_lufs(x, sr)

    flags = vad_flags(pcm16, sample_rate=sr, aggressiveness=vad_aggr, frame_ms=30)
    frame_s = 0.03
    _, silence_segs = rle_segments(flags, frame_s=frame_s)
    speech_ratio = float(np.mean(flags)) if flags else 0.0
    pause_ratio = 1.0 - speech_ratio

    frame_len = int(sr * 30 / 1000)
    snr_db = float("nan")
    if len(flags) > 0 and x.size >= frame_len:
        nframes = min(len(flags), x.size // frame_len)
        xr = x[: nframes * frame_len].reshape(nframes, frame_len)
        frame_rms = np.sqrt(np.mean(xr * xr, axis=1) + EPS)
        f = np.array(flags[:nframes], dtype=bool)
        if f.any() and (~f).any():
            speech_rms = float(np.mean(frame_rms[f]))
            noise_rms = float(np.mean(frame_rms[~f]))
            snr_db = 20.0 * math.log10(speech_rms / (noise_rms + EPS))

    pauses = [end - start for (start, end) in silence_segs]
    long_pauses_05s = sum(1 for p in pauses if p >= 0.5)
    duration_s = float(len(x) / sr) if x.size else 0.0
    long_pauses_per_min = float(long_pauses_05s / max(1e-6, duration_s / 60.0))

    tr = transcribe(model, wav_or_audio_path, language=language)
    transcript = str(tr["transcript"])
    asr_conf = float(tr["asr_conf"])

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
    )

    return {
        "filename": filename,
        "duration_s": duration_s,
        "clipping_ratio": clipping_ratio,
        "lufs": lufs,
        "snr_db_est": None if not math.isfinite(snr_db) else float(snr_db),
        "speech_ratio": speech_ratio,
        "pause_ratio": pause_ratio,
        "long_pauses_0p5s": int(long_pauses_05s),
        "long_pauses_per_min": long_pauses_per_min,
        "asr_conf": asr_conf,
        "asr_language": tr.get("asr_language"),
        "asr_language_prob": tr.get("asr_language_prob"),
        "avg_logprob_mean": tr.get("avg_logprob_mean"),
        "word_count": word_count,
        "filler_count": filler_count,
        "filler_per_100w": filler_per_100w,
        "repetition_rate": rep,
        "weird_token_rate": weird,
        "audio_quality_score": scores.audio_quality,
        "fluency_score": scores.fluency,
        "overall_score": scores.overall,
        "transcript": transcript,
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

    ap.add_argument("--max_new_files", type=int, default=8)
    ap.add_argument("--clip_seconds", type=int, default=300)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--download_workers", type=int, default=6)
    ap.add_argument("--score_workers", type=int, default=1)  # kept for CLI compatibility (scoring is sequential here)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "scores.full.json"
    scores_path = out_dir / "scores.json"
    metrics_path = out_dir / "metrics.csv"

    full: Dict[str, Dict[str, object]] = {}
    if full_path.exists():
        full = json.loads(full_path.read_text(encoding="utf-8") or "{}")

    files = fetch_ia_files(args.identifier)
    candidates = pick_audio_candidates(files)
    processed = set(full.keys())

    new_files_all = [f for f in candidates if f not in processed]
    new_files = new_files_all[: max(0, int(args.max_new_files))]

    print(f"IA candidates: {len(candidates)} | processed: {len(processed)} | new: {len(new_files_all)} | scoring_now: {len(new_files)}", flush=True)

    cache_root = Path(".cache/whisper").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(cache_root),
    )

    language = args.language.strip() or None
    filler_extra = args.filler_extra.strip() or None
    clip_seconds = int(args.clip_seconds)

    if new_files:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            for i in range(0, len(new_files), int(args.batch_size)):
                batch = new_files[i : i + int(args.batch_size)]
                local_map: Dict[str, Path] = {}

                # download parallel
                with ThreadPoolExecutor(max_workers=max(1, int(args.download_workers))) as ex:
                    futs = {}
                    for filename in batch:
                        local = tmpdir / Path(filename).name
                        local_map[filename] = local
                        futs[ex.submit(download_ia_file, args.identifier, filename, local)] = filename

                    for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", leave=False):
                        fut.result()

                # score sequential (stable on GH)
                for filename in tqdm(batch, desc="Scoring", leave=False):
                    local = local_map[filename]
                    if clip_seconds > 0:
                        clip_wav = tmpdir / f"{local.stem}.clip.wav"
                        ffmpeg_make_clip_wav(local, clip_wav, clip_seconds)
                        row = analyze_one(clip_wav, model, filename, language, filler_extra, vad_aggr=int(args.vad_aggr))
                    else:
                        row = analyze_one(local, model, filename, language, filler_extra, vad_aggr=int(args.vad_aggr))

                    full[filename] = row

                # persist state after each batch (so next run continues)
                full_path.write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")

    # write outputs every run (even if no new files)
    scores = {k: round(float(v["overall_score"]), 1) for k, v in full.items()}
    scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    if full:
        df = pd.DataFrame(list(full.values())).sort_values("overall_score", ascending=False, kind="mergesort")
        df.to_csv(metrics_path, index=False, encoding="utf-8")
    else:
        metrics_path.write_text("", encoding="utf-8")

    print(f"Wrote: {scores_path} | {full_path} | {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
