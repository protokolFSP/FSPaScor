from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from faster_whisper import WhisperModel

from .archive_org import ArchiveAudioRef, resolve_audio
from .srt_io import Segment

_UA = {"User-Agent": "fspascore/1.0 (github-actions)"}


@dataclass(frozen=True)
class AudioTranscript:
    assistant_segments: List[Segment]
    patient_segments: List[Segment]
    assistant_text: str
    audio_ref: ArchiveAudioRef
    whisper_model: str
    whisper_compute_type: str


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def _cache_key(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _ffmpeg_url_to_wav(url: str, out_wav: Path, max_seconds: float) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    headers = "User-Agent: fspascore/1.0 (github-actions)\r\n"
    cmd = [
        "ffmpeg",
        "-y",
        "-headers",
        headers,
        "-i",
        url,
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_wav),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg(url->wav) failed: {p.stderr[-2000:]}")


def _download(url: str, dest: Path, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout, headers=_UA) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ffmpeg_file_to_wav(src: Path, out_wav: Path, max_seconds: float) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_wav),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg(file->wav) failed: {p.stderr[-2000:]}")


class _WhisperSingleton:
    model: Optional[WhisperModel] = None
    model_name: Optional[str] = None
    compute_type: Optional[str] = None


def _get_whisper(model_name: str, compute_type: str) -> WhisperModel:
    if (
        _WhisperSingleton.model is None
        or _WhisperSingleton.model_name != model_name
        or _WhisperSingleton.compute_type != compute_type
    ):
        _WhisperSingleton.model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        _WhisperSingleton.model_name = model_name
        _WhisperSingleton.compute_type = compute_type
    return _WhisperSingleton.model


def _transcribe_whisper(wav_path: Path, model_name: str, compute_type: str) -> List[Segment]:
    model = _get_whisper(model_name, compute_type)
    segments, _info = model.transcribe(
        str(wav_path),
        language="de",
        vad_filter=True,
        beam_size=5,
    )
    out: List[Segment] = []
    for s in segments:
        text = " ".join((s.text or "").split()).strip()
        if not text:
            continue
        out.append(Segment(start=float(s.start), end=float(s.end), text=text))
    return out


def _split_by_srt_guide(
    whisper_segments: List[Segment],
    assistant_guide: List[Segment],
    guide_pad_sec: float = 0.35,
) -> Tuple[List[Segment], List[Segment]]:
    intervals = [(max(0.0, s.start - guide_pad_sec), s.end + guide_pad_sec) for s in assistant_guide]
    guide = _merge_intervals(intervals)

    assistant: List[Segment] = []
    patient: List[Segment] = []

    for seg in whisper_segments:
        seg_iv = (seg.start, seg.end)
        seg_dur = max(1e-6, seg.end - seg.start)

        ov = 0.0
        for g in guide:
            if g[0] > seg.end:
                break
            if g[1] < seg.start:
                continue
            ov += _overlap(seg_iv, g)

        if ov >= 0.40 or (ov / seg_dur) >= 0.30:
            assistant.append(seg)
        else:
            patient.append(seg)

    return assistant, patient


def transcribe_audio_for_title(
    title: str,
    assistant_guide_segments: List[Segment],
    max_seconds: float,
    cache_dir: Path,
    whisper_model: str,
    whisper_compute_type: str,
) -> Optional[AudioTranscript]:
    debug = os.getenv("AUDIO_DEBUG", "0") == "1"
    require_audio = os.getenv("REQUIRE_AUDIO", "1") == "1"

    audio_ref = resolve_audio(title)
    if not audio_ref:
        if require_audio:
            raise RuntimeError(
                f"Audio resolve failed for title='{title}'. "
                f"Set ARCHIVE_ITEM_IDENTIFIER (repo variable) to your IA item identifier."
            )
        if debug:
            print(f"[audio] resolve_audio=None title={title} (fallback allowed)")
        return None

    key = _cache_key(title)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"{key}_{int(max_seconds)}s.wav"

    if debug:
        print(f"[audio] resolved title={title} item={audio_ref.identifier} file={audio_ref.filename}")

    if not wav_path.exists():
        try:
            _ffmpeg_url_to_wav(audio_ref.download_url, wav_path, max_seconds=max_seconds)
        except Exception as e:
            if debug:
                print(f"[audio] url->wav failed, fallback to download: {e}")

            raw_ext = Path(audio_ref.filename).suffix.lower() or ".bin"
            raw_path = cache_dir / f"{key}{raw_ext}"
            if not raw_path.exists():
                _download(audio_ref.download_url, raw_path)
            _ffmpeg_file_to_wav(raw_path, wav_path, max_seconds=max_seconds)

    whisper_segments = _transcribe_whisper(wav_path, whisper_model, whisper_compute_type)
    assistant, patient = _split_by_srt_guide(whisper_segments, assistant_guide_segments)
    assistant_text = " ".join(s.text for s in assistant).strip()

    return AudioTranscript(
        assistant_segments=assistant,
        patient_segments=patient,
        assistant_text=assistant_text,
        audio_ref=audio_ref,
        whisper_model=whisper_model,
        whisper_compute_type=whisper_compute_type,
    )
