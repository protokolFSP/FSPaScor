"""
Speaker diarization + assistant speaker detection with WhisperX (German FSP-style dialogs).

Requirements:
  pip install -U whisperx torch torchaudio
  (Plus FFmpeg installed)

Auth:
  export HF_TOKEN="hf_..."

Usage:
  python diarize_whisperx.py --audio "file.m4a" --language de --device cpu --model small
  python diarize_whisperx.py --audio "file.m4a" --device cuda --compute_type float16

Outputs:
  - <out_dir>/segments_speaker.json  (all segments with speaker)
  - <out_dir>/assistant_segments.json (only assistant speaker segments)
  - <out_dir>/assistant_summary.json (assistant speaker id + stats)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# WhisperX API
import whisperx  # type: ignore


ASSISTANT_HINTS = [
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
    r"\bnehmen sie\b",
    r"\bskala von\b",
    r"\bist das in ordnung\b",
]

PATIENT_HINTS = [
    r"\bich habe\b",
    r"\bmir ist\b",
    r"\bes tut\b",
    r"\bschmerzen\b",
    r"\bübel\b",
    r"\berbrochen\b",
    r"\bich bin\b.*\bgekommen\b",
]


def _count(text: str, patterns: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for p in patterns if re.search(p, t))


def _is_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return "?" in t or _count(t, [r"\bkönnten\b", r"\bkönnen\b", r"\bdarf\b", r"\bhaben sie\b"]) > 0


def pick_assistant_speaker(
    segments: List[Dict[str, Any]],
    anamnesis_end_s: float = 1200.0,
) -> Optional[str]:
    """
    Pick assistant speaker by scoring speakers in first N seconds:
    - + assistant lexical hints
    - + question-ness
    - - patient lexical hints
    """
    speaker_scores: Dict[str, float] = {}
    speaker_dur: Dict[str, float] = {}

    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if s > anamnesis_end_s:
            continue
        txt = str(seg.get("text") or "")
        dur = max(0.0, e - s)

        a = _count(txt, ASSISTANT_HINTS)
        p = _count(txt, PATIENT_HINTS)
        q = 2 if _is_question(txt) else 0

        score = (a * 2.0) + q - (p * 1.5)
        speaker_scores[spk] = speaker_scores.get(spk, 0.0) + score
        speaker_dur[spk] = speaker_dur.get(spk, 0.0) + dur

    if not speaker_scores:
        return None

    # Prefer high score; tie-breaker by total duration in anamnesis window
    best = sorted(
        speaker_scores.keys(),
        key=lambda k: (speaker_scores[k], speaker_dur.get(k, 0.0)),
        reverse=True,
    )[0]
    return best


def run_whisperx(
    audio_path: Path,
    out_dir: Path,
    language: Optional[str],
    device: str,
    model_name: str,
    compute_type: str,
    batch_size: int,
    do_align: bool,
    diarize: bool,
    anamnesis_end_s: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(model_name, device=device, compute_type=compute_type, language=language)
    result: Dict[str, Any] = model.transcribe(audio, batch_size=batch_size)

    if do_align:
        lang = result.get("language") or language
        if not lang:
            lang = "de"
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

    if diarize:
        if not hf_token:
            raise RuntimeError("HF_TOKEN missing. Set env HF_TOKEN=hf_... for pyannote diarization.")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    segments: List[Dict[str, Any]] = result.get("segments", [])
    (out_dir / "segments_speaker.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    assistant = pick_assistant_speaker(segments, anamnesis_end_s=anamnesis_end_s)

    if assistant is None:
        summary = {
            "assistant_speaker": None,
            "note": "Could not infer assistant speaker. Check segments_speaker.json",
        }
        (out_dir / "assistant_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    assistant_segments = [s for s in segments if s.get("speaker") == assistant]
    (out_dir / "assistant_segments.json").write_text(
        json.dumps(assistant_segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_dur = sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in assistant_segments)
    summary = {
        "assistant_speaker": assistant,
        "assistant_segments": len(assistant_segments),
        "assistant_total_speech_s": round(total_dur, 2),
        "language": result.get("language"),
        "model": model_name,
        "device": device,
        "aligned": bool(do_align),
        "diarized": bool(diarize),
        "anamnesis_end_s": anamnesis_end_s,
    }
    (out_dir / "assistant_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_dir", default="out_diarize")
    ap.add_argument("--language", default="de")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model", default="small")  # small/base/medium/large-v2...
    ap.add_argument("--compute_type", default="int8")  # cpu: int8/float32, cuda: float16/int8_float16
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--no_diarize", action="store_true")
    ap.add_argument("--anamnesis_end_s", type=float, default=1200.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_whisperx(
        audio_path=Path(args.audio),
        out_dir=Path(args.out_dir),
        language=(args.language or None),
        device=args.device,
        model_name=args.model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        do_align=bool(args.align),
        diarize=not bool(args.no_diarize),
        anamnesis_end_s=float(args.anamnesis_end_s),
    )


if __name__ == "__main__":
    main()
