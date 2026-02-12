from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

_ARCHIVE_META = "https://archive.org/metadata/{identifier}"
_ARCHIVE_DL = "https://archive.org/download/{identifier}/{filename}"

_ALLOWED_MEDIA_EXT = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".webm")
_UA = {"User-Agent": "fspascore/1.0 (github-actions)"}


@dataclass(frozen=True)
class ArchiveAudioRef:
    identifier: str
    filename: str
    download_url: str


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split()).strip().casefold()


def _candidate_titles(title: str) -> List[str]:
    t = title.strip()
    out = [t]
    for prefix in ("a ", "A ", "b ", "B "):
        if t.startswith(prefix):
            out.append(t[len(prefix) :].strip())
    seen = set()
    uniq: List[str] = []
    for v in out:
        nv = _norm_name(v)
        if nv and nv not in seen:
            seen.add(nv)
            uniq.append(v)
    return uniq


def _get_json(url: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=timeout, headers=_UA)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_metadata(identifier: str) -> Optional[Dict[str, Any]]:
    return _get_json(_ARCHIVE_META.format(identifier=identifier))


def _download_url(identifier: str, filename: str) -> str:
    return _ARCHIVE_DL.format(identifier=identifier, filename=quote(filename))


def _exact_filename_match(files: List[Dict[str, Any]], title: str) -> Optional[str]:
    title_norms = {_norm_name(t) for t in _candidate_titles(title)}

    prefer = [".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".webm"]
    best = None  # (ext_rank, size, name)

    for f in files:
        name = (f.get("name") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if not lower.endswith(_ALLOWED_MEDIA_EXT):
            continue
        if lower.endswith((".torrent", ".xml", ".json", ".txt", ".srt", ".vtt")):
            continue

        stem = name
        for ext in _ALLOWED_MEDIA_EXT:
            if lower.endswith(ext):
                stem = name[: -len(ext)]
                break

        if _norm_name(stem) not in title_norms:
            continue

        size = 0
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0

        ext_rank = 0
        for i, ext in enumerate(prefer):
            if lower.endswith(ext):
                ext_rank = len(prefer) - i
                break

        cand = (ext_rank, size, name)
        if best is None or cand > best:
            best = cand

    return best[2] if best else None


def resolve_audio(title: str) -> Optional[ArchiveAudioRef]:
    """
    Deterministic mode: ARCHIVE_ITEM_IDENTIFIER must be set for your setup.
    We only look inside that IA item and pick exact same-stem .m4a.
    """
    debug = os.getenv("AUDIO_DEBUG", "0") == "1"
    forced_item = os.getenv("ARCHIVE_ITEM_IDENTIFIER", "").strip()

    def dbg(msg: str) -> None:
        if debug:
            print(msg)

    if not forced_item:
        dbg("[audio] ARCHIVE_ITEM_IDENTIFIER not set")
        return None

    dbg(f"[audio] forced_item={forced_item} title={title}")
    meta = fetch_metadata(forced_item)
    if not meta or not isinstance(meta.get("files"), list):
        dbg("[audio] forced item metadata fetch failed")
        return None

    fn = _exact_filename_match(meta["files"], title)
    if not fn:
        dbg("[audio] no exact same-stem media match inside forced item")
        return None

    dbg(f"[audio] exact match: {fn}")
    return ArchiveAudioRef(forced_item, fn, _download_url(forced_item, fn))
