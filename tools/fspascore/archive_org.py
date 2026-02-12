from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests


_ARCHIVE_META = "https://archive.org/metadata/{identifier}"
_ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&rows={rows}&page=1&output=json"
_ARCHIVE_DL = "https://archive.org/download/{identifier}/{filename}"

_ALLOWED_AUDIO_EXT = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus")


@dataclass(frozen=True)
class ArchiveAudioRef:
    identifier: str
    filename: str
    download_url: str


def _de_umlaut(s: str) -> str:
    rep = (
        ("ä", "ae"),
        ("ö", "oe"),
        ("ü", "ue"),
        ("ß", "ss"),
        ("Ä", "Ae"),
        ("Ö", "Oe"),
        ("Ü", "Ue"),
    )
    for a, b in rep:
        s = s.replace(a, b)
    return s


def slugify_archive_identifier(title: str) -> str:
    """
    Best-effort identifier guess from transcript stem.
    Archive identifiers commonly look like:
      vorhofflimmern-bei-bekannter-khk-dr-oemer-dr-remzi-09.05.25
    """
    t = title.strip()
    t = _de_umlaut(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    # Keep digits, letters, dots (dates), spaces -> hyphen
    t = re.sub(r"[^\w\s\.-]+", " ", t)
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t


def _get_json(url: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_metadata(identifier: str) -> Optional[Dict[str, Any]]:
    return _get_json(_ARCHIVE_META.format(identifier=identifier))


def _pick_audio_file(files: List[Dict[str, Any]]) -> Optional[str]:
    """
    Choose an audio file preferring:
    - source=original
    - common audio formats
    - largest size if available
    """
    candidates: List[Tuple[int, int, str]] = []
    for f in files:
        name = (f.get("name") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if not lower.endswith(_ALLOWED_AUDIO_EXT):
            continue
        if lower.endswith((".torrent", ".xml", ".json")):
            continue

        source = (f.get("source") or "").lower()
        is_original = 1 if source == "original" else 0

        size = 0
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0

        ext_rank = 0
        for i, ext in enumerate((".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")):
            if lower.endswith(ext):
                ext_rank = (len(_ALLOWED_AUDIO_EXT) - i)
                break

        score = is_original * 1000 + ext_rank * 10
        candidates.append((score, size, name))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _search_identifiers(query: str, rows: int = 8) -> List[str]:
    url = _ARCHIVE_SEARCH.format(q=quote_plus(query), rows=rows)
    data = _get_json(url)
    if not data:
        return []
    docs = (((data.get("response") or {}).get("docs")) or [])
    out: List[str] = []
    for d in docs:
        ident = d.get("identifier")
        if isinstance(ident, str) and ident:
            out.append(ident)
    return out


def resolve_audio(title: str) -> Optional[ArchiveAudioRef]:
    """
    Resolve archive.org audio for a given transcript stem.
    Strategy:
    1) guess identifier via slugify + metadata check
    2) fallback to advancedsearch (mediatype:audio) and try top identifiers
    """
    guess = slugify_archive_identifier(title)
    meta = fetch_metadata(guess)
    if meta and isinstance(meta.get("files"), list):
        filename = _pick_audio_file(meta["files"])
        if filename:
            return ArchiveAudioRef(
                identifier=guess,
                filename=filename,
                download_url=_ARCHIVE_DL.format(identifier=guess, filename=filename),
            )

    q = f'mediatype:audio AND (title:"{title}" OR identifier:"{guess}" OR "{guess}")'
    for ident in _search_identifiers(q):
        meta2 = fetch_metadata(ident)
        if not meta2 or not isinstance(meta2.get("files"), list):
            continue
        filename = _pick_audio_file(meta2["files"])
        if not filename:
            continue
        return ArchiveAudioRef(
            identifier=ident,
            filename=filename,
            download_url=_ARCHIVE_DL.format(identifier=ident, filename=filename),
        )

    return None
