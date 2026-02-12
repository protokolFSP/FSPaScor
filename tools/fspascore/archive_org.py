from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, quote_plus

import requests


_ARCHIVE_META = "https://archive.org/metadata/{identifier}"
_ARCHIVE_SEARCH = (
    "https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&rows={rows}&page=1&output=json"
)
_ARCHIVE_DL = "https://archive.org/download/{identifier}/{filename}"

_ALLOWED_MEDIA_EXT = (
    ".m4a",
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mkv",
    ".webm",
)


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
    t = title.strip()
    t = _de_umlaut(t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"[^\w\s\.-]+", " ", t)
    t = re.sub(r"[\s_]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t


def _get_json(url: str, timeout: float = 25.0) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_metadata(identifier: str) -> Optional[Dict[str, Any]]:
    return _get_json(_ARCHIVE_META.format(identifier=identifier))


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split()).strip().casefold()
    return s


def _candidate_titles(title: str) -> List[str]:
    t = title.strip()
    variants = [t]
    for prefix in ("a ", "A ", "b ", "B "):
        if t.startswith(prefix):
            variants.append(t[len(prefix) :].strip())

    # de-dup
    seen = set()
    out = []
    for v in variants:
        nv = _norm_name(v)
        if nv and nv not in seen:
            seen.add(nv)
            out.append(v)
    return out


def _exact_filename_match(files: List[Dict[str, Any]], title: str) -> Optional[str]:
    """
    Prefer exact filename match: "<title>.m4a" (or other allowed ext).
    Matching is done via normalized stem equality, because IA filenames can contain unicode/spaces.
    """
    titles = _candidate_titles(title)
    title_norms = {_norm_name(t) for t in titles}

    # 1) exact "<title>.m4a" (by normalized)
    preferred_ext = [".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".mp4", ".mkv", ".webm"]

    best: Optional[Tuple[int, int, str]] = None  # (ext_rank, size, filename)
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
        for i, ext in enumerate(preferred_ext):
            if lower.endswith(ext):
                ext_rank = len(preferred_ext) - i
                break

        cand = (ext_rank, size, name)
        if best is None or cand > best:
            best = cand

    return best[2] if best else None


def _rank_ext(filename: str) -> int:
    lower = filename.lower()
    order = [".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".mp4", ".mkv", ".webm"]
    for i, ext in enumerate(order):
        if lower.endswith(ext):
            return len(order) - i
    return 0


def _pick_media_file(files: List[Dict[str, Any]]) -> Optional[str]:
    candidates: List[Tuple[int, int, str]] = []
    for f in files:
        name = (f.get("name") or "").strip()
        if not name:
            continue
        lower = name.lower()
        if not lower.endswith(_ALLOWED_MEDIA_EXT):
            continue
        if lower.endswith((".torrent", ".xml", ".json", ".txt", ".srt", ".vtt")):
            continue

        source = (f.get("source") or "").lower()
        is_original = 1 if source == "original" else 0

        size = 0
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0

        ext_rank = _rank_ext(name)
        score = is_original * 10_000 + ext_rank * 100
        candidates.append((score, size, name))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _download_url(identifier: str, filename: str) -> str:
    # Quote filename path component (spaces/unicode safe)
    return _ARCHIVE_DL.format(identifier=identifier, filename=quote(filename))


def _search_identifiers(query: str, rows: int = 12) -> List[str]:
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
    Resolve media (audio/video) for transcript title (SRT stem).

    Priority:
    1) If ARCHIVE_ITEM_IDENTIFIER is set => look inside that IA item for exact "<title>.m4a" (or same-stem media).
    2) Else try identifier guesses (slugified) and exact filename inside that item.
    3) Else fallback search by title/identifier and then exact filename, else best media pick.
    """
    forced_item = os.getenv("ARCHIVE_ITEM_IDENTIFIER", "").strip()
    if forced_item:
        meta = fetch_metadata(forced_item)
        if meta and isinstance(meta.get("files"), list):
            fn = _exact_filename_match(meta["files"], title) or _pick_media_file(meta["files"])
            if fn:
                return ArchiveAudioRef(
                    identifier=forced_item,
                    filename=fn,
                    download_url=_download_url(forced_item, fn),
                )

    # Try per-item guess
    guesses: List[str] = []
    for t in _candidate_titles(title):
        guesses.append(slugify_archive_identifier(t))

    # unique
    seen = set()
    uniq = []
    for g in guesses:
        if g and g not in seen:
            seen.add(g)
            uniq.append(g)

    for ident in uniq:
        meta = fetch_metadata(ident)
        if meta and isinstance(meta.get("files"), list):
            fn = _exact_filename_match(meta["files"], title) or _pick_media_file(meta["files"])
            if fn:
                return ArchiveAudioRef(
                    identifier=ident,
                    filename=fn,
                    download_url=_download_url(ident, fn),
                )

    # Search fallback (not restricted to mediatype:audio)
    guess = slugify_archive_identifier(title)
    q = f'(identifier:"{guess}" OR title:"{title}" OR "{guess}")'
    for ident in _search_identifiers(q):
        meta2 = fetch_metadata(ident)
        if not meta2 or not isinstance(meta2.get("files"), list):
            continue
        fn = _exact_filename_match(meta2["files"], title) or _pick_media_file(meta2["files"])
        if not fn:
            continue
        return ArchiveAudioRef(
            identifier=ident,
            filename=fn,
            download_url=_download_url(ident, fn),
        )

    return None
