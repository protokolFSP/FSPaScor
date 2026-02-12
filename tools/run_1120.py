"""
Run scoring pipeline.

Env:
- TRANSCRIPTS_DIR (default: _deps/FSPtranskript/transcripts)
- PUBLIC_DIR (default: public)
- MAX_SECONDS (default: 1120)
- MAX_NEW_FILES (default: 10)
- ARCHIVE_ITEM_IDENTIFIER (required for deterministic audio match)
- REQUIRE_AUDIO (default: 1)
- AUDIO_DEBUG (default: 0)
- WHISPER_MODEL (default: medium)
- WHISPER_COMPUTE_TYPE (default: int8)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from fspascore.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    transcripts_dir = Path(os.getenv("TRANSCRIPTS_DIR", "_deps/FSPtranskript/transcripts")).resolve()
    public_dir = Path(os.getenv("PUBLIC_DIR", "public")).resolve()

    max_seconds = float(os.getenv("MAX_SECONDS", "1120"))
    max_new_files = int(os.getenv("MAX_NEW_FILES", "10"))

    run_pipeline(
        transcripts_dir=transcripts_dir,
        public_dir=public_dir,
        max_seconds=max_seconds,
        max_new_files=max_new_files,
    )


if __name__ == "__main__":
    main()
