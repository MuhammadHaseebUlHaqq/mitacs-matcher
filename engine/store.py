"""
Knowledge-base persistence.

bideez keeps knowledge documents in Postgres and the parsed `sourceText` on the
row, so a workspace survives a restart and the profile can be rebuilt without
re-uploading. This is the same idea at local-app scale: one JSON file per
session under `data/kb/`, holding the parsed text and the cached profile.

Deliberately NOT stored: API keys. Those stay in the browser and travel per
request, exactly as Galt RAG does.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from .corpus import Document

STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _path(session_id: str) -> Path:
    """Map a session id to a file, defensively — the id comes from the client
    and must never be able to escape the store directory."""
    safe = _SAFE.sub("", session_id)[:64]
    if not safe:
        safe = "default"
    return STORE_DIR / f"{safe}.json"


def load(session_id: str) -> dict:
    """Return {"docs", "profile", "profile_meta", "extract_cache"}."""
    empty = {"docs": {}, "profile": None, "profile_meta": {}, "extract_cache": {}}
    path = _path(session_id)
    if not path.exists():
        return dict(empty)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt store must not brick the app — start clean.
        return dict(empty)

    docs: dict[str, Document] = {}
    for d in raw.get("docs", []):
        try:
            docs[d["id"]] = Document(
                id=d["id"], title=d.get("title", ""), text=d.get("text", ""),
                kind=d.get("kind", "text"), error=d.get("error", ""),
            )
        except (KeyError, TypeError):
            continue
    return {
        "docs": docs,
        "profile": raw.get("profile"),
        "profile_meta": raw.get("profile_meta") or {},
        "extract_cache": raw.get("extract_cache") or {},
    }


def save(session_id: str, docs: dict[str, Document], profile: dict | None,
         profile_meta: dict, extract_cache: dict | None = None) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "docs": [
            {"id": d.id, "title": d.title, "text": d.text, "kind": d.kind, "error": d.error}
            for d in docs.values()
        ],
        "profile": profile,
        "profile_meta": profile_meta,
        # Per-passage extracts, so a rate-limited profile build resumes instead
        # of restarting. Keyed by content hash, so it stays valid across
        # re-uploads of the same document.
        "extract_cache": extract_cache or {},
    }
    # Write to a temp file in the same directory, then replace — a crash
    # mid-write leaves the previous good store intact rather than a truncated one.
    path = _path(session_id)
    fd, tmp = tempfile.mkstemp(dir=str(STORE_DIR), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def clear(session_id: str) -> None:
    _path(session_id).unlink(missing_ok=True)
