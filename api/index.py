"""
Vercel entrypoint.

Vercel's Python runtime looks for an ASGI callable named `app` in a file under
`api/`. The application itself lives at the repo root and knows nothing about
this — it is still an ordinary uvicorn app, which is what keeps `python app.py`
working locally and the test suite importing it directly.

The path shim below is the part that is easy to get wrong. `vercel.json` rewrites
every incoming URL to `/api/index`, and the runtime hands the function that
rewritten path rather than the original one — so FastAPI sees `/api/index` for a
request to `/` and returns its own 404. Stripping the prefix restores the real
route. It is written to be a no-op if the path arrives unmangled, so the same
file works whichever way the platform behaves.
"""

import sys
from pathlib import Path

# The function's working directory is not the repo root, so make the root
# importable before pulling in the app.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app as fastapi_app  # noqa: E402

_PREFIX = "/api/index"


def _strip(path: str) -> str:
    if path == _PREFIX:
        return "/"
    if path.startswith(_PREFIX + "/"):
        return path[len(_PREFIX):]
    return path


async def app(scope, receive, send):
    if scope["type"] in ("http", "websocket"):
        scope = dict(scope)
        scope["path"] = _strip(scope.get("path", "/"))
        raw = scope.get("raw_path")
        if raw:
            scope["raw_path"] = _strip(raw.decode("latin-1")).encode("latin-1")
    await fastapi_app(scope, receive, send)
