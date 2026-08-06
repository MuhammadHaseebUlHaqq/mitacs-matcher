"""
Vercel entrypoint.

Vercel's Python runtime looks for an ASGI callable named `app` in a file under
`api/`. The application itself lives at the repo root and knows nothing about
this — it is still an ordinary uvicorn app, which is what keeps `python app.py`
working locally and the test suite importing it directly.
"""

import sys
from pathlib import Path

# The function's working directory is not the repo root, so make the root
# importable before pulling in the app.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402,F401
