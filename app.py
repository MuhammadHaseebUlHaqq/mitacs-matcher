"""
Mitacs Matcher — FastAPI app.

Bring-your-own-key: the provider key arrives from the browser on each request,
is used for that request, and is never stored, cached, or logged. Two providers
are supported — OpenRouter and Google Gemini — behind one `LLMConfig`.

The server keeps nothing between requests. The browser holds the parsed
documents, the profile and the passage cache, and sends them with each call —
the same way the key already worked. That is what lets this run on a host with
no disk, and it means a shared deployment never holds anyone's CV.

Endpoints
  GET  /                          landing page
  GET  /app                       the matcher UI
  GET  /api/health                liveness + corpus size
  GET  /api/models                model catalogue for a provider
  GET  /api/kb                    list documents
  POST /api/kb                    upload documents (multipart)
  POST /api/kb/text               add a pasted note
  DEL  /api/kb/{doc_id}           remove one document
  DEL  /api/kb                    clear the knowledge base
  POST /api/profile               build the capability profile (Stage 0)
  POST /api/match                 start a match run -> job id
  GET  /api/match/{job_id}/events SSE: progress, then the final result
  POST /api/export                render the last result as Markdown
"""

from __future__ import annotations

import gzip
import json
import os
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from engine.corpus import Document, assemble
from engine.ingest import from_pasted_text, ingest
from engine.llm import DEFAULT_MODELS, MAX_CONCURRENCY, PROVIDERS, LLMConfig, list_models
from engine.matching import JUDGE_BATCH_SIZE, MatchRun, run_matching
from engine.prefilter import Prefilter
from engine.profile import build_profile, render_profile

ROOT = Path(__file__).parent
# The corpus ships gzipped: 12 MB of JSON compresses ~3.4x, which keeps clones
# and image builds small. The uncompressed file still wins if present, so a
# local scrape drops in without a re-zip.
DATA = ROOT / "data" / "projects.json"
DATA_GZ = ROOT / "data" / "projects.json.gz"

# How many projects survive the lexical filter and get read by the model. This
# single number is the cost of a run: at JUDGE_BATCH_SIZE 6 it decides the call
# count, the wall-clock time, and how much of the user's free quota a match
# spends. 60 costs ~10 calls.
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "60"))

app = FastAPI(title="Mitacs Matcher")


def _load_corpus() -> list[dict]:
    if DATA.exists():
        return json.loads(DATA.read_text(encoding="utf-8"))
    if DATA_GZ.exists():
        with gzip.open(DATA_GZ, "rt", encoding="utf-8") as f:
            return json.load(f)
    return []


PROJECTS: list[dict] = _load_corpus()
PREFILTER = Prefilter(PROJECTS) if PROJECTS else None


# --------------------------------------------------------------------------- #
# Session state — there is none
# --------------------------------------------------------------------------- #
# The server holds nothing between requests. The browser owns the workspace:
# parsed document text and the built profile live in its localStorage and travel
# with each request, exactly as the API key already did. Two things fall out of
# that. It runs unchanged on a host with no persistent disk, and a shared
# deployment cannot leak one user's CV to another because it never has it.
#
# The cost is that a request carries more: a profile is a few KB, and the parsed
# text of a CV is tens of KB. Both are far below any practical body limit.


def _documents_from(body: dict) -> list[Document]:
    """Rebuild the knowledge base from what the client sent.

    Text arrives already parsed — extraction happened server-side at upload and
    the result was handed back to the browser to keep.
    """
    out: list[Document] = []
    for d in body.get("documents") or []:
        if not isinstance(d, dict):
            continue
        text = str(d.get("text") or "")
        if not text.strip():
            continue
        out.append(Document(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            title=str(d.get("title") or "untitled"),
            text=text,
            kind=str(d.get("kind") or "text"),
            error=str(d.get("error") or ""),
        ))
    return out


def config_from(provider: str | None, key: str | None, model: str | None) -> LLMConfig:
    provider = (provider or "openrouter").strip().lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}.")
    if not key or not key.strip():
        label = "OpenRouter" if provider == "openrouter" else "Gemini"
        raise HTTPException(401, f"Missing {label} API key. Add it in the sidebar.")
    return LLMConfig(
        provider=provider,
        api_key=key.strip(),
        model=(model or "").strip() or DEFAULT_MODELS[provider],
    )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
async def landing() -> FileResponse:
    return FileResponse(ROOT / "static" / "landing.html")


@app.get("/app")
async def matcher_ui() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/brutal.css")
async def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "static" / "brutal.css", media_type="text/css")


# Both pages load this, so the PostHog config lives in one file rather than
# being pasted twice and drifting.
@app.get("/analytics.js")
async def analytics() -> FileResponse:
    return FileResponse(ROOT / "static" / "analytics.js", media_type="application/javascript")


# `/favicon.ico` is served the same SVG. The pages declare the icon explicitly,
# so a browser that honours the link never asks for the .ico — but crawlers and
# some clients request it regardless, and answering is cheaper than the 404s it
# otherwise leaves in the logs.
@app.get("/favicon.svg")
@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(ROOT / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "projects": len(PROJECTS),
        "providers": list(PROVIDERS),
        "defaultModels": DEFAULT_MODELS,
        "provinces": sorted({p.get("province", "") for p in PROJECTS if p.get("province")}),
    }


@app.get("/api/models")
async def models(provider: str = "openrouter", x_api_key: str | None = Header(None)) -> dict:
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'.")
    return {"provider": provider, "models": await list_models(provider, (x_api_key or "").strip())}


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
def _doc_payload(d: Document) -> dict:
    """A parsed document, including its text — the client stores this and sends
    it back, so the text has to go out with it."""
    return {"id": d.id, "title": d.title, "kind": d.kind, "text": d.text,
            "chars": len(d.text), "error": d.error}


@app.post("/api/kb")
async def kb_upload(files: list[UploadFile]) -> dict:
    """Parse uploads and hand the text straight back. Nothing is retained."""
    return {"documents": [
        _doc_payload(ingest(f.filename or "untitled", await f.read())) for f in files
    ]}


@app.post("/api/kb/text")
async def kb_paste(body: dict = Body(...)) -> dict:
    doc = from_pasted_text(body.get("title") or "Pasted note", body.get("text") or "")
    if doc.error:
        raise HTTPException(400, doc.error)
    return {"documents": [_doc_payload(doc)]}


# --------------------------------------------------------------------------- #
# Stage 0 — profile
# --------------------------------------------------------------------------- #
@app.post("/api/profile")
async def profile_build(
    body: dict = Body(default={}),
    x_api_key: str | None = Header(None),
) -> dict:
    cfg = config_from(body.get("provider"), x_api_key, body.get("model"))

    usable = [d for d in _documents_from(body) if not d.error and d.text.strip()]
    if not usable:
        raise HTTPException(400, "No readable documents were sent.")

    # The per-passage extract cache rides along with the request too, so a build
    # stopped by a rate limit still resumes where it left off rather than
    # re-spending quota on passages that already succeeded.
    cache = body.get("extractCache") if isinstance(body.get("extractCache"), dict) else {}
    result = await build_profile(assemble(usable), cfg, cache=cache)

    if result.is_empty:
        # Hand the partial cache back regardless — it is what makes the retry cheap.
        return JSONResponse(
            status_code=422,
            content={"detail": result.warning or "No profile information could be extracted.",
                     "extractCache": result.cache},
        )

    meta = {
        "units": result.n_units, "hits": result.n_hits, "failed": result.n_failed,
        "cached": result.n_cached, "reduceLayers": result.reduce_layers,
        "warning": result.warning, "model": cfg.model, "provider": cfg.provider,
    }
    return {"profile": result.profile, "meta": meta, "extractCache": result.cache,
            "rendered": render_profile(result.profile)}


# --------------------------------------------------------------------------- #
# Stages 1-3 — match
# --------------------------------------------------------------------------- #
def _filtered_projects(body: dict) -> list[dict]:
    provinces = set(body.get("provinces") or [])
    language = (body.get("language") or "").strip()
    out = PROJECTS
    if provinces:
        out = [p for p in out if p.get("province") in provinces]
    if language:
        out = [p for p in out if (p.get("language") or "").lower() == language.lower()]
    return out


@app.post("/api/match")
async def match(
    body: dict = Body(default={}),
    x_api_key: str | None = Header(None),
) -> dict:
    """Filter, judge and rank — in one request.

    This used to be a job: a background task pushing progress over SSE, because
    a run was ~101 model calls and no browser waits that long on one response.
    Collapsing the pipeline to a single judge pass (~10 calls, well under a
    minute) removed the reason for that machinery, and with it the in-memory job
    registry that made the app impossible to run on more than one instance.
    """
    cfg = config_from(body.get("provider"), x_api_key, body.get("model"))

    profile = body.get("profile")
    if not isinstance(profile, dict) or not profile:
        raise HTTPException(400, "Build the capability profile first.")

    top_n = max(1, min(int(body.get("topN") or 20), 100))
    width = int(body.get("width", DEFAULT_WIDTH))

    pool = _filtered_projects(body)
    if not pool:
        raise HTTPException(400, "No projects match those filters.")

    # Narrow before spending anything. Every survivor is read by the model, so
    # this width is the entire cost control.
    if width > 0 and PREFILTER is not None:
        if len(pool) == len(PROJECTS):
            screened = [PROJECTS[i] for i, _ in PREFILTER.rank(profile, width)]
        else:
            screened = [pool[i] for i, _ in Prefilter(pool).rank(profile, width)]
    else:
        screened = pool

    run = await run_matching(render_profile(profile), screened, cfg, top_n)
    return _run_view(run, len(pool))


def _run_view(run: MatchRun, pool_size: int) -> dict:
    return {
        "matches": [
            {"rank": i + 1, "score": m.score, "rationale": m.rationale,
             "evidence": m.evidence, "gaps": m.gaps, "project": m.project}
            for i, m in enumerate(run.matches)
        ],
        "stats": {
            "poolSize": pool_size, "screened": run.n_screened, "judged": run.n_judged,
            "judgeCalls": run.judge_calls, "failedJudge": run.n_failed_judge,
            "droppedIds": run.dropped_ids,
        },
        "warning": run.warning,
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@app.post("/api/export")
async def export(body: dict = Body(default={})) -> JSONResponse:
    """Render a finished run as markdown. The run travels in the body — the
    server kept no copy of it."""
    matches = body.get("matches") or []
    if not matches:
        raise HTTPException(400, "Nothing to export — run a match first.")

    stats = body.get("stats") or {}
    lines = [
        "# Mitacs project matches",
        "",
        f"*{len(matches)} projects, ranked. "
        f"Read {stats.get('screened', len(matches))} of "
        f"{stats.get('poolSize', len(PROJECTS))} · "
        f"{body.get('provider')} `{body.get('model')}`.*",
        "",
    ]
    if body.get("warning"):
        lines += [f"> ⚠️ {body['warning']}", ""]
    # Globalink's project detail is a modal, so there is no per-project URL to
    # link to. Searching the exact title returns that project alone, so that is
    # the hand-off we give.
    lines += [
        "> To open any project: go to "
        "<https://globalink.mitacs.ca/#/student/application/projects> and paste the "
        "project title into **Keyword search**. The exact title matches only that project.",
        "",
    ]

    for i, m in enumerate(matches, start=1):
        pr = m.get("project") or {}
        lines += [
            f"## {i}. [{m.get('score')}] {pr.get('title')}", "",
            f"**ID** `{pr.get('id')}` · **{pr.get('supervisor')}** — {pr.get('university')} "
            f"({pr.get('province')}) · {pr.get('language')}"
            + (f" · starts {pr.get('startDate')}" if pr.get("startDate") else ""), "",
            f"{m.get('rationale') or ''}", "",
            f"*Search this title on Globalink:* `{pr.get('title')}`", "",
        ]
        if m.get("evidence"):
            lines += ["**Evidence from your profile:**"] + [f"- {e}" for e in m["evidence"]] + [""]
        if m.get("gaps"):
            lines += ["**Gaps to close:**"] + [f"- {g}" for g in m["gaps"]] + [""]

    return JSONResponse({"markdown": "\n".join(lines)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8093")))
