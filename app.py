"""
Mitacs Matcher — FastAPI app.

Bring-your-own-key, following Galt RAG: the provider key arrives from the
browser on each request, is used for that request, and is never stored, cached,
or logged. Two providers are supported — OpenRouter and Google Gemini — behind
one `LLMConfig`.

Knowledge bases persist to `data/kb/<session>.json` (parsed text + cached
profile) so they survive a restart, mirroring how bideez keeps knowledge
documents on the workspace row. Keys are never part of that file.

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

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from engine import store
from engine.corpus import Document, assemble
from engine.ingest import from_pasted_text, ingest
from engine.llm import DEFAULT_MODELS, PROVIDERS, LLMConfig, list_models
from engine.matching import MatchRun, run_matching
from engine.prefilter import Prefilter
from engine.profile import build_profile, render_profile

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "projects.json"

app = FastAPI(title="Mitacs Matcher")

PROJECTS: list[dict] = json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else []
PREFILTER = Prefilter(PROJECTS) if PROJECTS else None


# --------------------------------------------------------------------------- #
# Session state — in memory, backed by the on-disk store
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    docs: dict[str, Document] = field(default_factory=dict)
    profile: dict | None = None
    profile_meta: dict = field(default_factory=dict)
    last_run: MatchRun | None = None
    last_params: dict = field(default_factory=dict)


SESSIONS: dict[str, Session] = {}
JOBS: dict[str, asyncio.Queue] = {}


def session_for(sid: str | None) -> Session:
    if not sid:
        raise HTTPException(400, "Missing X-Session-Id header.")
    if sid not in SESSIONS:
        loaded = store.load(sid)
        SESSIONS[sid] = Session(
            docs=loaded["docs"], profile=loaded["profile"], profile_meta=loaded["profile_meta"],
        )
    return SESSIONS[sid]


def persist(sid: str, s: Session) -> None:
    try:
        store.save(sid, s.docs, s.profile, s.profile_meta)
    except OSError:
        # Persistence is a convenience; never fail a request because the disk
        # store could not be written.
        pass


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
def _doc_view(d: Document) -> dict:
    return {"id": d.id, "title": d.title, "kind": d.kind, "chars": len(d.text), "error": d.error}


def _kb_view(s: Session) -> dict:
    docs = [_doc_view(d) for d in s.docs.values()]
    return {
        "documents": docs,
        "hasProfile": s.profile is not None,
        "profileMeta": s.profile_meta,
        "totalChars": sum(d["chars"] for d in docs),
        "readable": sum(1 for d in docs if not d["error"]),
    }


@app.get("/api/kb")
async def kb_list(x_session_id: str | None = Header(None)) -> dict:
    return _kb_view(session_for(x_session_id))


@app.post("/api/kb")
async def kb_upload(files: list[UploadFile], x_session_id: str | None = Header(None)) -> dict:
    s = session_for(x_session_id)
    for f in files:
        doc = ingest(f.filename or "untitled", await f.read())
        s.docs[doc.id] = doc
    # New material invalidates the cached profile.
    s.profile, s.profile_meta = None, {}
    persist(x_session_id or "", s)
    return _kb_view(s)


@app.post("/api/kb/text")
async def kb_paste(body: dict = Body(...), x_session_id: str | None = Header(None)) -> dict:
    s = session_for(x_session_id)
    doc = from_pasted_text(body.get("title") or "Pasted note", body.get("text") or "")
    if doc.error:
        raise HTTPException(400, doc.error)
    s.docs[doc.id] = doc
    s.profile, s.profile_meta = None, {}
    persist(x_session_id or "", s)
    return _kb_view(s)


@app.delete("/api/kb/{doc_id}")
async def kb_delete(doc_id: str, x_session_id: str | None = Header(None)) -> dict:
    s = session_for(x_session_id)
    s.docs.pop(doc_id, None)
    s.profile, s.profile_meta = None, {}
    persist(x_session_id or "", s)
    return _kb_view(s)


@app.delete("/api/kb")
async def kb_clear(x_session_id: str | None = Header(None)) -> dict:
    s = session_for(x_session_id)
    s.docs.clear()
    s.profile, s.profile_meta = None, {}
    store.clear(x_session_id or "")
    return _kb_view(s)


# --------------------------------------------------------------------------- #
# Stage 0 — profile
# --------------------------------------------------------------------------- #
@app.post("/api/profile")
async def profile_build(
    body: dict = Body(default={}),
    x_session_id: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> dict:
    s = session_for(x_session_id)
    cfg = config_from(body.get("provider"), x_api_key, body.get("model"))

    usable = [d for d in s.docs.values() if not d.error and d.text.strip()]
    if not usable:
        raise HTTPException(400, "No readable documents in the knowledge base.")

    result = await build_profile(assemble(usable), cfg)
    if result.is_empty:
        raise HTTPException(422, result.warning or "No profile information could be extracted.")

    s.profile = result.profile
    s.profile_meta = {
        "units": result.n_units, "hits": result.n_hits, "failed": result.n_failed,
        "reduceLayers": result.reduce_layers, "warning": result.warning,
        "model": cfg.model, "provider": cfg.provider,
    }
    persist(x_session_id or "", s)
    return {"profile": result.profile, "meta": s.profile_meta, "rendered": render_profile(result.profile)}


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
async def match_start(
    body: dict = Body(default={}),
    x_session_id: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> dict:
    s = session_for(x_session_id)
    cfg = config_from(body.get("provider"), x_api_key, body.get("model"))
    if not s.profile:
        raise HTTPException(400, "Build the capability profile first.")

    top_n = max(1, min(int(body.get("topN") or 50), 200))
    width = int(body.get("width", 400))

    pool = _filtered_projects(body)
    if not pool:
        raise HTTPException(400, "No projects match those filters.")

    job_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue = asyncio.Queue()
    JOBS[job_id] = queue

    async def work() -> None:
        def note(m: str) -> None:
            queue.put_nowait({"type": "progress", "message": m})

        try:
            if width > 0 and PREFILTER is not None:
                note(f"Narrowing {len(pool)} projects to the top {width} by term overlap…")
                if len(pool) == len(PROJECTS):
                    screened = [PROJECTS[i] for i, _ in PREFILTER.rank(s.profile or {}, width)]
                else:
                    sub = Prefilter(pool)
                    screened = [pool[i] for i, _ in sub.rank(s.profile or {}, width)]
            else:
                note(f"Exhaustive mode — every one of {len(pool)} projects will be read.")
                screened = pool

            result = await run_matching(
                render_profile(s.profile or {}), screened, cfg, top_n, progress=note,
            )
            s.last_run = result
            s.last_params = {"topN": top_n, "width": width, "model": cfg.model,
                             "provider": cfg.provider, "poolSize": len(pool)}
            queue.put_nowait({"type": "done", "result": _run_view(result, len(pool))})
        except Exception as e:  # noqa: BLE001 — report, never swallow
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            queue.put_nowait({"type": "_eof"})
            # If the browser never opens the SSE stream (tab closed mid-run), the
            # stream handler never gets to pop this job. Reap it so a long-lived
            # server does not accumulate finished queues.
            asyncio.get_running_loop().call_later(300, JOBS.pop, job_id, None)

    asyncio.create_task(work())
    return {"jobId": job_id}


def _run_view(run: MatchRun, pool_size: int) -> dict:
    return {
        "matches": [
            {"rank": i + 1, "score": m.score, "rationale": m.rationale,
             "evidence": m.evidence, "gaps": m.gaps, "project": m.project}
            for i, m in enumerate(run.matches)
        ],
        "stats": {
            "poolSize": pool_size, "screened": run.n_screened,
            "candidates": run.n_candidates, "judged": run.n_judged,
            "mapCalls": run.map_calls, "judgeCalls": run.judge_calls,
            "failedMap": run.n_failed_map, "failedJudge": run.n_failed_judge,
            "droppedIds": run.dropped_ids,
        },
        "warning": run.warning,
    }


@app.get("/api/match/{job_id}/events")
async def match_events(job_id: str) -> StreamingResponse:
    queue = JOBS.get(job_id)
    if queue is None:
        raise HTTPException(404, "Unknown job.")

    async def stream():
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "_eof":
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            JOBS.pop(job_id, None)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@app.post("/api/export")
async def export(x_session_id: str | None = Header(None)) -> JSONResponse:
    s = session_for(x_session_id)
    if not s.last_run or not s.last_run.matches:
        raise HTTPException(400, "Nothing to export — run a match first.")

    p = s.last_params
    lines = [
        "# Mitacs project matches",
        "",
        f"*{len(s.last_run.matches)} projects, ranked. "
        f"Screened {s.last_run.n_screened} of {p.get('poolSize', len(PROJECTS))} · "
        f"{p.get('provider')} `{p.get('model')}`.*",
        "",
    ]
    if s.last_run.warning:
        lines += [f"> ⚠️ {s.last_run.warning}", ""]

    for i, m in enumerate(s.last_run.matches, start=1):
        pr = m.project
        lines += [
            f"## {i}. [{m.score}] {pr.get('title')}", "",
            f"**ID** `{pr.get('id')}` · **{pr.get('supervisor')}** — {pr.get('university')} "
            f"({pr.get('province')}) · {pr.get('language')}", "",
            f"{m.rationale}", "",
        ]
        if m.evidence:
            lines += ["**Evidence from your profile:**"] + [f"- {e}" for e in m.evidence] + [""]
        if m.gaps:
            lines += ["**Gaps to close:**"] + [f"- {g}" for g in m.gaps] + [""]

    return JSONResponse({"markdown": "\n".join(lines)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8093")))
