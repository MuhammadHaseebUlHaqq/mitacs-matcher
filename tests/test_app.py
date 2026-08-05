"""
End-to-end tests through the FastAPI app.

These run against the REAL project corpus (data/projects.json, 3,359 rows) and
the real prefilter index. Only the LLM boundary is faked, so everything else —
upload, parsing, chunking, prefilter, map/judge wiring, SSE, export — is
exercised exactly as it runs in production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as app_mod  # noqa: E402
from engine import matching, profile as profile_mod, store as store_mod  # noqa: E402

SID = {"X-Session-Id": "test-session", "X-API-Key": "test-key"}


@pytest.fixture
def client():
    # Must be the context-manager form: it keeps ONE event loop alive across
    # requests, so the background match task started by POST /api/match is still
    # running when the SSE endpoint reads its queue. A bare TestClient() tears
    # the loop down after each request and orphans the task. Under uvicorn there
    # is a single long-lived loop, so this matches production behaviour.
    with TestClient(app_mod.app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_session(tmp_path, monkeypatch):
    # Knowledge bases now persist to disk, so clearing SESSIONS alone leaves the
    # previous test's documents on the store and the next test inherits them.
    # Redirect the store at a tmp dir so each test starts empty AND so the suite
    # never writes into the app's real data/kb/.
    monkeypatch.setattr(store_mod, "STORE_DIR", tmp_path / "kb")
    app_mod.SESSIONS.clear()
    yield
    app_mod.SESSIONS.clear()


@pytest.fixture
def fake_llm(monkeypatch):
    """Profile extraction returns a fixed DevOps/LLM profile; matching scores
    any project whose title mentions the themes we care about."""

    async def profile_call(client, cfg, system, user, max_tokens=2048):
        if system.startswith("You are a profile extractor"):
            return {
                "skills": [
                    {"name": "Kubernetes", "strength": "demonstrated", "snippet": "kubeadm cluster"},
                    {"name": "LLM inference optimization", "strength": "demonstrated", "snippet": "vLLM benchmark"},
                ],
                "domains": ["DevOps", "ML systems"],
                "projects": [{"name": "llm-inference-optimization", "what": "vLLM vs HuggingFace", "snippet": "KV cache"}],
                "experience": [{"role": "DevOps Intern", "org": "Systems Limited", "what": "AKS, Helm"}],
                "interests": ["LLM serving", "cloud native"],
                "achievements": [], "constraints": [],
            }
        return {  # reduce
            "skills": [
                {"name": "Kubernetes", "strength": "demonstrated", "snippet": "kubeadm cluster"},
                {"name": "LLM inference optimization", "strength": "demonstrated", "snippet": "vLLM benchmark"},
            ],
            "domains": ["DevOps", "ML systems"], "projects": [], "experience": [],
            "interests": ["LLM serving"], "achievements": [], "constraints": [],
        }

    async def match_call(client, cfg, system, user, max_tokens=2048):
        payload = json.loads(user.split("CANDIDATE PROJECTS:\n")[-1]) if "CANDIDATE PROJECTS:" in user else None
        if payload is not None:  # MAP
            return {"candidates": [
                {"projectId": p["id"], "fit": 70, "reason": "themes overlap",
                 "evidence": ["Kubernetes"], "concerns": []}
                for p in payload
            ]}
        judged = json.loads(user.split("PROJECTS TO JUDGE:\n")[-1])  # JUDGE
        return {"verdicts": [
            {"projectId": j["project"]["id"], "score": 90 - i, "rationale": "strong overlap",
             "evidence": ["vLLM benchmark"], "gaps": ["domain ramp-up"]}
            for i, j in enumerate(judged)
        ]}

    monkeypatch.setattr(profile_mod, "json_call", profile_call)
    monkeypatch.setattr(matching, "json_call", match_call)


# --------------------------------------------------------------------------- #
def test_health_exposes_the_real_corpus(client):
    r = client.get("/api/health").json()
    assert r["ok"] is True
    assert r["projects"] == 3359, "the full Mitacs corpus should be loaded"
    assert "Ontario" in r["provinces"] and "Québec" in r["provinces"]


def test_session_id_is_required(client):
    assert client.get("/api/kb").status_code == 400


def test_upload_parses_the_real_cv_pdf(client):
    cv = ROOT.parent / "resume_haseeb.pdf"
    if not cv.exists():
        pytest.skip("resume_haseeb.pdf not present")
    r = client.post("/api/kb", headers=SID, files={"files": ("resume_haseeb.pdf", cv.read_bytes(), "application/pdf")})
    assert r.status_code == 200
    doc = r.json()["documents"][0]
    assert doc["error"] == "", f"real CV failed to parse: {doc['error']}"
    assert doc["chars"] > 1000, "extracted suspiciously little text from the CV"


def test_unreadable_upload_is_reported_not_fatal(client):
    r = client.post("/api/kb", headers=SID, files={"files": ("broken.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 200
    doc = next(d for d in r.json()["documents"] if d["title"] == "broken.pdf")
    assert doc["error"] != ""


def test_profile_requires_documents(client, fake_llm):
    assert client.post("/api/profile", headers=SID, json={}).status_code == 400


def test_profile_requires_api_key(client):
    r = client.post("/api/profile", headers={"X-Session-Id": "s"}, json={})
    assert r.status_code == 401


def test_match_requires_a_profile(client):
    r = client.post("/api/match", headers=SID, json={"topN": 5})
    assert r.status_code == 400


def _upload_md(client, text=b"# Skills\n\nKubernetes, Helm, vLLM, KV cache, FastAPI.\n"):
    return client.post("/api/kb", headers=SID, files={"files": ("kb.md", text, "text/markdown")})


def _drain(client, job_id):
    events = []
    with client.stream("GET", f"/api/match/{job_id}/events") as resp:
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_full_pipeline_end_to_end(client, fake_llm):
    _upload_md(client)

    prof = client.post("/api/profile", headers=SID, json={})
    assert prof.status_code == 200
    assert prof.json()["profile"]["skills"], "profile should not be empty"

    start = client.post("/api/match", headers=SID, json={"topN": 5, "width": 40})
    assert start.status_code == 200
    events = _drain(client, start.json()["jobId"])

    assert any(e["type"] == "progress" for e in events), "should stream progress"
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1, f"expected one done event, got {[e['type'] for e in events]}"

    result = done[0]["result"]
    assert len(result["matches"]) == 5
    assert result["stats"]["screened"] == 40, "prefilter width should be honoured"

    scores = [m["score"] for m in result["matches"]]
    assert scores == sorted(scores, reverse=True), "results must be ranked by score"
    for m in result["matches"]:
        assert m["project"]["id"] and m["project"]["title"]
        assert m["rationale"]


def test_province_filter_restricts_results(client, fake_llm):
    _upload_md(client)
    client.post("/api/profile", headers=SID, json={})

    start = client.post("/api/match", headers=SID, json={"topN": 5, "width": 30, "provinces": ["Ontario"]})
    events = _drain(client, start.json()["jobId"])
    result = [e for e in events if e["type"] == "done"][0]["result"]

    assert result["matches"], "Ontario should yield matches"
    assert all(m["project"]["province"] == "Ontario" for m in result["matches"])


def test_top_n_is_capped_and_export_renders(client, fake_llm):
    _upload_md(client)
    client.post("/api/profile", headers=SID, json={})

    start = client.post("/api/match", headers=SID, json={"topN": 3, "width": 25})
    _drain(client, start.json()["jobId"])

    md = client.post("/api/export", headers=SID).json()["markdown"]
    assert md.startswith("# Mitacs project matches")
    assert md.count("\n## ") == 3, "one heading per match"
    assert "Evidence from your profile" in md


def test_export_before_a_run_is_rejected(client):
    assert client.post("/api/export", headers=SID).status_code == 400


def test_deleting_a_document_invalidates_the_profile(client, fake_llm):
    up = _upload_md(client)
    doc_id = up.json()["documents"][0]["id"]
    client.post("/api/profile", headers=SID, json={})
    assert client.get("/api/kb", headers=SID).json()["hasProfile"] is True

    client.delete(f"/api/kb/{doc_id}", headers=SID)
    assert client.get("/api/kb", headers=SID).json()["hasProfile"] is False


def test_exhaustive_width_screens_everything(client, fake_llm, monkeypatch):
    """width=0 must bypass the prefilter. Restricted to one small province so
    the test stays fast while still proving the branch."""
    _upload_md(client)
    client.post("/api/profile", headers=SID, json={})

    pei = [p for p in app_mod.PROJECTS if p.get("province") == "Prince Edward Island"]
    start = client.post("/api/match", headers=SID,
                        json={"topN": 3, "width": 0, "provinces": ["Prince Edward Island"]})
    events = _drain(client, start.json()["jobId"])
    result = [e for e in events if e["type"] == "done"][0]["result"]

    assert result["stats"]["screened"] == len(pei), "exhaustive mode must read every project in the pool"
    assert any("Exhaustive" in e.get("message", "") for e in events if e["type"] == "progress")


# --------------------------------------------------------------------------- #
# providers, persistence, new file types
# --------------------------------------------------------------------------- #
def test_health_lists_both_providers(client):
    r = client.get("/api/health").json()
    assert r["providers"] == ["openrouter", "gemini"]
    assert r["defaultModels"]["gemini"].startswith("gemini")


def test_gemini_catalogue_falls_back_without_a_key(client):
    r = client.get("/api/models?provider=gemini").json()
    assert r["provider"] == "gemini"
    assert any(m["id"].startswith("gemini") for m in r["models"])


def test_unknown_provider_is_rejected(client):
    assert client.get("/api/models?provider=bogus").status_code == 400
    _upload_md(client)
    r = client.post("/api/profile", headers=SID, json={"provider": "bogus"})
    assert r.status_code == 400


def test_knowledge_base_survives_session_eviction(client, fake_llm):
    """Persistence contract: dropping the in-memory session must not lose
    documents or the cached profile — they reload from disk."""
    _upload_md(client)
    client.post("/api/profile", headers=SID, json={})

    app_mod.SESSIONS.clear()          # simulate a restart

    kb = client.get("/api/kb", headers=SID).json()
    assert len(kb["documents"]) == 1, "documents should reload from the store"
    assert kb["hasProfile"] is True, "the cached profile should reload too"


def test_clearing_the_knowledge_base_wipes_the_store(client):
    _upload_md(client)
    client.delete("/api/kb", headers=SID)
    app_mod.SESSIONS.clear()
    assert client.get("/api/kb", headers=SID).json()["documents"] == []


def test_pasted_text_becomes_a_document(client):
    r = client.post("/api/kb/text", headers=SID,
                    json={"title": "Notes", "text": "vLLM benchmarking on an RTX 3060."})
    assert r.status_code == 200
    doc = next(d for d in r.json()["documents"] if d["title"] == "Notes")
    assert doc["kind"] == "pasted" and doc["chars"] > 10


def test_empty_paste_is_rejected(client):
    assert client.post("/api/kb/text", headers=SID, json={"title": "x", "text": "  "}).status_code == 400


def test_csv_upload_is_parsed_into_labelled_rows(client):
    csv = b"skill,level\nKubernetes,demonstrated\nvLLM,demonstrated\n"
    r = client.post("/api/kb", headers=SID, files={"files": ("skills.csv", csv, "text/csv")})
    doc = next(d for d in r.json()["documents"] if d["title"] == "skills.csv")
    assert doc["error"] == "" and doc["kind"] == "csv" and doc["chars"] > 20


def test_binary_upload_is_refused_clearly(client):
    r = client.post("/api/kb", headers=SID, files={"files": ("img.bin", bytes(range(256)), "application/octet-stream")})
    doc = next(d for d in r.json()["documents"] if d["title"] == "img.bin")
    assert "unsupported" in doc["error"] or "decode" in doc["error"]


def test_landing_and_app_pages_render(client):
    landing = client.get("/")
    assert landing.status_code == 200 and "MITACS" in landing.text
    ui = client.get("/app")
    assert ui.status_code == 200 and "KNOWLEDGE BASE" in ui.text
    css = client.get("/brutal.css")
    assert css.status_code == 200 and "--radius: 0rem" in css.text
