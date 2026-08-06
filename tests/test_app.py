"""
End-to-end tests through the FastAPI app.

These run against the REAL project corpus (data/projects.json[.gz], 3,359 rows)
and the real prefilter index. Only the LLM boundary is faked, so everything else
— upload, parsing, chunking, prefilter, judge wiring, export — is exercised
exactly as it runs in production.

The app keeps nothing between requests, so these tests carry the workspace the
way the browser does: parsed documents and the built profile go out in the
request body. That is the contract, and `test_the_server_keeps_nothing` is the
test that pins it.
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
from engine import matching, profile as profile_mod  # noqa: E402

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client():
    with TestClient(app_mod.app) as c:
        yield c


@pytest.fixture
def fake_llm(monkeypatch):
    """Profile extraction returns a fixed DevOps/LLM profile; the judge scores
    every project it is shown, descending, so ranking is checkable."""

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

    async def judge_call(client, cfg, system, user, max_tokens=2048):
        judged = json.loads(user.split("PROJECTS TO JUDGE:\n")[-1])
        return {"verdicts": [
            {"projectId": p["id"], "score": 90 - i, "rationale": "strong overlap",
             "evidence": ["vLLM benchmark"], "gaps": ["domain ramp-up"]}
            for i, p in enumerate(judged)
        ]}

    monkeypatch.setattr(profile_mod, "json_call", profile_call)
    monkeypatch.setattr(matching, "json_call", judge_call)


# --------------------------------------------------------------------------- #
# helpers — the browser's job, done by hand
# --------------------------------------------------------------------------- #
MD = b"# Skills\n\nKubernetes, Helm, vLLM, KV cache, FastAPI.\n"


def upload_md(client, text=MD):
    """Returns the parsed documents, exactly as the browser would store them."""
    r = client.post("/api/kb", files={"files": ("kb.md", text, "text/markdown")})
    assert r.status_code == 200
    return r.json()["documents"]


def build_profile(client, docs):
    r = client.post("/api/profile", headers=KEY, json={"documents": docs})
    assert r.status_code == 200, r.text
    return r.json()["profile"]


def workspace(client):
    docs = upload_md(client)
    return build_profile(client, docs)


# --------------------------------------------------------------------------- #
# corpus and pages
# --------------------------------------------------------------------------- #
def test_health_exposes_the_real_corpus(client):
    r = client.get("/api/health").json()
    assert r["ok"] is True
    assert r["projects"] == 3359, "the full Mitacs corpus should be loaded"
    assert "Ontario" in r["provinces"] and "Québec" in r["provinces"]


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
    docs = upload_md(client)
    r = client.post("/api/profile", headers=KEY, json={"provider": "bogus", "documents": docs})
    assert r.status_code == 400


def test_landing_and_app_pages_render(client):
    landing = client.get("/")
    assert landing.status_code == 200 and "MITACS" in landing.text
    # The landing page is for an applicant, not an engineer: it must lead with
    # the task, and must not leak pipeline vocabulary.
    body = landing.text
    assert "Find my projects" in body
    for jargon in ("map-reduce", "embeddings", "BM25", "IDF", "vector", "fan-out"):
        assert jargon.lower() not in body.lower(), f"landing copy leaks {jargon!r}"
    ui = client.get("/app")
    assert ui.status_code == 200 and "KNOWLEDGE BASE" in ui.text
    css = client.get("/brutal.css")
    assert css.status_code == 200 and "--radius: 0rem" in css.text


# --------------------------------------------------------------------------- #
# ingestion
# --------------------------------------------------------------------------- #
def test_upload_parses_the_real_cv_pdf(client):
    cv = ROOT.parent / "resume_haseeb.pdf"
    if not cv.exists():
        pytest.skip("resume_haseeb.pdf not present")
    r = client.post("/api/kb", files={"files": ("resume_haseeb.pdf", cv.read_bytes(), "application/pdf")})
    assert r.status_code == 200
    doc = r.json()["documents"][0]
    assert doc["error"] == "", f"real CV failed to parse: {doc['error']}"
    assert doc["chars"] > 1000, "extracted suspiciously little text from the CV"


def test_upload_returns_the_text_not_just_a_handle(client):
    """The client is the only thing that remembers a document, so the parsed
    text has to come back with it. Returning metadata alone would strand it."""
    doc = upload_md(client)[0]
    assert doc["text"].strip(), "parsed text must be returned to the caller"
    assert "Kubernetes" in doc["text"]
    assert doc["chars"] == len(doc["text"])


def test_unreadable_upload_is_reported_not_fatal(client):
    r = client.post("/api/kb", files={"files": ("broken.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 200
    doc = next(d for d in r.json()["documents"] if d["title"] == "broken.pdf")
    assert doc["error"] != ""


def test_pasted_text_becomes_a_document(client):
    r = client.post("/api/kb/text", json={"title": "Notes", "text": "vLLM benchmarking on an RTX 3060."})
    assert r.status_code == 200
    doc = next(d for d in r.json()["documents"] if d["title"] == "Notes")
    assert doc["kind"] == "pasted" and doc["chars"] > 10


def test_empty_paste_is_rejected(client):
    assert client.post("/api/kb/text", json={"title": "x", "text": "  "}).status_code == 400


def test_csv_upload_is_parsed_into_labelled_rows(client):
    csv = b"skill,level\nKubernetes,demonstrated\nvLLM,demonstrated\n"
    r = client.post("/api/kb", files={"files": ("skills.csv", csv, "text/csv")})
    doc = next(d for d in r.json()["documents"] if d["title"] == "skills.csv")
    assert doc["error"] == "" and doc["kind"] == "csv" and doc["chars"] > 20


def test_binary_upload_is_refused_clearly(client):
    r = client.post("/api/kb", files={"files": ("img.bin", bytes(range(256)), "application/octet-stream")})
    doc = next(d for d in r.json()["documents"] if d["title"] == "img.bin")
    assert "unsupported" in doc["error"] or "decode" in doc["error"]


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #
def test_profile_requires_documents(client, fake_llm):
    assert client.post("/api/profile", headers=KEY, json={}).status_code == 400


def test_profile_requires_api_key(client):
    assert client.post("/api/profile", json={}).status_code == 401


def test_match_requires_a_profile(client):
    r = client.post("/api/match", headers=KEY, json={"topN": 5})
    assert r.status_code == 400


def test_export_without_matches_is_rejected(client):
    assert client.post("/api/export", json={}).status_code == 400


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def test_full_pipeline_end_to_end(client, fake_llm):
    profile = workspace(client)
    assert profile["skills"], "profile should not be empty"

    r = client.post("/api/match", headers=KEY,
                    json={"topN": 5, "width": 40, "profile": profile})
    assert r.status_code == 200, r.text
    result = r.json()

    assert len(result["matches"]) == 5
    assert result["stats"]["screened"] == 40, "filter width should be honoured"

    scores = [m["score"] for m in result["matches"]]
    assert scores == sorted(scores, reverse=True), "results must be ranked by score"
    for m in result["matches"]:
        assert m["project"]["id"] and m["project"]["title"]
        assert m["rationale"]


def test_the_call_count_follows_the_width(client, fake_llm):
    """The whole cost model: one filter pass for free, then every survivor read
    in batches. Nothing else spends the user's quota."""
    profile = workspace(client)
    width = 30
    r = client.post("/api/match", headers=KEY,
                    json={"topN": 5, "width": width, "profile": profile}).json()

    expected = -(-width // matching.JUDGE_BATCH_SIZE)
    assert r["stats"]["judgeCalls"] == expected
    assert r["stats"]["screened"] == width


def test_province_filter_restricts_results(client, fake_llm):
    profile = workspace(client)
    r = client.post("/api/match", headers=KEY,
                    json={"topN": 5, "width": 30, "provinces": ["Ontario"], "profile": profile}).json()

    assert r["matches"], "Ontario should yield matches"
    assert all(m["project"]["province"] == "Ontario" for m in r["matches"])


def test_results_carry_every_field_needed_to_read_a_project(client, fake_llm):
    """Globalink's detail view is a modal with no URL of its own, so the app has
    to be able to show the whole project itself rather than linking out."""
    profile = workspace(client)
    r = client.post("/api/match", headers=KEY,
                    json={"topN": 3, "width": 25, "profile": profile}).json()

    assert r["matches"]
    for m in r["matches"]:
        p = m["project"]
        for field in ("id", "title", "description", "supervisor",
                      "university", "province", "language", "startDate"):
            assert field in p, f"result is missing {field!r}, needed for the detail view"
        assert p["title"].strip(), "an empty title would break the Globalink title search"


# --------------------------------------------------------------------------- #
# statelessness — the property the deployment depends on
# --------------------------------------------------------------------------- #
def test_the_server_keeps_nothing(client, fake_llm):
    """Every request must stand alone. If any of this leaked into process state,
    the app would break the moment it ran on more than one instance — and a
    shared deployment would be holding strangers' CVs."""
    profile = workspace(client)

    # A match works with no prior request having "set up" a session.
    fresh = TestClient(app_mod.app)
    r = fresh.post("/api/match", headers=KEY,
                   json={"topN": 3, "width": 25, "profile": profile})
    assert r.status_code == 200, "a match must not depend on earlier requests"
    assert len(r.json()["matches"]) == 3

    # And the same request without the profile fails, proving the profile came
    # from the body rather than from anything the server remembered.
    assert client.post("/api/match", headers=KEY,
                       json={"topN": 3, "width": 25}).status_code == 400

    assert not hasattr(app_mod, "SESSIONS"), "no session registry should exist"
    assert not hasattr(app_mod, "JOBS"), "no job registry should exist"


def test_uploads_are_not_retained_server_side(client):
    """The upload endpoint is a parser, not a store."""
    upload_md(client)
    kb_dir = ROOT / "data" / "kb"
    written = list(kb_dir.glob("*.json")) if kb_dir.exists() else []
    before = {p: p.stat().st_mtime for p in written}
    upload_md(client, b"# Other\n\nSomething else entirely.\n")
    after = {p: p.stat().st_mtime for p in written}
    assert before == after, "uploading must not write to the old on-disk store"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_renders_the_run_it_is_given(client, fake_llm):
    profile = workspace(client)
    run = client.post("/api/match", headers=KEY,
                      json={"topN": 3, "width": 25, "profile": profile}).json()

    md = client.post("/api/export", json={**run, "provider": "openrouter", "model": "m"}).json()["markdown"]
    assert md.startswith("# Mitacs project matches")
    assert md.count("\n## ") == 3, "one heading per match"
    assert "Evidence from your profile" in md


def test_export_explains_how_to_open_a_project(client, fake_llm):
    profile = workspace(client)
    run = client.post("/api/match", headers=KEY,
                      json={"topN": 2, "width": 25, "profile": profile}).json()

    md = client.post("/api/export", json=run).json()["markdown"]
    assert "globalink.mitacs.ca/#/student/application/projects" in md
    assert "Keyword search" in md
    assert md.count("Search this title on Globalink:") == 2
