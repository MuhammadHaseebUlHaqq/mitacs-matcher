"""
Engine tests.

Every LLM call is faked, so the suite runs offline and for free. What is under
test is the pipeline's *contract*: chunking never loses text, cited ids are
resolved and unresolvable ones dropped, failures degrade honestly instead of
silently, and the score the user sees comes from the judge rather than the
batch-local map.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import matching, profile as profile_mod  # noqa: E402
from engine.corpus import Document, assemble, build_units  # noqa: E402
from engine.ingest import ingest  # noqa: E402
from engine.llm import CallError, LLMConfig, parse_json_loose  # noqa: E402
from engine.prefilter import Prefilter  # noqa: E402
from engine.profile import render_profile  # noqa: E402


# --------------------------------------------------------------------------- #
# llm.parse_json_loose
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Here you go:\n```\n{"a": 1}\n```', {"a": 1}),
    ('Sure! {"a": [1, 2]} hope that helps', {"a": [1, 2]}),
    ('{"a": "brace } inside string"}', {"a": "brace } inside string"}),
])
def test_parse_json_loose(raw, expected):
    assert parse_json_loose(raw) == expected


def test_parse_json_loose_rejects_garbage():
    with pytest.raises(CallError):
        parse_json_loose("no json here at all")


# --------------------------------------------------------------------------- #
# corpus — chunking
# --------------------------------------------------------------------------- #
def test_sections_are_not_split_when_they_fit():
    doc = Document(id="d1", title="CV", text="# A\n" + "x" * 100 + "\n\n# B\n" + "y" * 100)
    units = build_units([doc], ceiling=10_000)
    assert len(units) == 1, "small sections should pack into one unit"


def test_oversize_section_is_split_with_overlap_and_loses_nothing():
    body = "".join(f"sentence-{i} " for i in range(4000))
    units = build_units([Document(id="d1", title="Big", text=body)], ceiling=2000)
    assert len(units) > 1
    joined = "".join(u.text for u in units)
    # Overlap means the join is longer than the source, but every token survives.
    assert len(joined) > len(body)
    for probe in ("sentence-0 ", "sentence-1999 ", "sentence-3999 "):
        assert probe in joined


def test_units_carry_resolvable_ids_and_anchors():
    body = "z" * 5000
    units = build_units([Document(id="abc", title="Notes.pdf", text=body)], ceiling=2000)
    assert all(u.id.startswith("doc:abc#") for u in units)
    assert len({u.id for u in units}) == len(units), "ids must be unique"
    assert "part 1/" in (units[0].source_anchor or "")


def test_failed_documents_are_excluded_from_the_corpus():
    good = Document(id="g", title="ok.txt", text="real content here")
    bad = Document(id="b", title="broken.pdf", text="", error="no extractable text")
    corpus = assemble([good, bad])
    assert {u.source_id for u in corpus.units} == {"g"}


def test_breadcrumb_falls_back_to_document_title():
    units = build_units([Document(id="d", title="Resume.pdf", text="no headings, just prose")])
    assert units[0].breadcrumb == "Resume.pdf"


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def test_ingest_reports_errors_instead_of_raising():
    doc = ingest("empty.txt", b"")
    assert doc.error and doc.text == ""


def test_ingest_oversize_file_is_rejected_cleanly():
    doc = ingest("huge.txt", b"x" * (26 * 1024 * 1024))
    assert "exceeds" in doc.error


def test_ingest_markdown_roundtrip():
    doc = ingest("notes.md", b"# Title\n\nbody text")
    assert doc.error == "" and doc.kind == "md" and "body text" in doc.text


# --------------------------------------------------------------------------- #
# prefilter
# --------------------------------------------------------------------------- #
@pytest.fixture
def projects():
    return [
        {"id": "1", "title": "Efficient LLM Inference Serving", "description": "KV cache, vLLM, GPU memory, throughput and latency for large language model serving."},
        {"id": "2", "title": "Kubernetes Resource Management", "description": "Cloud-native orchestration, containers, autoscaling and energy-aware scheduling."},
        {"id": "3", "title": "Marine Biology of Coral Reefs", "description": "Coral bleaching, reef ecology, ocean temperature surveys and species counts."},
        {"id": "4", "title": "Medieval French Poetry", "description": "Manuscript analysis of verse forms in twelfth-century courtly literature."},
    ]


def test_prefilter_ranks_relevant_projects_first(projects):
    pf = Prefilter(projects)
    prof = {"skills": [{"name": "vLLM inference optimization", "strength": "demonstrated"}],
            "interests": ["LLM serving"], "domains": [], "projects": [], "experience": []}
    ranked = pf.rank(prof, width=2)
    assert ranked[0][0] == 0, "the LLM serving project should rank first"


def test_prefilter_width_zero_admits_everything(projects):
    pf = Prefilter(projects)
    assert len(pf.rank({"skills": []}, width=0)) == len(projects)


def test_prefilter_empty_profile_still_returns_candidates(projects):
    pf = Prefilter(projects)
    ranked = pf.rank({"skills": [], "interests": [], "domains": [], "projects": [], "experience": []}, width=3)
    assert len(ranked) == 3, "an empty profile must not silently return nothing"


def test_prefilter_never_exceeds_requested_width(projects):
    pf = Prefilter(projects)
    prof = {"skills": [{"name": "kubernetes containers", "strength": "used"}]}
    assert len(pf.rank(prof, width=1)) <= 1


# --------------------------------------------------------------------------- #
# profile — render
# --------------------------------------------------------------------------- #
def test_render_profile_includes_strength_and_truncates():
    prof = {"skills": [{"name": "Kubernetes", "strength": "demonstrated"}],
            "interests": ["ML systems"], "domains": [], "projects": [],
            "experience": [], "achievements": [], "constraints": []}
    out = render_profile(prof)
    assert "Kubernetes (demonstrated)" in out and "ML systems" in out

    fat = {"skills": [{"name": "x" * 200, "strength": "used"} for _ in range(200)]}
    assert len(render_profile(fat, max_chars=500)) <= 520


def test_render_profile_handles_plain_strings():
    assert "Docker" in render_profile({"skills": ["Docker"], "domains": ["DevOps"]})


# --------------------------------------------------------------------------- #
# matching — the pipeline contract
# --------------------------------------------------------------------------- #
CFG = LLMConfig(provider="openrouter", api_key="test-key", model="test-model")

MATCH_PROJECTS = [
    {"id": "p1", "title": "LLM Serving", "description": "inference", "studentSkills": "", "studentRoles": ""},
    {"id": "p2", "title": "Kubernetes", "description": "orchestration", "studentSkills": "", "studentRoles": ""},
    {"id": "p3", "title": "Coral Reefs", "description": "marine biology", "studentSkills": "", "studentRoles": ""},
]


def fake_json_call(map_reply, judge_reply, fail_map_on=(), fail_judge=False):
    """Build a `json_call` stand-in that answers map and judge calls differently."""
    calls = {"map": 0, "judge": 0}

    async def _call(client, cfg, system, user, max_tokens=2048):
        if system.startswith("You match a person"):
            i = calls["map"]; calls["map"] += 1
            if i in fail_map_on:
                raise CallError("simulated upstream rate-limit")
            return map_reply(user, i)
        calls["judge"] += 1
        if fail_judge:
            raise CallError("simulated judge failure")
        return judge_reply(user)

    return _call, calls


def run(coro):
    return asyncio.run(coro)


def test_judge_score_overrides_batch_local_map_score(monkeypatch):
    """The map `fit` must not leak into the ranking — only the judge's calibrated
    score is shown. Map says p3 is best; the judge says p1 is."""
    def map_reply(user, i):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"candidates": [{"projectId": pid, "fit": 99 if pid == "p3" else 10,
                                "reason": "r", "evidence": ["e"], "concerns": []} for pid in ids]}

    def judge_reply(user):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"verdicts": [{"projectId": pid, "score": 95 if pid == "p1" else 20,
                              "rationale": "judged", "evidence": ["ev"], "gaps": []} for pid in ids]}

    call, _ = fake_json_call(map_reply, judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.matches[0].project["id"] == "p1"
    assert res.matches[0].score == 95
    assert res.matches[0].rationale == "judged"


def test_unresolvable_project_ids_are_dropped_and_counted(monkeypatch):
    def map_reply(user, i):
        return {"candidates": [{"projectId": "GHOST", "fit": 90, "reason": "r", "evidence": [], "concerns": []}]}

    def judge_reply(user):
        return {"verdicts": []}

    call, _ = fake_json_call(map_reply, judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.dropped_ids > 0
    assert res.matches == []
    assert res.n_candidates == 0


def test_failed_map_calls_produce_a_partial_warning(monkeypatch):
    def map_reply(user, i):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"candidates": [{"projectId": pid, "fit": 50, "reason": "r", "evidence": [], "concerns": []} for pid in ids]}

    def judge_reply(user):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"verdicts": [{"projectId": pid, "score": 70, "rationale": "j", "evidence": [], "gaps": []} for pid in ids]}

    call, _ = fake_json_call(map_reply, judge_reply, fail_map_on=(0,))
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.n_failed_map == 1
    assert "Partial results" in res.warning
    assert "never read" in res.warning


def test_project_omitted_by_judge_is_kept_and_flagged(monkeypatch):
    def map_reply(user, i):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"candidates": [{"projectId": pid, "fit": 60, "reason": "r", "evidence": [], "concerns": []} for pid in ids]}

    def judge_reply(user):
        return {"verdicts": []}  # judge silently drops everything

    call, _ = fake_json_call(map_reply, judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert len(res.matches) == 3, "omitted projects must not vanish"
    assert all("not re-judged" in m.rationale for m in res.matches)


def test_top_n_is_respected(monkeypatch):
    def map_reply(user, i):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"candidates": [{"projectId": pid, "fit": 80, "reason": "r", "evidence": [], "concerns": []} for pid in ids]}

    def judge_reply(user):
        ids = [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]
        return {"verdicts": [{"projectId": pid, "score": 50 + int(pid[1]), "rationale": "j", "evidence": [], "gaps": []} for pid in ids]}

    call, _ = fake_json_call(map_reply, judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=1))
    assert len(res.matches) == 1
    assert res.matches[0].project["id"] == "p3", "highest judged score wins"


def test_empty_profile_short_circuits_without_calling_the_model(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("must not call the model with an empty profile")

    monkeypatch.setattr(matching, "json_call", boom)
    res = run(matching.run_matching("   ", MATCH_PROJECTS, CFG, top_n=5))
    assert res.matches == [] and "profile is empty" in res.warning


def test_map_declining_to_match_yields_no_results(monkeypatch):
    """A model that correctly says "none of these fit" must produce an empty
    result, not a forced ranking."""
    call, _ = fake_json_call(lambda u, i: {"candidates": []}, lambda u: {"verdicts": []})
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=5))
    assert res.matches == []
    assert "No project" in res.warning


# --------------------------------------------------------------------------- #
# profile stage — map/reduce behaviour
# --------------------------------------------------------------------------- #
def test_profile_reduce_falls_back_to_local_merge_on_failure(monkeypatch):
    """If a reduce call dies, the branch must degrade to a deterministic union
    rather than losing that evidence."""
    async def flaky(client, cfg, system, user, max_tokens=2048):
        if system.startswith("You are a profile extractor"):
            name = "K8s" if "alpha" in user else "vLLM"
            return {"skills": [{"name": name, "strength": "demonstrated", "snippet": "s"}],
                    "domains": [], "projects": [], "experience": [],
                    "interests": [], "achievements": [], "constraints": []}
        raise CallError("reduce exploded")

    monkeypatch.setattr(profile_mod, "json_call", flaky)

    docs = [Document(id="a", title="alpha", text="alpha content"),
            Document(id="b", title="beta", text="beta content")]
    res = run(profile_mod.build_profile(assemble(docs), CFG))
    names = {s["name"] for s in res.profile["skills"]}
    assert names == {"K8s", "vLLM"}, "no branch may be lost when reduce fails"


def test_profile_reports_failed_units(monkeypatch):
    async def half_fail(client, cfg, system, user, max_tokens=2048):
        if "beta" in user:
            raise CallError("simulated 429")
        return {"skills": [{"name": "Docker", "strength": "used", "snippet": "s"}],
                "domains": [], "projects": [], "experience": [],
                "interests": [], "achievements": [], "constraints": []}

    monkeypatch.setattr(profile_mod, "json_call", half_fail)
    docs = [Document(id="a", title="alpha", text="alpha content"),
            Document(id="b", title="beta", text="beta content")]
    res = run(profile_mod.build_profile(assemble(docs), CFG))
    assert res.n_failed == 1
    assert "Partial profile" in res.warning
    assert not res.is_empty


def test_profile_on_empty_corpus_does_not_call_the_model(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("must not call the model with no units")

    monkeypatch.setattr(profile_mod, "json_call", boom)
    res = run(profile_mod.build_profile(assemble([]), CFG))
    assert res.is_empty and "empty" in res.warning
