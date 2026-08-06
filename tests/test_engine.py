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


def fake_json_call(judge_reply, fail_judge_on=()):
    """Build a `json_call` stand-in. The pipeline is one pass now, so there is
    only one kind of call to answer."""
    calls = {"judge": 0}

    async def _call(client, cfg, system, user, max_tokens=2048):
        i = calls["judge"]; calls["judge"] += 1
        if i in fail_judge_on:
            raise CallError("simulated upstream rate-limit")
        return judge_reply(user)

    return _call, calls


def ids_in(user):
    """Which projects the model was actually shown in this call."""
    return [p["id"] for p in MATCH_PROJECTS if f'"{p["id"]}"' in user]


def run(coro):
    return asyncio.run(coro)


def test_the_judge_score_is_what_ranks(monkeypatch):
    """The only score the user ever sees comes from the judge, and it is what
    orders the results."""
    def judge_reply(user):
        return {"verdicts": [{"projectId": pid, "score": 95 if pid == "p1" else 20,
                              "rationale": "judged", "evidence": ["ev"], "gaps": []}
                             for pid in ids_in(user)]}

    call, _ = fake_json_call(judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.matches[0].project["id"] == "p1"
    assert res.matches[0].score == 95
    assert res.matches[0].rationale == "judged"


def test_every_project_sent_is_read_in_one_pass(monkeypatch):
    """No screening stage means the call count follows directly from the number
    of projects handed in — that is the whole cost model."""
    def judge_reply(user):
        return {"verdicts": [{"projectId": pid, "score": 50, "rationale": "j",
                              "evidence": [], "gaps": []} for pid in ids_in(user)]}

    call, calls = fake_json_call(judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    expected = -(-len(MATCH_PROJECTS) // matching.JUDGE_BATCH_SIZE)
    assert calls["judge"] == expected
    assert res.judge_calls == expected
    assert res.n_judged == len(MATCH_PROJECTS)


def test_ids_the_model_invents_never_become_matches(monkeypatch):
    """A verdict citing a project we never sent must not resolve to anything."""
    def judge_reply(user):
        return {"verdicts": [{"projectId": "GHOST", "score": 90, "rationale": "r",
                              "evidence": [], "gaps": []}]}

    call, _ = fake_json_call(judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.matches == []
    assert res.dropped_ids == len(MATCH_PROJECTS)


def test_failed_calls_produce_a_partial_warning(monkeypatch):
    """A rate limit must be reported, not silently shrink the corpus."""
    def judge_reply(user):
        return {"verdicts": [{"projectId": pid, "score": 70, "rationale": "j",
                              "evidence": [], "gaps": []} for pid in ids_in(user)]}

    # One project per call, so failing call 0 loses exactly one project.
    monkeypatch.setattr(matching, "JUDGE_BATCH_SIZE", 1)
    call, _ = fake_json_call(judge_reply, fail_judge_on=(0,))
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.n_failed_judge == 1
    assert "Partial results" in res.warning
    assert "never read" in res.warning
    assert len(res.matches) == len(MATCH_PROJECTS) - 1


def test_project_omitted_by_the_judge_is_counted_not_invented(monkeypatch):
    """With no earlier pass there is no fallback score to show. Inventing one
    would be worse than admitting the project was skipped, so it is counted."""
    def judge_reply(user):
        return {"verdicts": []}  # the model silently drops everything

    call, _ = fake_json_call(judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=3))
    assert res.matches == []
    assert res.dropped_ids == len(MATCH_PROJECTS)
    assert "could be scored" in res.warning


def test_top_n_is_respected(monkeypatch):
    def judge_reply(user):
        return {"verdicts": [{"projectId": pid, "score": 50 + int(pid[1]), "rationale": "j",
                              "evidence": [], "gaps": []} for pid in ids_in(user)]}

    call, _ = fake_json_call(judge_reply)
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


def test_low_scores_are_reported_not_suppressed(monkeypatch):
    """The filter is lexical, so weak projects reach the judge by design. It
    must score them honestly rather than omit them — the user decides what a 20
    is worth."""
    def judge_reply(user):
        return {"verdicts": [{"projectId": pid, "score": 12, "rationale": "weak overlap",
                              "evidence": [], "gaps": ["everything"]} for pid in ids_in(user)]}

    call, _ = fake_json_call(judge_reply)
    monkeypatch.setattr(matching, "json_call", call)

    res = run(matching.run_matching("profile", MATCH_PROJECTS, CFG, top_n=5))
    assert len(res.matches) == len(MATCH_PROJECTS)
    assert all(m.score == 12 for m in res.matches)
    assert not res.warning


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


# --------------------------------------------------------------------------- #
# resumable profile builds
# --------------------------------------------------------------------------- #
def test_a_rate_limited_build_banks_what_succeeded(monkeypatch):
    """The failure that made this necessary: on a free tier some passages die on
    quota. Whatever was read must be kept so the next attempt resumes."""
    async def half_fail(client, cfg, system, user, max_tokens=2048):
        if "beta" in user:
            raise CallError("quota exceeded")
        return {"skills": [{"name": "K8s", "strength": "demonstrated", "snippet": "s"}],
                "domains": [], "projects": [], "experience": [],
                "interests": [], "achievements": [], "constraints": []}

    monkeypatch.setattr(profile_mod, "json_call", half_fail)
    docs = [Document(id="a", title="alpha", text="alpha content"),
            Document(id="b", title="beta", text="beta content")]
    res = run(profile_mod.build_profile(assemble(docs), CFG))
    assert res.n_failed == 1
    assert len(res.cache) == 1, "the passage that succeeded must be banked"
    assert "retry just those" in res.warning


def test_retry_only_re_reads_the_failed_passages(monkeypatch):
    calls = {"n": 0}

    async def count(client, cfg, system, user, max_tokens=2048):
        if system.startswith("You are a profile extractor"):
            calls["n"] += 1
        return {"skills": [{"name": "X", "strength": "used", "snippet": "s"}],
                "domains": [], "projects": [], "experience": [],
                "interests": [], "achievements": [], "constraints": []}

    monkeypatch.setattr(profile_mod, "json_call", count)
    docs = [Document(id="a", title="alpha", text="alpha content"),
            Document(id="b", title="beta", text="beta content")]
    corpus = assemble(docs)

    first = run(profile_mod.build_profile(corpus, CFG))
    assert calls["n"] == 2 and len(first.cache) == 2

    calls["n"] = 0
    second = run(profile_mod.build_profile(corpus, CFG, cache=first.cache))
    assert calls["n"] == 0, "a fully cached rebuild must not call the model again"
    assert second.n_cached == 2
    assert not second.is_empty


def test_total_failure_says_the_build_failed_not_that_it_is_thin(monkeypatch):
    """`may be missing information` reads like a warning and hides that there is
    no profile at all — which is why Find matches stayed disabled with no
    apparent reason."""
    async def all_fail(client, cfg, system, user, max_tokens=2048):
        raise CallError("quota exceeded")

    monkeypatch.setattr(profile_mod, "json_call", all_fail)
    docs = [Document(id="a", title="alpha", text="alpha content")]
    res = run(profile_mod.build_profile(assemble(docs), CFG))
    assert res.is_empty
    assert "Profile build failed" in res.warning
    assert "resume" in res.warning
    assert "may be missing information" not in res.warning


def test_cache_key_is_content_addressed_so_reupload_reuses_it(monkeypatch):
    """Document ids are regenerated on every upload; the cache must key on text
    so re-adding the same file does not re-spend quota."""
    calls = {"n": 0}

    async def count(client, cfg, system, user, max_tokens=2048):
        if system.startswith("You are a profile extractor"):
            calls["n"] += 1
        return {"skills": [{"name": "X", "strength": "used", "snippet": "s"}],
                "domains": [], "projects": [], "experience": [],
                "interests": [], "achievements": [], "constraints": []}

    monkeypatch.setattr(profile_mod, "json_call", count)
    first = run(profile_mod.build_profile(
        assemble([Document(id="id-one", title="cv.pdf", text="same body")]), CFG))
    calls["n"] = 0
    # same content, brand-new document id — as happens on re-upload
    second = run(profile_mod.build_profile(
        assemble([Document(id="id-two", title="cv.pdf", text="same body")]),
        CFG, cache=first.cache))
    assert calls["n"] == 0 and second.n_cached == 1
