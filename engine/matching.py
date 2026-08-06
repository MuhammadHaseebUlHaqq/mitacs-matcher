"""
Stage 2 — the comparative judge.

The pipeline is two stages, not three: a free lexical filter picks the plausible
set, and one model pass scores that set on a calibrated scale.

    PREFILTER  BM25 over the whole corpus (engine/prefilter.py). No model calls.
    JUDGE      read each survivor in full and score it 0-100 on a scale that
               means the same thing for every project, so the ranking the user
               sees is comparable by construction.

WHY THERE IS NO SCREENING PASS
------------------------------
An earlier design put a model "map" pass between the two: batches of five
projects, each asked only whether the person was a plausible fit, feeding a
judge pass over the survivors. Measured against the real corpus that stage was
mostly waste. BM25 scores decay gradually rather than falling off a cliff — at
width 400 the tail scored ~31% of the top hit and was visibly off-topic — so
the map pass spent ~75 of ~101 calls rejecting projects the lexical filter had
already ranked near the bottom.

Narrowing the filter instead (3,359 -> ~60) and judging that set directly costs
~10 calls. Quality does not drop, because the judge reads the full project text
where the map pass only skimmed a trimmed description to make a keep/drop call:
the plausible set is now read more carefully, not less.

The cost of this is honest and worth stating: BM25 cannot match meaning across
different vocabulary, and the judge only ever sees what the filter admits. A
project describing the same work in words the profile never uses is missed
before any model reads it. The map pass, being semantic, could catch that case.
Widening the filter is the dial that trades calls for that recall.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .llm import CallError, LLMConfig, MAX_CONCURRENCY, json_call

# Projects per judge call. The judge reasons comparatively and writes several
# fields per project, so batches stay small enough that each one is read
# properly rather than skimmed.
JUDGE_BATCH_SIZE = 6

JUDGE_SYSTEM = (
    "You make the FINAL, COMPARABLE call on how well a person fits research "
    "projects. You are given their CAPABILITY PROFILE and a batch of projects.\n"
    "The projects reached you through a keyword filter, which is a crude "
    "instrument: some of them will be poor fits. Say so with a low score rather "
    "than inflating one — a batch where most projects score below 40 is a "
    "correct outcome, not a failure.\n"
    "Match on MEANING, not exact wording — use world knowledge:\n"
    "  - 'Helm charts on AKS, kubeadm, Calico CNI' fits a project asking for "
    "'cloud-native orchestration' or 'Kubernetes resource management'.\n"
    "  - 'benchmarked vLLM vs HuggingFace, KV-cache OOM analysis' fits "
    "'efficient LLM serving', 'inference optimization', or 'ML systems'.\n"
    "  - 'multi-agent RFP pipeline' fits 'agentic AI', 'LLM agents', "
    "'autonomous workflows'.\n"
    "  - A related-but-not-identical domain still fits if the underlying "
    "technique transfers; say so in `rationale`.\n"
    "Score on a CALIBRATED 0-100 scale that means the same thing for every "
    "project:\n"
    "  90-100 the person's own demonstrated work is directly on this topic\n"
    "  75-89  strongly adjacent — core techniques transfer with little ramp-up\n"
    "  60-74  solid general fit; relevant foundation but not the specific topic\n"
    "  40-59  plausible with real ramp-up; some genuine overlap\n"
    "  1-39   weak — only generic overlap\n"
    "For EACH project return:\n"
    "  - `score` on that scale.\n"
    "  - `rationale`: one or two sentences naming the SPECIFIC thing in the "
    "profile that earns the score. Be concrete; never generic praise.\n"
    "  - `evidence`: the profile facts that support it, quoted. Copy the "
    "specific skill, project or experience — never invent a capability the "
    "profile does not state.\n"
    "  - `gaps`: what the person would have to learn. Be honest — a project "
    "with real gaps still scores well if the foundation transfers.\n"
    "RULES: echo every project id exactly; return an entry for EVERY project in "
    "the batch.\n\n"
    "Return ONLY JSON: {\"verdicts\": [{\"projectId\": str, \"score\": int, "
    "\"rationale\": str, \"evidence\": [str], \"gaps\": [str]}]}"
)


@dataclass
class Match:
    project: dict
    score: int
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass
class MatchRun:
    matches: list[Match] = field(default_factory=list)
    n_screened: int = 0        # projects the filter admitted, i.e. what the model read
    n_judged: int = 0          # projects that came back with a comparable score
    judge_calls: int = 0
    n_failed_judge: int = 0
    dropped_ids: int = 0
    warning: str = ""


def _batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _render_project(p: dict) -> dict:
    """What the model sees. Every project is now read in full — there is no
    cheaper skimming pass to render for."""
    desc = " ".join(filter(None, [p.get("description", ""), p.get("researchArea", "")]))
    out = {
        "id": p["id"],
        "title": p.get("title", ""),
        "description": desc[:2200],
    }
    if skills := (p.get("studentSkills") or ""):
        out["requiredSkills"] = skills[:600]
    if roles := (p.get("studentRoles") or ""):
        out["studentRole"] = roles[:600]
    return out


async def _judge_call(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    cfg: LLMConfig, profile_text: str, batch: list[dict],
) -> tuple[dict[str, dict], str]:
    user = (
        f"CAPABILITY PROFILE:\n{profile_text}\n\n"
        f"PROJECTS TO JUDGE:\n"
        f"{json.dumps([_render_project(p) for p in batch], ensure_ascii=False)}"
    )
    try:
        async with sem:
            data = await json_call(client, cfg, JUDGE_SYSTEM, user, max_tokens=3000)
    except (CallError, httpx.HTTPError) as e:
        return {}, str(e)

    out: dict[str, dict] = {}
    for v in (data or {}).get("verdicts", []) if isinstance(data, dict) else []:
        if isinstance(v, dict) and v.get("projectId"):
            out[str(v["projectId"]).strip()] = v
    return out, ""


def _clamp(n: object) -> int:
    try:
        v = int(float(n))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, v))


async def run_matching(
    profile_text: str,
    projects: list[dict],
    cfg: LLMConfig,
    top_n: int,
    progress: Callable[[str], None] | None = None,
) -> MatchRun:
    """Judge `projects` against `profile_text` and return the best `top_n`.

    `projects` is expected to be the prefiltered set — every one of them is read
    by the model, so the caller controls cost by controlling that list.
    """

    def note(m: str) -> None:
        if progress:
            progress(m)

    run = MatchRun(n_screened=len(projects))
    if not projects:
        run.warning = "No projects to judge."
        return run
    if not profile_text.strip():
        run.warning = "The capability profile is empty — add documents to the knowledge base first."
        return run

    batches = _batch(projects, JUDGE_BATCH_SIZE)
    run.judge_calls = len(batches)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        # Progress is reported per call, not per stage: on a rate-limited free
        # tier a single "judging N projects" line with nothing after it is
        # indistinguishable from a hang.
        note(f"Reading and scoring {len(projects)} projects in {len(batches)} calls "
             f"({cfg.model}, {MAX_CONCURRENCY} at a time)…")
        done = {"n": 0, "failed": 0}

        async def judged(batch: list[dict]) -> tuple[dict[str, dict], str]:
            out = await _judge_call(client, sem, cfg, profile_text, batch)
            done["n"] += 1
            if out[1]:
                done["failed"] += 1
            note(f"  scored {done['n']}/{len(batches)} batches"
                 + (f" · {done['failed']} failed" if done["failed"] else ""))
            return out

        results = await asyncio.gather(*(judged(b) for b in batches))

    matches: list[Match] = []
    first_error = ""
    for (verdicts, err), batch in zip(results, batches):
        if err:
            run.n_failed_judge += 1
            first_error = first_error or err
            continue
        for p in batch:
            v = verdicts.get(p["id"])
            if not v:
                # The model skipped a project we showed it. Dropping it silently
                # would misreport the corpus as smaller than it was.
                run.dropped_ids += 1
                continue
            matches.append(Match(
                project=p,
                score=_clamp(v.get("score")),
                rationale=str(v.get("rationale") or ""),
                evidence=[str(x) for x in (v.get("evidence") or []) if x],
                gaps=[str(x) for x in (v.get("gaps") or []) if x],
            ))

    if run.n_failed_judge:
        pct = round(100 * run.n_failed_judge / max(len(batches), 1))
        run.warning = (
            f"Partial results — {run.n_failed_judge} of {len(batches)} scoring calls "
            f"({pct}%) failed, so some projects were never read."
            + (f" Reason: {first_error}" if first_error else "")
        )
        note(run.warning)

    if not matches and not run.warning:
        run.warning = "No project in the filtered set could be scored."

    run.n_judged = len(matches)
    matches.sort(key=lambda m: -m.score)
    run.matches = matches[:top_n]
    return run
