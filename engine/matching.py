"""
Stages 2 & 3 — the map-reduce matcher.

Ported from bideez `src/mastra/matching/run.ts`, with requirements/corpus
swapped for profile/projects:

  MAP    fan out over batches of projects with a bounded semaphore; each call
         reads the capability profile against ONE batch and returns batch-local
         candidate fits with extractive evidence. Batch-local only — no ranking.
  GATHER collect candidates, validating that every returned projectId resolves
         to a real project. Ids that do not resolve are dropped and counted,
         exactly as bideez drops unresolvable evidenceUnitIds.
  JUDGE  re-judge the strongest candidates head-to-head in batches. Map scores
         from different batches are NOT comparable (bideez: "chunk-local
         relevance scores are NOT comparable — re-judge from the snippets
         themselves"), so the only scores the user ever sees come from here.

Projects with no candidate anywhere skip the judge entirely, saving calls —
the same shortcut bideez takes for requirements with no evidence.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .llm import CallError, LLMConfig, MAX_CONCURRENCY, json_call

# Projects per map call. Small enough that each project's description is read
# properly rather than skimmed; large enough to keep the call count sane.
MAP_BATCH_SIZE = 5
# Projects per judge call. Smaller — the judge reasons comparatively and writes
# longer output per project.
JUDGE_BATCH_SIZE = 6
# How many map survivors to judge, as a multiple of the requested result count.
# Over-admitting here costs calls but protects against a batch-local score
# wrongly cutting a strong project before the comparable pass.
JUDGE_POOL_MULTIPLIER = 2.5
# Floor on the judge pool. Without it a small `top_n` makes the pool so narrow
# that ties in the (deliberately non-comparable) map score decide the ranking —
# the exact failure the judge stage exists to prevent. Costs a few extra calls
# on small requests and nothing on large ones.
MIN_JUDGE_POOL = 24

MAP_SYSTEM = (
    "You match a person against research projects. You are given that person's "
    "CAPABILITY PROFILE and a BATCH of candidate projects.\n"
    "For EACH project in the batch, decide whether this person is a plausible "
    "fit, and return a candidate entry ONLY if they are.\n"
    "Match on MEANING, not exact wording — use world knowledge:\n"
    "  - 'Helm charts on AKS, kubeadm, Calico CNI' fits a project asking for "
    "'cloud-native orchestration' or 'Kubernetes resource management'.\n"
    "  - 'benchmarked vLLM vs HuggingFace, KV-cache OOM analysis' fits "
    "'efficient LLM serving', 'inference optimization', or 'ML systems'.\n"
    "  - 'multi-agent RFP pipeline' fits 'agentic AI', 'LLM agents', "
    "'autonomous workflows'.\n"
    "  - A related-but-not-identical domain still fits if the underlying "
    "technique transfers; say so in `reason`.\n"
    "RULES:\n"
    "  - Echo each project's id EXACTLY as given.\n"
    "  - `evidence` must quote the profile — copy the specific skill, project or "
    "experience that supports the fit. Never invent a capability the profile "
    "does not state.\n"
    "  - `fit` is a rough 0-100 signal within THIS batch only. Do not try to "
    "rank against projects you cannot see.\n"
    "  - If the person is NOT a plausible fit for a project, omit it entirely. "
    "Do not force a match.\n\n"
    "Return ONLY JSON: {\"candidates\": [{\"projectId\": str, \"fit\": int, "
    "\"reason\": str, \"evidence\": [str], \"concerns\": [str]}]}"
)

JUDGE_SYSTEM = (
    "You make the FINAL, COMPARABLE call on how well a person fits research "
    "projects. You are given their CAPABILITY PROFILE and a batch of projects, "
    "each with the candidate evidence gathered for it earlier.\n"
    "The earlier scores were produced in isolation and are NOT comparable — "
    "re-judge each project from the profile and the project text themselves.\n"
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
    "  - `evidence`: the profile facts that support it, quoted.\n"
    "  - `gaps`: what the person would have to learn. Be honest — a project "
    "with real gaps still scores well if the foundation transfers.\n"
    "RULES: echo every project id exactly; return an entry for EVERY project in "
    "the batch; never invent capabilities the profile does not state.\n\n"
    "Return ONLY JSON: {\"verdicts\": [{\"projectId\": str, \"score\": int, "
    "\"rationale\": str, \"evidence\": [str], \"gaps\": [str]}]}"
)


@dataclass
class Candidate:
    project_id: str
    fit: int
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


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
    n_screened: int = 0        # projects that reached the model
    n_candidates: int = 0      # projects the map stage judged plausible
    n_judged: int = 0          # projects that got a comparable score
    map_calls: int = 0
    judge_calls: int = 0
    n_failed_map: int = 0
    n_failed_judge: int = 0
    dropped_ids: int = 0
    warning: str = ""


def _batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _render_project(p: dict, full: bool) -> dict:
    """What the model sees. The map pass gets a trimmed description (it only
    needs enough to decide plausibility); the judge gets more room."""
    limit = 2200 if full else 1100
    desc = " ".join(filter(None, [p.get("description", ""), p.get("researchArea", "")]))
    out = {
        "id": p["id"],
        "title": p.get("title", ""),
        "description": desc[:limit],
    }
    if full:
        skills = p.get("studentSkills") or ""
        roles = p.get("studentRoles") or ""
        if skills:
            out["requiredSkills"] = skills[:600]
        if roles:
            out["studentRole"] = roles[:600]
    return out


async def _map_call(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    cfg: LLMConfig, profile_text: str, batch: list[dict],
) -> tuple[list[Candidate], str]:
    user = (
        f"CAPABILITY PROFILE:\n{profile_text}\n\n"
        f"CANDIDATE PROJECTS:\n{json.dumps([_render_project(p, False) for p in batch], ensure_ascii=False)}"
    )
    try:
        async with sem:
            data = await json_call(client, cfg, MAP_SYSTEM, user, max_tokens=1800)
    except (CallError, httpx.HTTPError) as e:
        return [], str(e)

    out: list[Candidate] = []
    for c in (data or {}).get("candidates", []) if isinstance(data, dict) else []:
        if not isinstance(c, dict):
            continue
        pid = str(c.get("projectId") or "").strip()
        if not pid:
            continue
        out.append(
            Candidate(
                project_id=pid,
                fit=_clamp(c.get("fit")),
                reason=str(c.get("reason") or ""),
                evidence=[str(x) for x in (c.get("evidence") or []) if x],
                concerns=[str(x) for x in (c.get("concerns") or []) if x],
            )
        )
    return out, ""


async def _judge_call(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    cfg: LLMConfig, profile_text: str,
    batch: list[tuple[dict, Candidate]],
) -> tuple[dict[str, dict], str]:
    payload = [
        {
            "project": _render_project(p, True),
            "gatheredEvidence": cand.evidence,
            "priorReason": cand.reason,
        }
        for p, cand in batch
    ]
    user = (
        f"CAPABILITY PROFILE:\n{profile_text}\n\n"
        f"PROJECTS TO JUDGE:\n{json.dumps(payload, ensure_ascii=False)}"
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
    """Screen `projects` against `profile_text` and return the best `top_n`."""

    def note(m: str) -> None:
        if progress:
            progress(m)

    run = MatchRun(n_screened=len(projects))
    if not projects:
        run.warning = "No projects to screen."
        return run
    if not profile_text.strip():
        run.warning = "The capability profile is empty — add documents to the knowledge base first."
        return run

    by_id = {p["id"]: p for p in projects}
    map_batches = _batch(projects, MAP_BATCH_SIZE)
    run.map_calls = len(map_batches)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        # ---- MAP ---------------------------------------------------------- #
        # Progress is reported per call, not per stage. On a rate-limited free
        # tier this phase can run for minutes, and a single "screening N
        # projects" line with nothing after it is indistinguishable from a hang.
        note(f"Screening {len(projects)} projects in {len(map_batches)} calls "
             f"({cfg.model}, {MAX_CONCURRENCY} at a time)…")
        done = {"n": 0, "hits": 0, "failed": 0}

        async def mapped(batch: list[dict]) -> tuple[list[Candidate], str]:
            out = await _map_call(client, sem, cfg, profile_text, batch)
            done["n"] += 1
            if out[1]:
                done["failed"] += 1
            else:
                done["hits"] += len(out[0])
            if done["n"] % 2 == 0 or done["n"] == len(map_batches):
                note(f"  screened {done['n']}/{len(map_batches)} batches · "
                     f"{done['hits']} candidates · {done['failed']} failed")
            return out

        results = await asyncio.gather(*(mapped(b) for b in map_batches))

        # ---- GATHER ------------------------------------------------------- #
        best: dict[str, Candidate] = {}
        first_error = ""
        for cands, err in results:
            if err:
                run.n_failed_map += 1
                first_error = first_error or err
                continue
            for c in cands:
                if c.project_id not in by_id:
                    # Model cited an id we never showed it — cannot resolve, drop it.
                    run.dropped_ids += 1
                    continue
                prev = best.get(c.project_id)
                if prev is None or c.fit > prev.fit:
                    best[c.project_id] = c

        run.n_candidates = len(best)
        note(f"{len(best)} plausible candidates found ({run.n_failed_map} map calls failed).")

        if run.n_failed_map:
            pct = round(100 * run.n_failed_map / max(len(map_batches), 1))
            run.warning = (
                f"Partial results — {run.n_failed_map} of {len(map_batches)} screening calls "
                f"({pct}%) failed, so some projects were never read."
                + (f" Reason: {first_error}" if first_error else "")
            )
            note(run.warning)

        if not best:
            if not run.warning:
                run.warning = "No project in the screened set matched this profile."
            return run

        # ---- JUDGE -------------------------------------------------------- #
        # Map `fit` is batch-local, so it is used only to choose WHO gets judged,
        # never to rank the output.
        pool_size = min(len(best), max(int(top_n * JUDGE_POOL_MULTIPLIER), MIN_JUDGE_POOL))
        pool_ids = sorted(best, key=lambda pid: -best[pid].fit)[:pool_size]
        pool = [(by_id[pid], best[pid]) for pid in pool_ids]

        judge_batches = _batch(pool, JUDGE_BATCH_SIZE)
        run.judge_calls = len(judge_batches)
        note(f"Judging the top {len(pool)} head-to-head in {len(judge_batches)} calls…")

        jdone = {"n": 0, "failed": 0}

        async def judged_batch(batch):
            out = await _judge_call(client, sem, cfg, profile_text, batch)
            jdone["n"] += 1
            if out[1]:
                jdone["failed"] += 1
            note(f"  judged {jdone['n']}/{len(judge_batches)} batches"
                 + (f" · {jdone['failed']} failed" if jdone["failed"] else ""))
            return out

        judged = await asyncio.gather(*(judged_batch(b) for b in judge_batches))

        matches: list[Match] = []
        judge_error = ""
        for (verdicts, err), batch in zip(judged, judge_batches):
            if err:
                run.n_failed_judge += 1
                judge_error = judge_error or err
                continue
            for p, cand in batch:
                v = verdicts.get(p["id"])
                if not v:
                    # Judge omitted this project — keep it, flagged, rather than
                    # dropping it silently.
                    matches.append(Match(
                        project=p, score=_clamp(cand.fit),
                        rationale="(not re-judged — screening score shown)",
                        evidence=cand.evidence, gaps=cand.concerns,
                    ))
                    continue
                matches.append(Match(
                    project=p,
                    score=_clamp(v.get("score")),
                    rationale=str(v.get("rationale") or cand.reason),
                    evidence=[str(x) for x in (v.get("evidence") or cand.evidence) if x],
                    gaps=[str(x) for x in (v.get("gaps") or []) if x],
                ))

        if run.n_failed_judge:
            extra = (
                f"{run.n_failed_judge} of {len(judge_batches)} judging calls failed; "
                f"those projects are excluded from the ranking."
                + (f" Reason: {judge_error}" if judge_error else "")
            )
            run.warning = f"{run.warning} {extra}".strip()
            note(extra)

        run.n_judged = len(matches)
        matches.sort(key=lambda m: -m.score)
        run.matches = matches[:top_n]

    return run
