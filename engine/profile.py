"""
Step 1 — documents → capability profile (map-reduce).

Every passage of the knowledge base is read by the model and asked the same
question: *what does this passage prove this person can do?* The per-passage
extracts are then folded into one profile by a real consolidation call.

That fold is deliberately not string concatenation. Overlapping text and
repeated boilerplate produce duplicate entries, and passages read independently
classify the same skill inconsistently — a merge pass resolves both.

Cost is proportional to the number of passages, not to the corpus size: the
profile is built once and cached, then reused for every match run.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Callable

import httpx

from .corpus import Corpus, EvidenceUnit
from .llm import CallError, LLMConfig, MAX_CONCURRENCY, json_call


def unit_key(unit: EvidenceUnit) -> str:
    """Content-addressed id for a passage.

    Keyed on the text rather than the unit id so the extract cache survives
    re-uploading the same document (ids are regenerated per upload), while
    editing a document invalidates only the passages that actually changed.
    """
    return hashlib.sha1(unit.text.encode("utf-8")).hexdigest()[:16]

# How many per-unit extracts one reduce call folds before another layer is needed.
REDUCE_FANIN = 6

EXTRACT_SYSTEM = (
    "You are a profile extractor. You are given ONE PASSAGE from a person's own "
    "documents (CV, portfolio, project write-ups, transcripts, repo READMEs).\n"
    "Extract what this passage PROVES about the person, for the purpose of "
    "matching them to research projects.\n"
    "Judge GENEROUSLY and use world knowledge — record the underlying capability, "
    "not just the literal words:\n"
    "  - 'kubeadm, Calico CNI, Helm charts on AKS' also proves container "
    "orchestration and cloud-native infrastructure.\n"
    "  - 'benchmarked vLLM against HuggingFace, KV-cache OOM sweep' also proves "
    "LLM inference optimization and ML systems benchmarking.\n"
    "  - 'VAPI voice agent with Deepgram STT' also proves conversational/voice AI.\n"
    "Copy concrete evidence VERBATIM into `snippet` — never invent detail that is "
    "not in the passage. Do not inflate: a tool merely listed is a weaker claim "
    "than one described in use, and you should say so in `strength`.\n\n"
    "Return ONLY a JSON object with this exact shape:\n"
    "{\n"
    '  "skills":    [{"name": str, "strength": "demonstrated"|"used"|"mentioned", "snippet": str}],\n'
    '  "domains":   [str],\n'
    '  "projects":  [{"name": str, "what": str, "snippet": str}],\n'
    '  "experience":[{"role": str, "org": str, "what": str}],\n'
    '  "interests": [str],\n'
    '  "achievements": [str],\n'
    '  "constraints": [str]\n'
    "}\n"
    "Use [] for any field the passage says nothing about. If the passage proves "
    "nothing about the person at all, return all fields empty."
)

REDUCE_SYSTEM = (
    "You are a profile consolidator. You are given several PARTIAL PROFILES "
    "extracted independently from different passages of one person's documents.\n"
    "Merge them into a SINGLE profile. This is a real consolidation pass, not a "
    "concatenation:\n"
    "  - De-duplicate aggressively. The same skill phrased differently "
    "('K8s', 'Kubernetes', 'kubeadm cluster') is ONE skill — keep the clearest name.\n"
    "  - Resolve conflicting `strength` values by keeping the STRONGEST supported "
    "one, and keep the snippet that best evidences it.\n"
    "  - Merge duplicate projects and roles; keep the fullest description.\n"
    "  - Preserve concrete evidence verbatim. Never invent anything that is not "
    "in the partial profiles.\n"
    "Return ONLY a JSON object with the SAME shape as the inputs."
)

_EMPTY: dict[str, list] = {
    "skills": [], "domains": [], "projects": [],
    "experience": [], "interests": [], "achievements": [], "constraints": [],
}


@dataclass
class UnitExtract:
    unit: EvidenceUnit
    data: dict | None = None
    failed: bool = False
    error: str = ""


@dataclass
class ProfileResult:
    profile: dict = field(default_factory=lambda: dict(_EMPTY))
    n_units: int = 0
    n_hits: int = 0
    n_failed: int = 0
    n_cached: int = 0
    reduce_layers: int = 0
    warning: str = ""
    extracts: list[UnitExtract] = field(default_factory=list)
    # Successful per-passage extracts, keyed by `unit_key`. Persisted by the
    # caller so a rate-limited build can be resumed instead of restarted.
    cache: dict[str, dict] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(self.profile.get(k) for k in _EMPTY)


def _normalize(obj: object) -> dict:
    """Coerce whatever the model returned into the profile shape."""
    out = {k: [] for k in _EMPTY}
    if not isinstance(obj, dict):
        return out
    for key in _EMPTY:
        val = obj.get(key)
        if isinstance(val, list):
            out[key] = [v for v in val if v not in (None, "", {}, [])]
        elif val:
            out[key] = [val]
    return out


def _merge_local(profiles: list[dict]) -> dict:
    """Deterministic union used as the fallback when a reduce call fails —
    lossy on duplicates, but never drops evidence."""
    out = {k: [] for k in _EMPTY}
    seen: dict[str, set[str]] = {k: set() for k in _EMPTY}
    for p in profiles:
        for key in _EMPTY:
            for item in p.get(key, []):
                fingerprint = (
                    item.get("name") or item.get("role") or str(item)
                    if isinstance(item, dict) else str(item)
                ).strip().lower()
                if fingerprint and fingerprint not in seen[key]:
                    seen[key].add(fingerprint)
                    out[key].append(item)
    return out


async def _extract_unit(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    cfg: LLMConfig, unit: EvidenceUnit,
) -> UnitExtract:
    user = (
        f"PASSAGE (source: {unit.source_anchor or unit.source_id}"
        f"{f', section: {unit.breadcrumb}' if unit.breadcrumb else ''}):\n\n{unit.text}"
    )
    try:
        async with sem:
            data = await json_call(client, cfg, EXTRACT_SYSTEM, user, max_tokens=1600)
    except (CallError, httpx.HTTPError) as e:
        return UnitExtract(unit=unit, failed=True, error=str(e))
    return UnitExtract(unit=unit, data=_normalize(data))


async def _reduce(
    client: httpx.AsyncClient, sem: asyncio.Semaphore,
    cfg: LLMConfig, profiles: list[dict],
) -> tuple[dict, int]:
    """Fold partial profiles by a tree of consolidation calls until one remains."""
    import json as _json

    pieces = [p for p in profiles if any(p.get(k) for k in _EMPTY)]
    layers = 0
    if not pieces:
        return dict(_EMPTY), 0

    while len(pieces) > 1:
        layers += 1
        groups = [pieces[i : i + REDUCE_FANIN] for i in range(0, len(pieces), REDUCE_FANIN)]

        async def fold(group: list[dict]) -> dict:
            body = "\n\n".join(
                f"[PARTIAL PROFILE {j + 1}]\n{_json.dumps(g, ensure_ascii=False)}"
                for j, g in enumerate(group)
            )
            try:
                async with sem:
                    merged = await json_call(
                        client, cfg, REDUCE_SYSTEM,
                        f"PARTIAL PROFILES:\n{body}", max_tokens=3000,
                    )
                return _normalize(merged)
            except (CallError, httpx.HTTPError):
                # Honest degradation: fall back to a deterministic union rather
                # than losing this branch of the tree entirely.
                return _merge_local(group)

        pieces = list(await asyncio.gather(*(fold(g) for g in groups)))

    return pieces[0], layers


async def build_profile(
    corpus: Corpus,
    cfg: LLMConfig,
    progress: Callable[[str], None] | None = None,
    cache: dict[str, dict] | None = None,
) -> ProfileResult:
    """Read the knowledge base into a capability profile.

    `cache` carries per-passage extracts from earlier attempts. Passages already
    in it are not re-read, so a build interrupted by a rate limit resumes where
    it stopped rather than re-spending quota on work that already succeeded —
    which is the difference between finishing and never finishing on a free tier.
    """

    def note(m: str) -> None:
        if progress:
            progress(m)

    units = corpus.units
    result = ProfileResult(n_units=len(units), cache=dict(cache or {}))
    if not units:
        result.warning = "The knowledge base is empty — add documents first."
        return result

    todo = [u for u in units if unit_key(u) not in result.cache]
    result.n_cached = len(units) - len(todo)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        if todo:
            if result.n_cached:
                note(f"{result.n_cached} passage(s) already read; resuming with {len(todo)}.")
            note(f"Reading {len(todo)} knowledge-base passage(s) in parallel ({cfg.model})…")
            extracts = list(await asyncio.gather(
                *(_extract_unit(client, sem, cfg, u) for u in todo)
            ))
            result.extracts = extracts
            result.n_failed = sum(1 for e in extracts if e.failed)
            for e in extracts:
                if e.data and any(e.data.get(k) for k in _EMPTY):
                    result.cache[unit_key(e.unit)] = e.data
        else:
            note(f"All {len(units)} passage(s) already read — reusing cached extracts.")

        good = [result.cache[unit_key(u)] for u in units if unit_key(u) in result.cache]
        result.n_hits = len(good)
        note(f"{len(good)}/{len(units)} passages usable ({result.n_failed} failed this attempt).")

        if result.n_failed:
            reason = next((e.error for e in result.extracts if e.failed and e.error), "")
            if not good:
                # Nothing at all came back: this is a failed build, not a thin
                # one. Say so plainly — "may be missing information" reads like a
                # warning and hides that there is no profile to match against.
                result.warning = (
                    f"Profile build failed — none of the {len(units)} passage(s) could be read, "
                    f"so there is no profile yet. Nothing was lost: press Build profile again and "
                    f"it will resume."
                    + (f" Reason: {reason}" if reason else "")
                )
            else:
                pct = round(100 * result.n_failed / len(units))
                result.warning = (
                    f"Partial profile — {result.n_failed} of {len(units)} passages ({pct}%) could "
                    f"not be read this time, so the profile may be missing information. Press "
                    f"Build profile again to retry just those."
                    + (f" Reason: {reason}" if reason else "")
                )
            note(result.warning)

        if not good:
            if not result.warning:
                result.warning = "No profile information could be extracted from these documents."
            return result

        note(f"Consolidating {len(good)} partial profiles…")
        merged, layers = await _reduce(client, sem, cfg, good)
        result.profile = merged
        result.reduce_layers = layers

    return result


def render_profile(profile: dict, max_chars: int = 6000) -> str:
    """Flatten the profile into the compact text block shown to the matcher.

    Kept small on purpose: it is resent with every map call, so its size
    multiplies across the whole fan-out.
    """
    lines: list[str] = []

    def add(label: str, items: list, fmt) -> None:
        if not items:
            return
        rendered = [fmt(i) for i in items]
        rendered = [r for r in rendered if r and r.strip()]
        if rendered:
            lines.append(f"{label}: " + "; ".join(rendered))

    add("SKILLS", profile.get("skills", []),
        lambda s: (f"{s.get('name')} ({s.get('strength')})" if isinstance(s, dict) else str(s)))
    add("DOMAINS", profile.get("domains", []), str)
    add("RESEARCH INTERESTS", profile.get("interests", []), str)
    add("PROJECTS", profile.get("projects", []),
        lambda p: (f"{p.get('name')} — {p.get('what')}" if isinstance(p, dict) else str(p)))
    add("EXPERIENCE", profile.get("experience", []),
        lambda e: (f"{e.get('role')} at {e.get('org')} — {e.get('what')}" if isinstance(e, dict) else str(e)))
    add("ACHIEVEMENTS", profile.get("achievements", []), str)
    add("CONSTRAINTS", profile.get("constraints", []), str)

    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + " …[truncated]"
