"""
LLM transport — OpenRouter and Google Gemini.

Retry policy, error surfacing and the bring-your-own-key posture are ported from
Galt RAG (`grag.py`): the key arrives per request, is used, and is never stored
or logged. A call that gives up carries the provider's own reason so the
pipeline can report WHY rather than silently degrading.

Two providers behind one interface. Everything upstream (profile, matching)
takes an `LLMConfig` and never learns which provider it is talking to.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

PROVIDERS = ("openrouter", "gemini")

DEFAULT_MODELS = {
    "openrouter": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
    # Lite by default: measured live, the full flash alias resolves to
    # gemini-3.6-flash whose free tier is ~20 req/min and which spends ~1.9k
    # thinking tokens per call. The lite alias showed zero thinking overhead
    # and a far more usable free limit, which is what a fan-out pipeline needs.
    "gemini": os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
}

# Cap on concurrent API calls. BYOK, so the real ceiling is the user's own
# account limits; keep the default modest to stay friendly to free tiers.
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))

# Fallback Gemini catalogue, used when we cannot list models live (no key yet).
# Prefer the rolling `-latest` aliases: pinned versions retire (as of Aug 2026
# `gemini-2.5-flash` is already closed to new users) and a stale hardcoded list
# hands people a model id that no longer works.
GEMINI_FALLBACK = [
    {"id": "gemini-flash-lite-latest", "name": "Gemini Flash Lite (latest) — best free tier", "free": True},
    {"id": "gemini-flash-latest", "name": "Gemini Flash (latest)", "free": True},
    {"id": "gemini-pro-latest", "name": "Gemini Pro (latest)", "free": False},
    {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "free": False},
]


class CallError(Exception):
    """A call gave up. Carries a human-readable reason (upstream rate-limit,
    bad model, no credits, ...) so failures can be reported, not swallowed."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str

    def validated(self) -> "LLMConfig":
        if self.provider not in PROVIDERS:
            raise CallError(f"unknown provider '{self.provider}'")
        if not self.api_key.strip():
            raise CallError(f"missing API key for {self.provider}")
        return self


# --------------------------------------------------------------------------- #
# Error extraction
# --------------------------------------------------------------------------- #
def _error_message(r: httpx.Response, provider: str = "") -> str:
    """Pull the most useful message out of a provider error response.

    The provider and status code are prepended deliberately. A bare upstream
    string like "Your project has been denied access" reads as if this app
    refused the request, and gives no clue which of the two providers — or which
    account — to go and fix.
    """
    label = {"openrouter": "OpenRouter", "gemini": "Gemini"}.get(provider, provider or "Provider")
    where = f"{label} refused (HTTP {r.status_code})"
    try:
        body = r.json()
    except ValueError:
        return where

    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        meta = err.get("metadata") or {}
        msg = meta.get("raw") or err.get("message")
        if msg:
            return f"{where}: {str(msg).strip()}"
    return where


# Longest we will wait on a single backoff. Galt RAG capped this at 5s to keep a
# rate-limited free model from feeling hung. That is too aggressive here: Gemini
# free tiers are per-MINUTE (gemini-3.6-flash allows 5 req/min) and reply
# "retry in 45s", so a 5s ceiling guarantees every retry is wasted and the whole
# fan-out fails. Wait as long as the provider asks, up to this bound.
MAX_RETRY_WAIT = float(os.getenv("MAX_RETRY_WAIT", "50"))

_RETRY_SECONDS = re.compile(r"(\d+(?:\.\d+)?)\s*s")


def _retry_after_seconds(r: httpx.Response) -> float:
    """How long the provider wants us to wait, in seconds.

    OpenRouter uses the standard `Retry-After` header. Gemini does not — it puts
    the delay in the error body as a `RetryInfo` detail (`retryDelay: "45.3s"`)
    and repeats it in the message text, so fall back to both.
    """
    header = r.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    try:
        body = r.json()
    except ValueError:
        return 0.0
    if not isinstance(body, dict):
        return 0.0

    err = body.get("error") or {}
    for detail in err.get("details") or []:
        if isinstance(detail, dict) and "retryDelay" in detail:
            match = _RETRY_SECONDS.search(str(detail["retryDelay"]))
            if match:
                return float(match.group(1))

    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)", str(err.get("message") or ""), re.I)
    return float(match.group(1)) if match else 0.0


# --------------------------------------------------------------------------- #
# Per-provider request shaping
# --------------------------------------------------------------------------- #
def _openrouter_request(cfg: LLMConfig, system: str, user: str, max_tokens: int,
                        temperature: float, want_json: bool) -> tuple[str, dict, dict]:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if want_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "HTTP-Referer": "http://localhost:8093",
        "X-Title": "Mitacs Matcher",
    }
    return OPENROUTER_URL, payload, headers


# Current Gemini models spend "thinking" tokens out of the SAME budget as the
# visible answer, and `thinkingConfig.thinkingBudget: 0` is rejected with
# HTTP 400 — reasoning cannot be turned off. Measured live on the profile
# extraction prompt: ~1,900 thought tokens before a single output token.
#
# At maxOutputTokens=2048 that left 77 tokens for the answer, so the call came
# back 200-OK with JSON truncated mid-object — which surfaced as "model did not
# return parseable JSON" and failed every unit. At 8192 the same call finishes
# with finishReason STOP and parses cleanly.
#
# maxOutputTokens is a cap, not a reservation: unused budget is not billed, so
# reserving generous headroom costs nothing and prevents silent truncation.
GEMINI_THINKING_RESERVE = 4096
GEMINI_MIN_OUTPUT_TOKENS = 8192


def _gemini_request(cfg: LLMConfig, system: str, user: str, max_tokens: int,
                    temperature: float, want_json: bool) -> tuple[str, dict, dict]:
    budget = max(max_tokens + GEMINI_THINKING_RESERVE, GEMINI_MIN_OUTPUT_TOKENS)
    generation: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": budget}
    if want_json:
        generation["responseMimeType"] = "application/json"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": generation,
    }
    # Key travels in a header, not the query string, so it cannot leak into logs.
    url = f"{GEMINI_BASE}/models/{cfg.model}:generateContent"
    return url, payload, {"x-goog-api-key": cfg.api_key}


def _openrouter_text(body: dict) -> str:
    return body["choices"][0]["message"]["content"].strip()


def _gemini_text(body: dict) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback") or {}
        blocked = feedback.get("blockReason")
        raise CallError(f"Gemini returned no candidates{f' (blocked: {blocked})' if blocked else ''}")
    cand = candidates[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        reason = cand.get("finishReason")
        if reason == "MAX_TOKENS":
            raise CallError(
                "Gemini spent the whole output budget on reasoning tokens and returned "
                "no text — raise max_tokens (see GEMINI_MIN_OUTPUT_TOKENS)"
            )
        raise CallError(f"Gemini returned an empty response{f' ({reason})' if reason else ''}")
    return text


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
async def llm_call(
    client: httpx.AsyncClient,
    cfg: LLMConfig,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    want_json: bool = False,
) -> str:
    cfg = cfg.validated()
    build = _openrouter_request if cfg.provider == "openrouter" else _gemini_request
    read = _openrouter_text if cfg.provider == "openrouter" else _gemini_text
    url, payload, headers = build(cfg, system, user, max_tokens, temperature, want_json)

    # Retry transient rate limits / 5xx with jittered backoff, honouring
    # Retry-After. Bounded low: an upstream-rate-limited free model will 429
    # every time, and hammering it just makes the request feel hung. Fail fast
    # with the provider's own reason instead.
    attempts = 3
    last_reason = f"request to {cfg.model} failed"
    for attempt in range(attempts):
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code in (400, 401, 402, 403, 404):
                # Auth / credits / bad model — retrying won't help.
                raise CallError(_error_message(r, cfg.provider))
            if r.status_code == 429 or r.status_code >= 500:
                last_reason = _error_message(r, cfg.provider)
                if attempt == attempts - 1:
                    break
                wait = _retry_after_seconds(r) or 1.5 * (attempt + 1)
                await asyncio.sleep(min(wait, MAX_RETRY_WAIT) + random.uniform(0, 1.5))
                continue
            r.raise_for_status()
            return read(r.json())
        except CallError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            last_reason = f"{type(e).__name__}: {e}"
            if attempt == attempts - 1:
                break
            await asyncio.sleep(1.5 * (attempt + 1) + random.uniform(0, 1.5))
    raise CallError(last_reason)


# --------------------------------------------------------------------------- #
# JSON handling
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_loose(text: str) -> Any:
    """Parse model JSON that may be fenced or wrapped in prose.

    Models drop ```json fences even when told not to, and smaller models
    sometimes prepend a sentence. Strip fences, then fall back to the outermost
    balanced {...} / [...] span.
    """
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise CallError("model did not return parseable JSON")


async def json_call(
    client: httpx.AsyncClient,
    cfg: LLMConfig,
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> Any:
    """An `llm_call` whose output is parsed as JSON. Raises CallError on
    unparseable output so the caller counts it as a failed unit rather than
    treating it as an empty result."""
    raw = await llm_call(client, cfg, system, user, max_tokens=max_tokens,
                         temperature=0.0, want_json=True)
    return parse_json_loose(raw)


# --------------------------------------------------------------------------- #
# Model catalogues
# --------------------------------------------------------------------------- #
async def list_models(provider: str, api_key: str = "") -> list[dict]:
    """Live model list for the picker.

    OpenRouter publishes its catalogue without auth. Gemini needs the key, so we
    fall back to a curated list until one is supplied. Either way the UI also
    accepts a typed model id, so a stale catalogue never blocks the user.
    """
    async with httpx.AsyncClient(timeout=20) as client:
        if provider == "openrouter":
            try:
                r = await client.get(OPENROUTER_MODELS_URL)
                r.raise_for_status()
                out = []
                for m in r.json().get("data", []):
                    pricing = m.get("pricing") or {}
                    free = str(pricing.get("prompt", "0")) in ("0", "0.0", "-1")
                    out.append({
                        "id": m.get("id", ""),
                        "name": m.get("name") or m.get("id", ""),
                        "free": bool(free) or ":free" in str(m.get("id", "")),
                        "context": m.get("context_length"),
                    })
                out.sort(key=lambda m: (not m["free"], m["id"]))
                return [m for m in out if m["id"]]
            except (httpx.HTTPError, ValueError, KeyError):
                return [{"id": DEFAULT_MODELS["openrouter"], "name": DEFAULT_MODELS["openrouter"], "free": False}]

        if provider == "gemini":
            if not api_key.strip():
                return GEMINI_FALLBACK
            try:
                r = await client.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": api_key})
                r.raise_for_status()
                out = []
                for m in r.json().get("models", []):
                    if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                        continue
                    mid = str(m.get("name", "")).replace("models/", "")
                    if mid:
                        out.append({"id": mid, "name": m.get("displayName") or mid,
                                    "free": False, "context": m.get("inputTokenLimit")})
                out.sort(key=lambda m: m["id"])
                return out or GEMINI_FALLBACK
            except (httpx.HTTPError, ValueError, KeyError):
                return GEMINI_FALLBACK

    return []
