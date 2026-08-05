"""
Provider transport tests.

The Gemini path cannot be exercised against the live API here (no key), so its
request shaping and response parsing are pinned against a mocked transport
instead. These are the tests that catch a malformed body or a response shape we
read wrongly — the failure modes that would otherwise only show up in front of
the user with a real key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.llm import (  # noqa: E402
    GEMINI_FALLBACK, GEMINI_MIN_OUTPUT_TOKENS, GEMINI_THINKING_RESERVE,
    PROVIDERS, CallError, LLMConfig,
    MAX_RETRY_WAIT, _error_message, _gemini_request, _gemini_text,
    _openrouter_request, _openrouter_text, _retry_after_seconds,
    json_call, list_models, llm_call,
)

OR = LLMConfig(provider="openrouter", api_key="sk-or-test", model="meta-llama/llama-3.3-70b-instruct")
GM = LLMConfig(provider="gemini", api_key="AIza-test", model="gemini-flash-latest")


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_providers_are_the_two_supported():
    assert PROVIDERS == ("openrouter", "gemini")


def test_missing_key_is_rejected_before_any_network_call():
    with pytest.raises(CallError, match="missing API key"):
        LLMConfig(provider="gemini", api_key="  ", model="x").validated()


def test_unknown_provider_is_rejected():
    with pytest.raises(CallError, match="unknown provider"):
        LLMConfig(provider="anthropic", api_key="k", model="x").validated()


# --------------------------------------------------------------------------- #
# request shaping
# --------------------------------------------------------------------------- #
def test_gemini_request_uses_the_documented_shape():
    url, payload, headers = _gemini_request(GM, "SYS", "USER", 4096, 0.0, True)
    assert url.endswith("/models/gemini-flash-latest:generateContent")
    assert payload["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "USER"
    # budget carries thinking headroom on top of the request (see _gemini_request)
    assert payload["generationConfig"]["maxOutputTokens"] == 4096 + GEMINI_THINKING_RESERVE
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_key_travels_in_a_header_not_the_url():
    """A key in the query string leaks into access logs and referrers."""
    url, _, headers = _gemini_request(GM, "s", "u", 10, 0.0, False)
    assert "AIza-test" not in url
    assert "key=" not in url
    assert headers["x-goog-api-key"] == "AIza-test"


def test_gemini_omits_json_mime_when_not_requested():
    _, payload, _ = _gemini_request(GM, "s", "u", 10, 0.0, False)
    assert "responseMimeType" not in payload["generationConfig"]


def test_openrouter_request_uses_chat_completions_shape():
    url, payload, headers = _openrouter_request(OR, "SYS", "USER", 256, 0.0, True)
    assert url.endswith("/chat/completions")
    assert payload["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert payload["response_format"] == {"type": "json_object"}
    assert headers["Authorization"] == "Bearer sk-or-test"


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #
def test_gemini_text_joins_multiple_parts():
    body = {"candidates": [{"content": {"parts": [{"text": "he"}, {"text": "llo"}]}}]}
    assert _gemini_text(body) == "hello"


def test_gemini_blocked_prompt_reports_the_reason():
    with pytest.raises(CallError, match="blocked: SAFETY"):
        _gemini_text({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})


def test_openrouter_text_extraction():
    assert _openrouter_text({"choices": [{"message": {"content": "  hi  "}}]}) == "hi"


def test_error_message_prefers_the_providers_own_text():
    r = httpx.Response(429, json={"error": {"message": "rate limited upstream"}})
    assert _error_message(r) == "rate limited upstream"
    r2 = httpx.Response(500, text="not json")
    assert _error_message(r2) == "HTTP 500"


# --------------------------------------------------------------------------- #
# end-to-end through a mocked transport
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_gemini_json_call_roundtrip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": '{"skills": ["K8s"]}'}]}}]
        })

    async with mock_client(handler) as client:
        out = await json_call(client, GM, "sys", "user")
    assert out == {"skills": ["K8s"]}
    assert seen["key"] == "AIza-test"
    assert "generativelanguage.googleapis.com" in seen["url"]


@pytest.mark.asyncio
async def test_openrouter_json_call_roundtrip():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}]
        })

    async with mock_client(handler) as client:
        assert await json_call(client, OR, "sys", "user") == {"ok": True}


@pytest.mark.asyncio
async def test_auth_failure_does_not_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    async with mock_client(handler) as client:
        with pytest.raises(CallError, match="invalid key"):
            await llm_call(client, GM, "s", "u")
    assert calls["n"] == 1, "auth errors must fail fast, not burn retries"


@pytest.mark.asyncio
async def test_rate_limit_retries_then_reports_the_upstream_reason():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "upstream rate-limit"}},
                              headers={"retry-after": "0"})

    async with mock_client(handler) as client:
        with pytest.raises(CallError, match="upstream rate-limit"):
            await llm_call(client, OR, "s", "u")
    assert calls["n"] == 3, "should exhaust the bounded retry budget"


@pytest.mark.asyncio
async def test_transient_500_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})

    async with mock_client(handler) as client:
        assert await llm_call(client, OR, "s", "u") == "recovered"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_gemini_catalogue_falls_back_when_unlisted():
    assert await list_models("gemini", "") == GEMINI_FALLBACK


# --------------------------------------------------------------------------- #
# thinking-token budget (found against the live API)
# --------------------------------------------------------------------------- #
def test_small_gemini_budget_is_floored():
    """Verified live: reasoning tokens come out of the SAME budget as the answer
    (~1,900 thoughts before any output). At 2048 the JSON truncated mid-object
    and every unit failed to parse; the floor prevents that."""
    _, payload, _ = _gemini_request(GM, "s", "u", 10, 0.0, True)
    assert payload["generationConfig"]["maxOutputTokens"] == GEMINI_MIN_OUTPUT_TOKENS


def test_gemini_budget_scales_with_the_request():
    """A large request must get thinking headroom ON TOP of what it asked for,
    not be silently capped at the floor."""
    _, payload, _ = _gemini_request(GM, "s", "u", 9000, 0.0, True)
    assert payload["generationConfig"]["maxOutputTokens"] == 9000 + GEMINI_THINKING_RESERVE


def test_pipeline_budgets_all_clear_the_measured_thinking_overhead():
    """The real call sites (1600 extract / 1800 map / 3000 judge) must each end
    up with room for ~2k thought tokens plus their own output."""
    for requested in (1600, 1800, 3000):
        _, payload, _ = _gemini_request(GM, "s", "u", requested, 0.0, True)
        assert payload["generationConfig"]["maxOutputTokens"] >= requested + 4000


def test_openrouter_budget_is_not_floored():
    """The floor is a Gemini quirk; OpenRouter must keep the caller's number."""
    _, payload, _ = _openrouter_request(OR, "s", "u", 10, 0.0, True)
    assert payload["max_tokens"] == 10


def test_exhausted_budget_error_names_reasoning_tokens():
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    with pytest.raises(CallError, match="reasoning tokens"):
        _gemini_text(body)


def test_fallback_catalogue_uses_rolling_aliases():
    """Pinned Gemini versions retire — `gemini-2.0-flash` now returns limit: 0
    and `gemini-2.5-flash` is closed to new users. The offline fallback must not
    hand out an id that no longer works."""
    ids = [m["id"] for m in GEMINI_FALLBACK]
    assert "gemini-2.5-flash" not in ids and "gemini-2.0-flash" not in ids
    assert any(i.endswith("-latest") for i in ids)


# --------------------------------------------------------------------------- #
# retry pacing (found against the live API)
# --------------------------------------------------------------------------- #
def test_retry_delay_read_from_standard_header():
    r = httpx.Response(429, headers={"retry-after": "12"}, json={})
    assert _retry_after_seconds(r) == 12.0


def test_retry_delay_read_from_gemini_retryinfo_detail():
    """Gemini sends no Retry-After header — the delay lives in a RetryInfo
    detail on the error body."""
    r = httpx.Response(429, json={"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45.3s"}
    ]}})
    assert _retry_after_seconds(r) == 45.3


def test_retry_delay_falls_back_to_the_message_text():
    r = httpx.Response(429, json={"error": {"message": "Quota exceeded. Please retry in 30.7s."}})
    assert _retry_after_seconds(r) == 30.7


def test_retry_delay_absent_is_zero_not_an_error():
    assert _retry_after_seconds(httpx.Response(429, json={"error": {}})) == 0.0
    assert _retry_after_seconds(httpx.Response(500, text="nope")) == 0.0


def test_retry_ceiling_covers_per_minute_free_tiers():
    """gemini-3.6-flash free tier is 5 req/min and asks for ~45s. A ceiling
    below that makes every retry useless, which is what broke the live run."""
    assert MAX_RETRY_WAIT >= 45
