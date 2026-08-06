---
title: Mitacs Matcher
emoji: 🎓
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Mitacs Matcher

Upload your own documents, get the Mitacs Globalink projects you actually fit —
scored, ranked, and each one justified by a quote from your own CV.

Corpus: **3,359 projects** (Summer 2027 cycle, scraped from
`globalink.mitacs.ca/api/sasprojectlistpaging`).

---

## The RAG method, and why it is not the usual one

The retrieval here follows **`ZaynIkhlaq/Grag`** (Galt RAG) and the matcher in
**`bilalrana351/bideez-backend`**. Both refuse the same thing: **there are no
embeddings, no vector database, no cosine similarity, and no ANN index anywhere
in this codebase.**

Galt RAG's argument, which applies directly here:

> Conventional RAG pre-computes embeddings and retrieves by vector similarity.
> That retrieval step can't *reason* — it misses paraphrases, synonyms,
> antonyms, and inferences.

For this problem that failure is not hypothetical. A CV saying
`kubeadm, Calico CNI, Helm on AKS` has to match a project asking for
*"cloud-native orchestration"*. `benchmarked vLLM against HuggingFace, KV-cache
OOM sweep` has to match *"efficient LLM serving"*. Those are **inferences**, not
nearest neighbours. So relevance is **judged by a model that reads the text**,
never estimated from a vector.

### Pipeline

```
  documents ──► STAGE 0  chunk → parallel map → tree reduce ──► capability profile
                         (Galt RAG's map-reduce, query held fixed)

  profile   ──► STAGE 1  lexical recall filter (BM25, no embeddings)
                         3,359 projects → N candidates

            ──► STAGE 2  MAP    batches of projects, bounded fan-out
                                → batch-local candidates + evidence

            ──► STAGE 3  JUDGE  survivors re-scored head-to-head
                                → calibrated, comparable 0–100  ──► ranked results
```

**Stage 0 — profile (`engine/profile.py`)** is Galt RAG's map-reduce with the
query fixed: every passage of your knowledge base is read by the model and asked
*what does this prove this person can do?*, then the partial profiles are folded
by a real consolidation pass. Not concatenation — bideez is explicit that
concatenating produces duplicates and inconsistent classification. Built once
per knowledge base and cached.

**Stage 2/3 — match (`engine/matching.py`)** is bideez's
`src/mastra/matching/run.ts`, with requirements/corpus swapped for
profile/projects:

| bideez | here |
|---|---|
| MAP over (requirement batch × corpus chunk) | MAP over batches of projects against the profile |
| GATHER — validate each cited `evidenceUnitId` resolves | GATHER — validate each cited `projectId` resolves; drop and count the rest |
| JUDGE — re-judge gathered evidence head-to-head | JUDGE — re-score survivors on a calibrated scale |
| "chunk-local relevance scores are NOT comparable" | map `fit` chooses *who* gets judged, never the ranking |
| requirements with no evidence skip the judge | projects with no candidate skip the judge |

Also carried over: **structure-aligned chunking** (`engine/corpus.py`) — sections
packed to a ceiling and never split, only an oversize section is hard-split with
overlap, every unit keeping a provenance anchor; **bounded fan-out** with a
semaphore; **jittered retries** honouring `Retry-After`; and **honest
degradation** — a failed unit is counted and surfaced as a partial-result
warning, never silently dropped.

### The one deviation, stated plainly

Galt RAG reads *every* token for *every* query. That is affordable for one
person's documents; for 3,359 projects it is ~670 LLM calls per run. So
**Stage 1 (`engine/prefilter.py`)** narrows the field first. Two rules keep it
honest:

1. **It never ranks.** It is a recall filter tuned to over-admit. Every survivor
   is still read and scored by the model, and every score you see comes from
   Stage 3.
2. **It is not vector similarity.** IDF-weighted term matching — transparent, so
   any admission or omission can be explained by pointing at the terms that fired.

Set the width selector to **Exhaustive** (`width=0`) to skip it and put every
project through the model. That is the faithful-to-Galt-RAG setting, available
whenever you want to pay for it.

---

## Run it

```bash
cd C:/Users/786/Documents/mitacs-prep/matcher
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn app:app --port 8093 --reload
```

Then open <http://127.0.0.1:8093> — landing page, with the console at `/app`.

### Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # 71 passed
```

Every LLM call is faked, so the suite is offline and free. It runs against the
real 3,359-project corpus and parses the real CV PDF. What is under test is the
pipeline's contract: chunking loses no text, unresolvable ids are dropped and
counted, failures degrade honestly, and the judge's score — not the batch-local
map score — decides the ranking.

---

## Layout

```
engine/llm.py         OpenRouter + Gemini transport — retries, honest errors, model catalogues
engine/store.py       per-session knowledge-base persistence (parsed text + cached profile)
engine/corpus.py      structure-aligned chunking → citable evidence units
engine/ingest.py      PDF / DOCX / MD / TXT → Document (swap in LlamaParse here)
engine/profile.py     STAGE 0 — knowledge base → capability profile (map-reduce)
engine/prefilter.py   STAGE 1 — BM25 recall filter (the only non-LLM step)
engine/matching.py    STAGES 2-3 — map → gather → judge
app.py                FastAPI: KB upload, profile, match (SSE progress), export
static/landing.html   brutalist landing page
static/index.html     the matcher console
static/brutal.css     design system (tokens taken from the v0 brutalist template)
data/projects.json    3,359 Mitacs projects
```

## Tuning

| knob | where | default |
|---|---|---|
| provider | sidebar | OpenRouter · Gemini |
| model | sidebar | `meta-llama/llama-3.3-70b-instruct` · `gemini-2.0-flash` |
| results wanted | sidebar | 50 (max 200) |
| screening width | sidebar | top 400 · `0` = exhaustive |
| concurrency | `MAX_CONCURRENCY` env | 8 |
| projects per map call | `matching.MAP_BATCH_SIZE` | 5 |
| judge pool floor | `matching.MIN_JUDGE_POOL` | 24 |
| chunk ceiling | `corpus.CHUNK_CEILING_CHARS` | 12,000 |

`MIN_JUDGE_POOL` exists because without it a small `top_n` makes the judge pool
so narrow that ties in the deliberately non-comparable map score decide the
ranking — the exact failure the judge stage is there to prevent.

## Providers

Pick either in the sidebar. The key is sent per request, used, and discarded —
never stored, cached, or logged, and never written to the knowledge-base store.

| | OpenRouter | Gemini |
|---|---|---|
| key format | `sk-or-…` | `AIza…` |
| get one | [openrouter.ai/keys](https://openrouter.ai/keys) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| catalogue | live, 338 models, free tier flagged | live once a key is entered, curated list before |
| key transport | `Authorization: Bearer` | `x-goog-api-key` header (never the query string) |

Any model id can be typed directly, so a stale catalogue never blocks you.

### Gemini free-tier notes (measured, Aug 2026)

Three things bit during live testing and are now handled in code — worth knowing
if you tune anything:

1. **Pinned model ids retire.** `gemini-2.0-flash` returns `limit: 0` (no free
   quota at all) and `gemini-2.5-flash` is closed to new users. Use the rolling
   `gemini-flash-latest` / `gemini-flash-lite-latest` aliases, which is what the
   defaults and the offline fallback now do.
2. **Reasoning tokens come out of the answer's budget**, and
   `thinkingConfig.thinkingBudget: 0` is rejected (HTTP 400) — thinking cannot be
   switched off. The profile-extraction prompt burned ~1,900 thought tokens
   before emitting anything; at `maxOutputTokens: 2048` that left 77 tokens and
   the JSON truncated mid-object. `_gemini_request` now adds
   `GEMINI_THINKING_RESERVE` on top of every request, floored at 8,192.
3. **Free tiers are per-minute and small.** `gemini-flash-latest` resolves to
   `gemini-3.6-flash` at **5 requests/minute**, which a fan-out pipeline
   exhausts instantly. Gemini also sends no `Retry-After` header — the delay is
   in a `RetryInfo` detail on the error body — and asks for ~45s, so the old 5s
   retry ceiling wasted every attempt. Fixed by `_retry_after_seconds` plus
   `MAX_RETRY_WAIT` (50s).

For a free Gemini key, run with `MAX_CONCURRENCY=2` and prefer
`gemini-flash-lite-latest`:

```bash
MAX_CONCURRENCY=2 .venv/Scripts/python.exe -m uvicorn app:app --port 8093
```

## Design

The interface follows the v0 **Brutalist AI SaaS Landing Page** template. The
tokens in `static/brutal.css` are lifted from that template's compiled
stylesheet rather than eyeballed — `--radius: 0rem`, 2px solid borders,
JetBrains Mono throughout, the `43 23% 93%` / `0 0% 4%` light pair and
`0 0% 6%` / `43 23% 93%` dark pair, the `20 90% 50%` orange, and the dot grid
`radial-gradient(circle, #c4c2b8 1px, transparent 1px)`.

The one thing not lifted is the display face: `GeistPixelGrid` is Vercel
proprietary, so `.dotted` reproduces the dot-matrix letterforms by clipping a
radial-dot gradient to the text instead.

## Not done yet

- **Scanned PDFs** — `pypdf` extracts no text from image-only PDFs; the file is
  flagged with an error rather than silently contributing nothing. OCR or
  LlamaParse would fix it.
- **Auth** — sessions are a client-generated id with no login behind them. The
  store is keyed by that id, so anyone with it can read that knowledge base.
- **OpenRouter is still unverified live.** Gemini has now been exercised
  end-to-end against the real API (profile build + full match run on a real CV);
  OpenRouter's path is only pinned by mocked-transport tests.
- **Occasional malformed JSON.** `gemini-flash-lite-latest` failed 1 map call in
  6 with unparseable output. That is counted and surfaced as a partial-result
  warning rather than hidden, but a schema-repair retry would recover it.
