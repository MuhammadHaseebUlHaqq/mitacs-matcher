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
                         (Galt RAG's map-reduce, query held fixed)   ~1-2 calls

  profile   ──► STAGE 1  lexical recall filter (BM25, no embeddings)
                         3,359 projects → 60                         0 calls

            ──► STAGE 2  JUDGE  each survivor read in full and scored
                                on a calibrated 0–100 scale          ~10 calls
                                                          ──► ranked results
```

**Stage 0 — profile (`engine/profile.py`)** is Galt RAG's map-reduce with the
query fixed: every passage of your knowledge base is read by the model and asked
*what does this prove this person can do?*, then the partial profiles are folded
by a real consolidation pass. Not concatenation — concatenating produces
duplicates and inconsistent classification. Built once per knowledge base and
cached, so a CV costs one or two calls and then nothing.

**Stage 1 — filter (`engine/prefilter.py`)** narrows the field before anything
is spent. IDF-weighted term matching with title boosting, no embeddings, so any
admission or omission can be explained by pointing at the terms that fired.

**Stage 2 — judge (`engine/matching.py`)** reads each survivor in full and scores
it against the profile on a scale calibrated to mean the same thing for every
project. Every score you see comes from here; the filter never ranks.

Carried over from the source projects: **structure-aligned chunking**
(`engine/corpus.py`) — sections packed to a ceiling and never split, only an
oversize section is hard-split with overlap, every unit keeping a provenance
anchor; **bounded fan-out** with a semaphore; **jittered retries** honouring
`Retry-After`; and **honest degradation** — a failed unit is counted and
surfaced as a partial-result warning, never silently dropped.

### Why there is no screening pass

An earlier design put a model "map" stage between the filter and the judge:
batches of five projects, each asked only whether the person was a plausible
fit, feeding a judge pass over the survivors. Measured against the real corpus,
that stage was mostly waste.

BM25 scores decay gradually rather than falling off a cliff. Against a real
profile, the projects at rank 400 still scored **31%** of the top hit, and rank
200 was *"Memorize book reading Android App based on GAIOLA"* at 41% — visibly
off-topic. So a width of 400 admitted several hundred junk projects, and the map
pass then spent ~75 of the run's ~101 calls rejecting things the filter had
already ranked near the bottom.

Narrowing the filter instead (3,359 → 60) and judging that set directly costs
**~10 calls instead of ~101**. Quality does not drop: the judge reads the full
project text where the map pass only skimmed a trimmed description to make a
keep/drop call. The plausible set is now read more carefully, not less.

**The cost, stated plainly.** BM25 cannot match meaning across different
vocabulary, and the judge only ever sees what the filter admits. A project
describing the same work in words your profile never uses is missed before any
model reads it — the map pass, being semantic, could have caught that. The
`Projects read` selector is the dial that trades calls for that recall; raise it
to 240 if you would rather pay for the cushion.

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
.venv/Scripts/python.exe -m pytest tests/ -q     # 87 passed
```

Every LLM call is faked, so the suite is offline and free. It runs against the
real 3,359-project corpus and parses the real CV PDF. What is under test is the
pipeline's contract: chunking loses no text, ids the model invents never become
matches, failures degrade honestly, and the server keeps nothing between
requests.

---

## Layout

```
engine/llm.py         OpenRouter + Gemini transport — retries, honest errors, model catalogues
engine/corpus.py      structure-aligned chunking → citable evidence units
engine/ingest.py      PDF / DOCX / MD / TXT → Document (swap in LlamaParse here)
engine/profile.py     STAGE 0 — knowledge base → capability profile (map-reduce)
engine/prefilter.py   STAGE 1 — BM25 recall filter (the only non-LLM step)
engine/matching.py    STAGE 2 — the comparative judge
app.py                FastAPI: parse, profile, match, export — stateless throughout
static/landing.html   brutalist landing page
static/index.html     the matcher console — owns the workspace in localStorage
static/brutal.css     design system (tokens taken from the v0 brutalist template)
data/projects.json.gz 3,359 Mitacs projects
```

## Tuning

| knob | where | default |
|---|---|---|
| provider | sidebar | OpenRouter · Gemini |
| model | sidebar | `meta-llama/llama-3.3-70b-instruct` · `gemini-flash-lite-latest` |
| results wanted | sidebar | 15 (max 120) |
| projects read | sidebar | top 60 (30 · 60 · 120 · 240) |
| default width | `DEFAULT_WIDTH` env | 60 |
| concurrency | `MAX_CONCURRENCY` env | 8 (4 in deployment) |
| projects per judge call | `matching.JUDGE_BATCH_SIZE` | 6 |
| chunk ceiling | `corpus.CHUNK_CEILING_CHARS` | 12,000 |

`Projects read` is the only number that costs anything: at `JUDGE_BATCH_SIZE` 6
it sets the call count, the wall-clock time, and how much of your free quota a
run spends.

## Where state lives

Nowhere on the server. The browser owns the workspace — parsed document text,
the built profile, and the per-passage extract cache all live in its
`localStorage` and travel with each request, the same way the API key always
has. Two things follow: the app runs on a host with no persistent disk, and a
shared deployment cannot leak one person's CV to another because it never holds
one. The trade is that clearing site data clears the workspace.

## Providers

Pick either in the sidebar. The key is sent per request, used, and discarded —
never stored, cached, or logged.

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
