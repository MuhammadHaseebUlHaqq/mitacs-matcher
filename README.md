# Mitacs Matcher

Upload your CV and get the Mitacs Globalink projects you actually fit — scored,
ranked, and each one backed by a quote from your own documents.

**Live:** <https://mitacs-matcher.vercel.app>

Corpus: **3,359 projects** (Summer 2027 cycle, scraped from
`globalink.mitacs.ca/api/sasprojectlistpaging`).

---

## How it works

Three steps. Only two of them cost anything.

```
your documents ──► BUILD PROFILE   read each passage, merge into one profile
                                   ~1-2 model calls, cached

profile        ──► FILTER          keyword match over all 3,359 projects
                                   0 model calls, instant

               ──► JUDGE           read the top 60 in full, score each 0-100
                                   ~10 model calls          ──► ranked results
```

### 1. Build profile

Your uploaded files are split into passages. Each passage goes to the model with
one question: *what does this prove this person can do?* The answers are then
merged into a single profile — skills, projects, experience, interests, each
with the evidence behind it.

Merging is a real model call, not string concatenation. Concatenating produces
duplicates and inconsistent labelling.

A typical CV is one or two passages, so this costs one or two calls. Results are
cached by content, so re-running only re-reads passages that changed.

### 2. Filter

BM25 keyword matching (`engine/prefilter.py`) scores all 3,359 projects against
your profile and keeps the top 60. No model, no embeddings, no vector database —
just term matching with three adjustments:

- **Rare words count more.** "kubernetes" appears in 14 of 3,359 projects and is
  a strong signal; "learning" appears in 1,262 and is nearly useless.
- **Repetition counts, with diminishing returns.**
- **Long descriptions don't win just for being long.**

Plus two additions: a term in the title counts 1.6x, and terms from skills you
have actually demonstrated outweigh ones you only mentioned.

This step is free and takes milliseconds. It never ranks anything — it only
decides which projects the model gets to see.

### 3. Judge

Each of the 60 projects is read in full and scored 0-100 on a fixed scale, in
batches of 6. Every score, rationale, evidence quote and gap you see comes from
here. The results are sorted by score and the top N are returned.

## Why no embeddings

A CV saying `kubeadm, Calico CNI, Helm on AKS` has to match a project asking for
*"cloud-native orchestration"*. `benchmarked vLLM against HuggingFace, KV-cache
OOM sweep` has to match *"efficient LLM serving"*.

Those are inferences, not nearest neighbours. Vector similarity can't reason
about them. So relevance is decided by a model that reads the text.

## What this design costs

The filter is keyword-based, so it cannot match meaning across different
wording. A project describing the same work in words your CV never uses is
dropped before any model reads it. The **Projects read** setting is the dial:
raise it to 120 or 240 to trade more model calls for more coverage.

An earlier version put a cheap model pass over 400 projects before judging. It
was measured and mostly wasted — BM25 scores fall off gradually, so at width 400
the bottom of the list was visibly off-topic, and that pass spent about 75 of
the run's 101 calls rejecting projects the filter had already ranked last.
Narrowing to 60 and judging directly costs ~10 calls with no loss in quality,
because the judge reads the full project text where the screening pass only
skimmed.

---

## Run it locally

```bash
cd matcher
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn app:app --port 8093 --reload
```

Open <http://127.0.0.1:8093> — landing page, with the app at `/app`.

### Tests

```bash
.venv/Scripts/python.exe -m pytest -q     # 87 passed
```

Every model call is faked, so the suite is offline and free. It runs against the
real 3,359-project corpus and parses a real CV PDF. What it checks: chunking
loses no text, project ids the model invents never become matches, failures are
reported instead of hidden, and the server keeps nothing between requests.

### Deploy

Push to `main` — Vercel builds from `vercel.json`. There is nothing to
configure and no environment variables to set, because users bring their own
API key.

---

## Where your data lives

**Nowhere on the server.** Your browser holds the parsed text, the profile and
the passage cache in `localStorage`, and sends them with each request — the same
way the API key already worked.

Two consequences: the app runs on a host with no disk, and a shared deployment
cannot leak one person's CV to another because it never has it. The trade-off is
that clearing your browser data clears your workspace.

---

## Layout

```
engine/llm.py         OpenRouter + Gemini transport - retries, error reporting, model lists
engine/ingest.py      PDF / DOCX / XLSX / CSV / MD / TXT -> plain text (no model involved)
engine/corpus.py      splits documents into passages, keeping section boundaries
engine/profile.py     step 1 - documents to capability profile
engine/prefilter.py   step 2 - BM25 filter (the only step with no model call)
engine/matching.py    step 3 - the judge
app.py                FastAPI: parse, profile, match, export. Stateless.
static/landing.html   landing page
static/index.html     the app - owns the workspace in localStorage
static/brutal.css     design system
data/projects.json.gz 3,359 Mitacs projects
```

## Settings

| Setting | Where | Default |
|---|---|---|
| Provider | sidebar | OpenRouter or Gemini |
| Model | sidebar | `meta-llama/llama-3.3-70b-instruct` / `gemini-flash-lite-latest` |
| Results wanted | sidebar | 15 (max 120) |
| Projects read | sidebar | 60 (30 / 60 / 120 / 240) |
| Default width | `DEFAULT_WIDTH` env | 60 |
| Concurrent calls | `MAX_CONCURRENCY` env | 8 locally, 4 deployed |
| Projects per judge call | `matching.JUDGE_BATCH_SIZE` | 6 |
| Passage size limit | `corpus.CHUNK_CEILING_CHARS` | 12,000 chars |

**Projects read** is the only setting that costs money. At 6 projects per call it
decides the number of calls, how long a run takes, and how much of your free
quota it uses.

## Providers

Pick either in the sidebar. The key is sent with the request, used, and
discarded — never stored, cached, or logged.

| | OpenRouter | Gemini |
|---|---|---|
| Key format | `sk-or-…` | `AIza…` |
| Get one | [openrouter.ai/keys](https://openrouter.ai/keys) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Model list | live, free models flagged | live once a key is entered |
| Key sent as | `Authorization: Bearer` | `x-goog-api-key` header (never in the URL) |

You can type any model id directly, so an out-of-date model list never blocks you.

### Gemini free-tier notes (measured, Aug 2026)

Three things caused failures in live testing and are now handled in code:

1. **Pinned model ids retire.** `gemini-2.0-flash` has no free quota at all and
   `gemini-2.5-flash` is closed to new users. Use the rolling
   `gemini-flash-latest` / `gemini-flash-lite-latest` names, which is what the
   defaults do.
2. **Reasoning tokens come out of the answer's budget**, and reasoning cannot be
   switched off. One prompt spent ~1,900 tokens thinking before writing
   anything; at a 2,048-token limit that left 77 tokens and the JSON was cut off
   mid-object. Requests now reserve extra headroom, floored at 8,192.
3. **Free limits are per minute and small.** `gemini-flash-latest` allows 5
   requests per minute. Gemini also sends no `Retry-After` header — the delay is
   inside the error body — and asks for about 45 seconds, so a short retry
   ceiling wasted every attempt. Retries now read that delay, up to 50 seconds.

On a free Gemini key, use `gemini-flash-lite-latest` and lower the concurrency:

```bash
MAX_CONCURRENCY=2 .venv/Scripts/python.exe -m uvicorn app:app --port 8093
```

## Design

Brutalist: zero border radius, 2px borders, JetBrains Mono throughout, an orange
accent, and a dot-grid background. Tokens live in `static/brutal.css` and were
taken from a brutalist landing-page template rather than picked by eye.

## Known limits

- **Scanned PDFs don't work.** `pypdf` extracts no text from image-only PDFs. The
  file is flagged with an error rather than silently contributing nothing. OCR
  would fix it.
- **Keyword filtering can miss differently-worded projects.** See *What this
  design costs* above.
- **The judge prompt has not been run against a real model yet.** Every test
  fakes the model call. The first real run is what will show whether scores are
  well spread or bunched together.
- **Occasional malformed JSON.** In earlier testing `gemini-flash-lite-latest`
  returned unparseable output on about 1 call in 6. That is counted and reported
  as a partial result rather than hidden, but a retry that repairs the JSON
  would recover it.
