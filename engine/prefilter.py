"""
Stage 1 — lexical recall filter (BM25-style, embeddings-free).

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
Galt RAG reads every token of the corpus for every query. That is affordable
when the corpus is one person's documents. Here the candidate set is 3,359
Mitacs projects (~12 MB); reading all of them on every run costs ~340 LLM calls
per query even at a generous batch size. For an interactive UI that is too slow
and, on a paid model, too expensive.

So this stage narrows the field before the model does the judging. Two rules
keep it honest:

  1. It is NOT a relevance verdict. It is a *recall* filter, tuned to over-admit.
     Nothing here decides ranking — every survivor is still read and judged by
     the model in Stage 2, and the scores the user sees come only from there.
  2. It is NOT vector similarity. No embeddings, no cosine, no ANN index. It is
     transparent term matching with IDF weighting, so any admission or omission
     can be explained by pointing at the terms that fired.

Set `width=0` to disable it entirely and put every project through the model —
the faithful-to-Galt-RAG setting, available to anyone willing to pay for it.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Terms this generic carry no signal about fit; they appear in most projects.
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "will", "from", "are", "was", "have",
    "has", "not", "but", "can", "all", "its", "our", "their", "them", "they", "these",
    "those", "such", "who", "which", "what", "when", "where", "how", "why", "into",
    "than", "then", "also", "more", "most", "some", "any", "each", "other", "using",
    "use", "used", "new", "well", "may", "one", "two", "both", "over", "under",
    "project", "projects", "research", "student", "students", "intern", "internship",
    "work", "working", "develop", "developing", "development", "study", "studies",
    "approach", "method", "methods", "based", "within", "between", "during", "including",
    "university", "canada", "canadian", "supervisor", "faculty", "team", "field",
    "area", "areas", "system", "systems", "data", "model", "models", "high", "large",
    "novel", "aims", "aim", "goal", "goals", "results", "provide", "support",
}

_TOKEN = re.compile(r"[a-z][a-z0-9+#.\-]{1,}")


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall((text or "").lower())
    return [t.strip(".-") for t in toks if len(t) > 2 and t not in STOPWORDS]


def _project_text(p: dict) -> str:
    return " ".join(
        str(p.get(k) or "")
        for k in ("title", "description", "researchArea", "studentRoles", "studentSkills")
    )


class Prefilter:
    """IDF-weighted term index over the project corpus.

    Built once at startup (~3s for 3,359 projects) and reused for every run.
    """

    def __init__(self, projects: list[dict]) -> None:
        self.projects = projects
        self.doc_tokens: list[Counter] = []
        self.doc_len: list[int] = []
        df: Counter = Counter()

        for p in projects:
            counts = Counter(tokenize(_project_text(p)))
            self.doc_tokens.append(counts)
            self.doc_len.append(sum(counts.values()) or 1)
            df.update(counts.keys())

        n = max(len(projects), 1)
        self.idf: dict[str, float] = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0

        # Title tokens are weighted higher — a term in the title is a much
        # stronger signal of what a project is actually about.
        self.title_tokens: list[set[str]] = [set(tokenize(p.get("title") or "")) for p in projects]

    def query_terms(self, profile: dict) -> Counter:
        """Build a weighted bag of terms from the capability profile.

        Skills and interests weigh more than incidental prose, and a
        `demonstrated` skill weighs more than one merely `mentioned`.
        """
        weights: Counter = Counter()

        def add(text: str, w: float) -> None:
            for t in tokenize(text):
                weights[t] += w

        strength_w = {"demonstrated": 3.0, "used": 2.0, "mentioned": 1.0}
        for s in profile.get("skills", []):
            if isinstance(s, dict):
                add(str(s.get("name") or ""), strength_w.get(str(s.get("strength")), 1.5))
            else:
                add(str(s), 1.5)

        for i in profile.get("interests", []):
            add(str(i), 3.0)
        for d in profile.get("domains", []):
            add(str(d), 2.5)
        for p in profile.get("projects", []):
            if isinstance(p, dict):
                add(str(p.get("name") or ""), 2.0)
                add(str(p.get("what") or ""), 1.0)
            else:
                add(str(p), 1.0)
        for e in profile.get("experience", []):
            if isinstance(e, dict):
                add(str(e.get("role") or ""), 1.0)
                add(str(e.get("what") or ""), 1.0)
            else:
                add(str(e), 1.0)

        return weights

    def rank(self, profile: dict, width: int) -> list[tuple[int, float]]:
        """Return `[(project_index, score)]` for the top `width` projects.

        `width <= 0` means no filtering: every project goes to the model.
        """
        if width <= 0:
            return [(i, 0.0) for i in range(len(self.projects))]

        terms = self.query_terms(profile)
        if not terms:
            # No usable profile terms — admit an arbitrary slice rather than
            # silently returning nothing, and let the model do the judging.
            return [(i, 0.0) for i in range(min(width, len(self.projects)))]

        k1, b = 1.5, 0.75
        scored: list[tuple[int, float]] = []
        for i, counts in enumerate(self.doc_tokens):
            score = 0.0
            dl = self.doc_len[i]
            for term, qw in terms.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / self.avg_len))
                hit = qw * idf * norm
                if term in self.title_tokens[i]:
                    hit *= 1.6
                score += hit
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:width]
