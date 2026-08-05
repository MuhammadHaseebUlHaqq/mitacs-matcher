"""
Knowledge-base corpus assembly.

Ported from bideez `src/mastra/matching/corpus.ts` + `src/mastra/chunking.ts`.

The rule inherited from bideez: do NOT slice documents into fixed char windows.
That severs facts across boundaries and throws away the provenance anchors that
citations need. Instead:

  - Walk the document's heading structure and pack whole sections into a unit
    up to a char ceiling. Never split a section across units.
  - A single section larger than the ceiling is the only case that gets hard
    split, with an overlap so a fact spanning the cut survives whole somewhere.
  - Carry a provenance anchor ("Resume.pdf — part 2/3", "p.4") and a heading
    breadcrumb so unit-local extraction inherits its section's meaning.

Every unit gets a stable citation id. Downstream, the model cites units by id
and we resolve those ids back to real source rows — a cited id that does not
resolve is dropped, exactly as bideez's GATHER step does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ~40K chars (~10-12K tokens) is the bideez ceiling. Kept here so a unit always
# fits comfortably inside a worker call alongside its instructions.
CHUNK_CEILING_CHARS = 12_000
DOC_OVERLAP_CHARS = 800

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class EvidenceUnit:
    """One citable piece of the user's knowledge base."""

    id: str
    source_type: str          # "document"
    source_id: str            # document id
    source_anchor: str | None  # e.g. "resume.pdf — part 2/3"
    breadcrumb: str           # nearest heading, or the document title
    text: str


@dataclass
class Document:
    id: str
    title: str
    text: str
    kind: str = "text"        # pdf | docx | md | text
    error: str = ""           # set when parsing failed; excluded from the corpus


@dataclass
class Corpus:
    units: list[EvidenceUnit] = field(default_factory=list)

    @property
    def units_by_id(self) -> dict[str, EvidenceUnit]:
        return {u.id: u for u in self.units}

    @property
    def total_chars(self) -> int:
        return sum(len(u.text) for u in self.units)


def _split_with_overlap(text: str, ceiling: int) -> list[str]:
    """Hard-split an oversize block, overlapping so a fact spanning the cut
    survives whole in at least one part."""
    if len(text) <= ceiling:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + ceiling, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - DOC_OVERLAP_CHARS
    return out


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (breadcrumb, body) sections on headings.

    Plain text with no headings yields a single section — the packer below then
    handles it by size alone, which is the correct degenerate case.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]

    out: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        out.append(("", preamble))
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start() : end].strip()
        if body:
            out.append((heading, body))
    return out


def build_units(docs: list[Document], ceiling: int = CHUNK_CEILING_CHARS) -> list[EvidenceUnit]:
    """Render documents into a flat list of citable evidence units.

    Sections are packed up to the ceiling and never split; only a single
    oversize section is hard-split, with overlap.
    """
    units: list[EvidenceUnit] = []

    for doc in docs:
        if doc.error or not doc.text.strip():
            continue

        parts: list[tuple[str, str]] = []  # (breadcrumb, text)
        buf = ""
        buf_crumb = ""

        def flush() -> None:
            nonlocal buf, buf_crumb
            if buf.strip():
                parts.append((buf_crumb or doc.title, buf.strip()))
            buf = ""

        for crumb, body in _sections(doc.text):
            if len(body) > ceiling:
                # Oversize single section — the one case we hard-split.
                flush()
                for slice_ in _split_with_overlap(body, ceiling):
                    parts.append((crumb or doc.title, slice_))
                continue
            if buf and len(buf) + len(body) > ceiling:
                flush()
            if not buf:
                buf_crumb = crumb
            buf += ("\n\n" if buf else "") + body

        flush()

        total = len(parts)
        for i, (crumb, text) in enumerate(parts):
            anchor = doc.title if total == 1 else f"{doc.title} — part {i + 1}/{total}"
            units.append(
                EvidenceUnit(
                    id=f"doc:{doc.id}#{i}",
                    source_type="document",
                    source_id=doc.id,
                    source_anchor=anchor,
                    breadcrumb=crumb or doc.title,
                    text=text,
                )
            )

    return units


def assemble(docs: list[Document]) -> Corpus:
    return Corpus(units=build_units(docs))
