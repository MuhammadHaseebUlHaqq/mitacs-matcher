"""
Document ingestion — PDF / DOCX / Markdown / plain text → `Document`.

Parsing happens locally with ordinary libraries — no model call, no external
service, nothing to pay for. Output contract: plain text plus a title, with page
breaks kept as markdown-style separators so `corpus.build_units` can still find
the document's structure.

A hosted parser (for scanned PDFs, which pypdf cannot read) would slot in by
replacing `parse_pdf` — nothing else depends on how the bytes became text.
"""

from __future__ import annotations

import io
import re
import uuid
from html import unescape

from .corpus import Document

MAX_BYTES = 25 * 1024 * 1024


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def parse_pdf(data: bytes, title: str) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError:
        return Document(id=_new_id(), title=title, text="", kind="pdf",
                        error="pypdf is not installed — run: pip install -r requirements.txt")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for n, page in enumerate(reader.pages, start=1):
            body = (page.extract_text() or "").strip()
            if body:
                # A page heading keeps provenance visible to the chunker and to
                # the model reading the unit.
                pages.append(f"## p.{n}\n\n{body}")
        text = "\n\n".join(pages)
        if not text.strip():
            return Document(id=_new_id(), title=title, text="", kind="pdf",
                            error="no extractable text (likely a scanned PDF — OCR required)")
        return Document(id=_new_id(), title=title, text=text, kind="pdf")
    except Exception as e:  # noqa: BLE001 — surface any parser failure to the user
        return Document(id=_new_id(), title=title, text="", kind="pdf", error=f"{type(e).__name__}: {e}")


def parse_docx(data: bytes, title: str) -> Document:
    try:
        import docx  # python-docx
    except ImportError:
        return Document(id=_new_id(), title=title, text="", kind="docx",
                        error="python-docx is not installed — run: pip install -r requirements.txt")
    try:
        d = docx.Document(io.BytesIO(data))
        lines: list[str] = []
        for p in d.paragraphs:
            t = p.text.strip()
            if not t:
                continue
            # Promote Word heading styles to markdown so chunking sees sections.
            style = (p.style.name or "").lower() if p.style else ""
            if style.startswith("heading"):
                depth = "".join(c for c in style if c.isdigit()) or "1"
                lines.append(f"{'#' * min(int(depth), 6)} {t}")
            else:
                lines.append(t)
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        text = "\n\n".join(lines)
        if not text.strip():
            return Document(id=_new_id(), title=title, text="", kind="docx", error="document is empty")
        return Document(id=_new_id(), title=title, text=text, kind="docx")
    except Exception as e:  # noqa: BLE001
        return Document(id=_new_id(), title=title, text="", kind="docx", error=f"{type(e).__name__}: {e}")


def parse_text(data: bytes, title: str, kind: str = "text") -> Document:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return Document(id=_new_id(), title=title, text="", kind=kind, error="could not decode as text")
    if not text.strip():
        return Document(id=_new_id(), title=title, text="", kind=kind, error="file is empty")
    return Document(id=_new_id(), title=title, text=text, kind=kind)


def parse_csv(data: bytes, title: str) -> Document:
    """Render a CSV/TSV as markdown-style rows. The model reads a table the same
    way it reads prose, so flattening to labelled rows is enough."""
    import csv as _csv

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            raw = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return Document(id=_new_id(), title=title, text="", kind="csv", error="could not decode as text")

    try:
        dialect = _csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except _csv.Error:
        dialect = _csv.excel
    rows = list(_csv.reader(raw.splitlines(), dialect))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return Document(id=_new_id(), title=title, text="", kind="csv", error="no rows found")

    header, *body = rows
    lines = [f"# {title}", ""]
    for i, row in enumerate(body, start=1):
        pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row) if v.strip()]
        if pairs:
            lines.append(f"## Row {i}\n" + "\n".join(pairs))
    return Document(id=_new_id(), title=title, text="\n\n".join(lines), kind="csv")


def parse_xlsx(data: bytes, title: str) -> Document:
    try:
        from openpyxl import load_workbook
    except ImportError:
        return Document(id=_new_id(), title=title, text="", kind="xlsx",
                        error="openpyxl is not installed — run: pip install -r requirements.txt")
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        lines = [f"# {title}"]
        for ws in wb.worksheets:
            rows = [[("" if c is None else str(c)).strip() for c in r] for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            lines.append(f"\n## Sheet: {ws.title}")
            header, *body = rows
            for i, row in enumerate(body, start=1):
                pairs = [f"{h}: {v}" for h, v in zip(header, row) if v and h]
                if pairs:
                    lines.append(f"\n### Row {i}\n" + "\n".join(pairs))
        wb.close()
        text = "\n".join(lines)
        if len(text.strip()) <= len(title) + 4:
            return Document(id=_new_id(), title=title, text="", kind="xlsx", error="workbook has no data rows")
        return Document(id=_new_id(), title=title, text=text, kind="xlsx")
    except Exception as e:  # noqa: BLE001
        return Document(id=_new_id(), title=title, text="", kind="xlsx", error=f"{type(e).__name__}: {e}")


def parse_html(data: bytes, title: str) -> Document:
    """Strip tags. Good enough for a saved web page or an HTML export."""
    doc = parse_text(data, title, kind="html")
    if doc.error:
        return doc
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", doc.text)
    body = re.sub(r"(?i)<(h[1-6])[^>]*>", lambda m: "\n\n" + "#" * int(m.group(1)[1]) + " ", body)
    body = re.sub(r"(?i)<(br|/p|/div|/li|/tr)[^>]*>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = unescape(body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        return Document(id=_new_id(), title=title, text="", kind="html", error="no readable text in the HTML")
    return Document(id=_new_id(), title=title, text=body, kind="html")


# Extensions that are already plain text — source files, configs, notes.
TEXT_EXTS = (
    ".txt", ".text", ".log", ".rst", ".org", ".tex",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".sql",
)


def ingest(filename: str, data: bytes) -> Document:
    """Dispatch on extension. Always returns a Document — parse failures come
    back with `.error` set rather than raising, so one bad file in an upload
    never kills the batch (honest degradation, same as a failed chunk).

    Anything not recognised is attempted as UTF-8 text, so an unlisted but
    text-shaped file still works instead of being refused.
    """
    name = (filename or "untitled").strip()
    lower = name.lower()

    if len(data) > MAX_BYTES:
        return Document(id=_new_id(), title=name, text="", error=f"file exceeds {MAX_BYTES // (1024*1024)}MB limit")
    if not data:
        return Document(id=_new_id(), title=name, text="", error="file is empty")

    if lower.endswith(".pdf"):
        return parse_pdf(data, name)
    if lower.endswith((".docx", ".doc")):
        return parse_docx(data, name)
    if lower.endswith((".xlsx", ".xlsm")):
        return parse_xlsx(data, name)
    if lower.endswith((".csv", ".tsv")):
        return parse_csv(data, name)
    if lower.endswith((".html", ".htm")):
        return parse_html(data, name)
    if lower.endswith((".md", ".markdown", ".mdx")):
        return parse_text(data, name, kind="md")
    if lower.endswith(TEXT_EXTS):
        return parse_text(data, name, kind="text")

    # Unknown extension: try text, and say so clearly if it turns out binary.
    doc = parse_text(data, name, kind="text")
    if doc.error or "\x00" in doc.text[:2000]:
        return Document(id=_new_id(), title=name, text="", error="unsupported or binary file type")
    return doc


def from_pasted_text(title: str, text: str) -> Document:
    """A pasted note — same contract as an uploaded file."""
    title = (title or "Pasted note").strip()
    if not text.strip():
        return Document(id=_new_id(), title=title, text="", kind="text", error="pasted text is empty")
    return Document(id=_new_id(), title=title, text=text, kind="pasted")
