"""python-docx mechanics: read, create and edit .docx documents.

A document is a stream of blocks, not a grid, so the contract here is "append a block"
or "replace this text", never "edit paragraph 7" - a positional edit breaks as soon as
anything above it shifts. Everything is plain Python working on a Document object;
``ops.py`` is the only caller and validates every argument first.

python-docx writes the modern .docx format only. The old binary .doc cannot be read or
written at all, and macro-enabled .docm would lose its macros on save, so both are refused.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

from . import config
from .errors import FileError

ALIGNMENTS = {
    "left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
    "centre": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}
BULLET_STYLES = {1: "List Bullet", 2: "List Bullet 2", 3: "List Bullet 3"}
NUMBER_STYLES = {1: "List Number", 2: "List Number 2", 3: "List Number 3"}
MAX_TABLE_CELLS = 20_000


# --------------------------------------------------------------------- paths
def check_path(path: str | Path, *, must_exist: bool = True) -> Path:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".doc":
        raise FileError(f"{p.name}: the old binary .doc format is not supported - "
                        f"open it in Word and save it as .docx")
    if suffix in (".docm", ".dotm"):
        raise FileError(f"{p.name}: macro-enabled Word files are not supported "
                        f"(the macros would be lost) - save it as .docx")
    if suffix not in config.WORD_SUFFIXES:
        raise FileError(f"{p.name}: not a Word file (expected "
                        f"{', '.join(config.WORD_SUFFIXES)})")
    if must_exist:
        if not p.is_file():
            raise FileError(f"no such file: {p}")
        size = p.stat().st_size
        if size > config.MAX_UPLOAD_BYTES:
            raise FileError(f"{p.name} is {size / 1048576:.1f} MB, over the "
                            f"{config.MAX_UPLOAD_BYTES // 1048576} MB limit")
    return p


def open_document(path: str | Path) -> Document:
    p = check_path(path)
    try:
        return Document(str(p))
    except FileError:
        raise
    except Exception as e:  # noqa: BLE001 - python-docx raises several types on bad files
        raise FileError(f"{p.name}: cannot read this document "
                        f"({type(e).__name__}: {e})") from e


def new_document(template: str | Path | None = None) -> Document:
    """A new document, optionally from a .docx whose styles, header and footer are kept."""
    if template is None:
        return Document()
    return Document(str(check_path(template)))


def save_document(doc: Document, target: str | Path) -> Path:
    dst = Path(target).expanduser()
    if dst.suffix.lower() not in config.WORD_SUFFIXES:
        raise FileError(f"{dst.name}: refusing to save a document with that extension")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(str(dst))
    except PermissionError as e:
        raise FileError(f"{dst.name} is open in another program (Word?) - "
                        f"close it and retry") from e
    except Exception as e:  # noqa: BLE001
        raise FileError(f"cannot write {dst.name} ({type(e).__name__}: {e})") from e
    return dst


# ------------------------------------------------------------------- reading
def iter_blocks(doc: Document) -> Iterator[Paragraph | Table]:
    """Paragraphs and tables in the order they appear (python-docx lists them apart)."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def heading_level(paragraph: Paragraph) -> int | None:
    name = (paragraph.style.name or "") if paragraph.style is not None else ""
    if name == "Title":
        return 0
    m = re.fullmatch(r"Heading (\d)", name)
    return int(m.group(1)) if m else None


def count_words(doc: Document) -> int:
    words = sum(len(p.text.split()) for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            words += sum(len(cell.text.split()) for cell in row.cells)
    return words


def read_text(doc: Document, limit: int = 4000) -> str:
    out: list[str] = []
    total = 0
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        out.append(text)
        total += len(text)
        if total >= limit:
            out.append("...")
            break
    return "\n".join(out)


def outline(doc: Document) -> list[dict]:
    found = []
    for paragraph in doc.paragraphs:
        level = heading_level(paragraph)
        if level is not None and paragraph.text.strip():
            found.append({"level": level, "text": paragraph.text.strip()})
    return found


def describe(path: str | Path) -> dict:
    """What is in a document, as plain data: outline, sizes, tables, opening paragraphs."""
    p = check_path(path)
    doc = open_document(p)
    paragraphs = [q.text.strip() for q in doc.paragraphs if q.text.strip()]
    tables = [{"rows": len(t.rows), "cols": len(t.columns),
               "header": [c.text.strip()[: config.DIGEST_CELL_CHARS] for c in t.rows[0].cells]
               if t.rows else []}
              for t in doc.tables[:8]]
    props = doc.core_properties
    return {
        "kind": "word",
        "file": p.name,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "paragraphs": len(paragraphs),
        "words": count_words(doc),
        "tables": tables,
        "outline": outline(doc)[:40],
        "sample": paragraphs[: config.DIGEST_SAMPLE_ROWS],
        "title": (props.title or "") if props is not None else "",
        "author": (props.author or "") if props is not None else "",
    }


def digest_text(desc: dict) -> str:
    """The prompt-sized rendering of ``describe``."""
    lines = [f'file: {desc["file"]} (Word)',
             f'{desc["paragraphs"]} paragraphs, {desc["words"]} words, '
             f'{len(desc["tables"])} tables']
    if desc["title"]:
        lines.append(f'title: {desc["title"]}')
    if desc["outline"]:
        lines.append("outline:")
        for item in desc["outline"][:12]:
            lines.append(f'  {"  " * max(0, item["level"] - 1)}H{item["level"]} {item["text"]}')
    for n, table in enumerate(desc["tables"][:3], start=1):
        head = " | ".join(h for h in table["header"] if h)
        lines.append(f'table {n}: {table["rows"]}x{table["cols"]}' + (f" ({head})" if head else ""))
    if not desc["outline"] and desc["sample"]:
        lines.append("first paragraphs:")
        for text in desc["sample"][:3]:
            lines.append(f"  {text[:120]}")
    return "\n".join(lines)


# ------------------------------------------------------------------- writing
def _style_or_none(doc: Document, name: str | None):
    """A style by name, or None when this template has no such style."""
    if not name:
        return None
    try:
        return doc.styles[name]
    except KeyError:
        return None


def add_heading(doc: Document, text: str, level: int = 1) -> str:
    level = max(0, min(int(level), 9))
    style = "Title" if level == 0 else f"Heading {level}"
    if _style_or_none(doc, style) is None:
        # A template without heading styles still gets a visibly bigger, bold line.
        run = doc.add_paragraph().add_run(str(text))
        run.bold = True
        run.font.size = Pt(max(12, 20 - 2 * level))
    else:
        doc.add_paragraph(str(text), style=style)
    return str(text)


def add_paragraph(doc: Document, text: str = "", style: str | None = None,
                  bold: bool | None = None, italic: bool | None = None,
                  align: str | None = None, size: float | None = None) -> str:
    if style and _style_or_none(doc, style) is None:
        raise FileError(f'no paragraph style "{style}" in this document')
    paragraph = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    run = paragraph.add_run(str(text))
    if bold is not None:
        run.bold = bool(bold)
    if italic is not None:
        run.italic = bool(italic)
    if size is not None:
        run.font.size = Pt(float(size))
    if align is not None:
        key = str(align).strip().lower()
        if key not in ALIGNMENTS:
            raise FileError(f'align must be one of {", ".join(sorted(ALIGNMENTS))}')
        paragraph.alignment = ALIGNMENTS[key]
    return str(text)


def add_bullets(doc: Document, items: Sequence[Any], numbered: bool = False,
                level: int = 1) -> int:
    """Bullet or numbered list. Items are strings, or {"text", "level"} for sub-levels."""
    if not items:
        raise FileError('"items" must not be empty')
    styles = NUMBER_STYLES if numbered else BULLET_STYLES
    written = 0
    for item in items:
        text, item_level = str(item), int(level or 1)
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            item_level = int(item.get("level", level) or level)
        item_level = max(1, min(item_level, 3))
        style = styles.get(item_level, styles[1])
        if _style_or_none(doc, style) is None:
            doc.add_paragraph(f'{"    " * (item_level - 1)}- {text}')
        else:
            doc.add_paragraph(text, style=style)
        written += 1
    return written


def add_table(doc: Document, rows: Sequence[Sequence[Any]], header: bool = True,
              style: str | None = "Table Grid") -> tuple[int, int]:
    if not rows or not isinstance(rows, (list, tuple)):
        raise FileError('"rows" must be a non-empty list of rows')
    body = [r if isinstance(r, (list, tuple)) else [r] for r in rows]
    cols = max(len(r) for r in body)
    if len(body) * cols > MAX_TABLE_CELLS:
        raise FileError(f"{len(body) * cols} table cells is over the {MAX_TABLE_CELLS} limit")
    table = doc.add_table(rows=len(body), cols=cols)
    if style and _style_or_none(doc, style) is not None:
        table.style = style
    for r, row in enumerate(body):
        for c in range(cols):
            value = row[c] if c < len(row) else ""
            cell = table.cell(r, c)
            cell.text = "" if value is None else str(value)
            if header and r == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
    return len(body), cols


def add_page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def set_properties(doc: Document, title: str | None = None, author: str | None = None,
                   subject: str | None = None) -> dict:
    props = doc.core_properties
    changed = {}
    for name, value in (("title", title), ("author", author), ("subject", subject)):
        if value is not None:
            setattr(props, name, str(value))
            changed[name] = str(value)
    if not changed:
        raise FileError("give title, author or subject")
    props.modified = dt.datetime.now()
    return changed


# ------------------------------------------------------------------- editing
def _replace_in_paragraph(paragraph: Paragraph, pattern: re.Pattern, replacement: str) -> int:
    """Replace inside one paragraph, keeping run formatting where the match allows it.

    A match inside a single run keeps that run's formatting. A match spanning several
    runs cannot keep all of them, so the paragraph collapses to its first run's
    formatting - which is why this is for words and names, not for rebuilding layout.
    """
    if not pattern.search(paragraph.text):
        return 0
    hits = 0
    for run in paragraph.runs:
        if pattern.search(run.text):
            run.text, n = pattern.subn(replacement, run.text)
            hits += n
    if hits:
        return hits
    text, n = pattern.subn(replacement, paragraph.text)
    if n:
        for run in list(paragraph.runs)[1:]:
            run._element.getparent().remove(run._element)
        if paragraph.runs:
            paragraph.runs[0].text = text
        else:
            paragraph.add_run(text)
    return n


def replace_text(doc: Document, find: str, replace: str, match_case: bool = False) -> int:
    """Replace text in body, tables, headers and footers. Returns the number of matches."""
    if not find:
        raise FileError('"find" must not be empty')
    pattern = re.compile(re.escape(str(find)), 0 if match_case else re.IGNORECASE)
    new = str(replace)
    total = 0
    for paragraph in doc.paragraphs:
        total += _replace_in_paragraph(paragraph, pattern, new)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    total += _replace_in_paragraph(paragraph, pattern, new)
    for section in doc.sections:
        for part in (section.header, section.footer):
            for paragraph in part.paragraphs:
                total += _replace_in_paragraph(paragraph, pattern, new)
    return total


def delete_paragraphs_where(doc: Document, contains: str, match_case: bool = False) -> int:
    if not contains:
        raise FileError('"contains" must not be empty')
    needle = str(contains) if match_case else str(contains).lower()
    removed = 0
    for paragraph in list(doc.paragraphs):
        text = paragraph.text if match_case else paragraph.text.lower()
        if needle in text:
            paragraph._element.getparent().remove(paragraph._element)
            removed += 1
    return removed
