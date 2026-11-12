"""PDF mechanics: pypdf to read and rearrange, reportlab to render new pages.

A PDF is not an editable document - it is a fixed rendering - so the two halves are
deliberately separate:

* **rearranging** an existing PDF (merge, keep/delete/rotate pages, metadata) is pypdf
  work on a writer seeded from the source, which is lossless.
* **new content** is rendered by reportlab from the same block list Word uses (headings,
  paragraphs, bullets, tables) and appended as real pages.

Croatian text forces one detail: reportlab's built-in Helvetica is WinAnsi-encoded and has
no č, ć or đ, so a TrueType font covering Latin Extended-A is registered from the system
before anything is rendered. Without it those words come out silently mangled.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Sequence

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import config
from .errors import FileError

PAGE_RE = re.compile(r"\s*([0-9]{1,5})\s*(?:-\s*([0-9]{1,5}))?\s*$")
ALIGNMENTS = {"left": TA_LEFT, "center": TA_CENTER, "centre": TA_CENTER,
              "right": TA_RIGHT, "justify": TA_JUSTIFY}
BLOCK_KINDS = ("heading", "para", "paragraph", "bullets", "numbers", "table", "page_break")
MAX_PAGES = 5000
MAX_TABLE_CELLS = 5000

# Fonts tried in order; the first one present covers Croatian diacritics.
FONT_CANDIDATES = (
    ("Calibri", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf"),
    ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)
_font: tuple[str, str] | None = None  # (regular, bold), registered once


def fonts() -> tuple[str, str]:
    """(regular, bold) font names that can render Croatian, registered on first use."""
    global _font
    if _font is not None:
        return _font
    for name, regular, bold in FONT_CANDIDATES:
        if Path(regular).is_file():
            try:
                pdfmetrics.registerFont(TTFont(name, regular))
                bold_name = name
                if Path(bold).is_file():
                    bold_name = f"{name}-Bold"
                    pdfmetrics.registerFont(TTFont(bold_name, bold))
                _font = (name, bold_name)
                return _font
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    # Last resort: the built-in font. Latin-1 only, so Croatian loses c-caron and d-stroke.
    _font = ("Helvetica", "Helvetica-Bold")
    return _font


# --------------------------------------------------------------------- paths
def check_path(path: str | Path, *, must_exist: bool = True) -> Path:
    p = Path(path).expanduser()
    if p.suffix.lower() not in config.PDF_SUFFIXES:
        raise FileError(f"{p.name}: not a PDF file")
    if must_exist:
        if not p.is_file():
            raise FileError(f"no such file: {p}")
        size = p.stat().st_size
        if size > config.MAX_UPLOAD_BYTES:
            raise FileError(f"{p.name} is {size / 1048576:.1f} MB, over the "
                            f"{config.MAX_UPLOAD_BYTES // 1048576} MB limit")
    return p


def open_reader(path: str | Path) -> PdfReader:
    p = check_path(path)
    try:
        reader = PdfReader(str(p))
    except Exception as e:  # noqa: BLE001
        raise FileError(f"{p.name}: cannot read this PDF ({type(e).__name__}: {e})") from e
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many PDFs are "encrypted" with an empty owner password
        except Exception as e:  # noqa: BLE001
            raise FileError(f"{p.name} is password protected") from e
        if not reader.pages:
            raise FileError(f"{p.name} is password protected")
    return reader


def new_writer(source: str | Path | None = None) -> PdfWriter:
    """A writer holding the source's pages, or an empty one for a PDF built from scratch."""
    if source is None:
        return PdfWriter()
    writer = PdfWriter()
    writer.append(open_reader(source))
    return writer


def save_writer(writer: PdfWriter, target: str | Path) -> Path:
    dst = Path(target).expanduser()
    if dst.suffix.lower() not in config.PDF_SUFFIXES:
        raise FileError(f"{dst.name}: refusing to save a PDF with that extension")
    if not len(writer.pages):
        raise FileError("the PDF would have no pages")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst.open("wb") as fh:
            writer.write(fh)
    except PermissionError as e:
        raise FileError(f"{dst.name} is open in another program - close it and retry") from e
    except Exception as e:  # noqa: BLE001
        raise FileError(f"cannot write {dst.name} ({type(e).__name__}: {e})") from e
    return dst


# ------------------------------------------------------------------- reading
def page_numbers(spec: Any, total: int) -> list[int]:
    """"1-3,7", [1,2] or 4 -> [1,2,3,7]: 1-based and validated against the page count."""
    if spec is None or isinstance(spec, bool):
        raise FileError('give pages as "1-3,7"')
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
    pages: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise FileError(f"not a page number: {item!r}")
        if isinstance(item, (int, float)):
            lo = hi = int(item)
        else:
            m = PAGE_RE.match(str(item))
            if not m:
                raise FileError(f'not a page or range: "{item}" (use 1-3,7)')
            lo = int(m.group(1))
            hi = int(m.group(2) or m.group(1))
        if lo < 1 or hi < lo:
            raise FileError(f"page range {lo}-{hi} makes no sense")
        if hi > total:
            raise FileError(f"the PDF has {total} pages, so page {hi} does not exist")
        pages.extend(range(lo, hi + 1))
    if not pages:
        raise FileError("no pages given")
    return sorted(set(pages))


def extract_text(reader: PdfReader, pages: Any = None, limit: int = 6000) -> str:
    wanted = (page_numbers(pages, len(reader.pages)) if pages is not None
              else range(1, len(reader.pages) + 1))
    out: list[str] = []
    total = 0
    for number in wanted:
        try:
            text = (reader.pages[number - 1].extract_text() or "").strip()
        except Exception as e:  # noqa: BLE001 - a broken page must not kill the read
            text = f"[page {number}: {type(e).__name__}]"
        if not text:
            continue
        out.append(f"-- page {number} --\n{text}")
        total += len(text)
        if total >= limit:
            out.append("...")
            break
    return "\n".join(out)


def describe(path: str | Path) -> dict:
    """What is in a PDF: pages, metadata and the first lines of the opening pages.

    A PDF with no extractable text is a scan, which is worth knowing before a model is
    asked anything about its contents.
    """
    p = check_path(path)
    reader = open_reader(p)
    meta = reader.metadata or {}
    sample = []
    for page in reader.pages[:3]:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        sample.append(lines[:4])
    box = reader.pages[0].mediabox if reader.pages else None
    return {
        "kind": "pdf",
        "file": p.name,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "pages": len(reader.pages),
        "encrypted": bool(reader.is_encrypted),
        "title": str(meta.get("/Title", "") or ""),
        "author": str(meta.get("/Author", "") or ""),
        "page_size": (f"{float(box.width) / 72 * 25.4:.0f}x{float(box.height) / 72 * 25.4:.0f} mm"
                      if box is not None else "?"),
        "has_text": any(any(lines) for lines in sample),
        "sample": sample,
    }


def digest_text(desc: dict) -> str:
    lines = [f'file: {desc["file"]} (PDF)',
             f'{desc["pages"]} pages, {desc["page_size"]}'
             + ("" if desc["has_text"] else " - no extractable text (a scan?)")]
    if desc["title"]:
        lines.append(f'title: {desc["title"]}')
    for n, page in enumerate(desc["sample"], start=1):
        if page:
            lines.append(f"page {n}: " + " / ".join(text[:90] for text in page[:3]))
    return "\n".join(lines)


# ----------------------------------------------------------------- rendering
def _styles(base_font: str, bold_font: str):
    sheet = getSampleStyleSheet()
    body = ParagraphStyle("hfBody", parent=sheet["BodyText"], fontName=base_font,
                          fontSize=11, leading=15, spaceAfter=6)
    heads = {level: ParagraphStyle(f"hfH{level}", parent=sheet["Heading1"], fontName=bold_font,
                                   fontSize=max(12, 20 - 2 * level),
                                   leading=max(15, 24 - 2 * level),
                                   spaceBefore=10 if level > 1 else 0, spaceAfter=6)
             for level in range(0, 5)}
    return body, heads


def render_blocks(blocks: Sequence[Any], *, title: str | None = None,
                  author: str | None = None, orientation: str = "portrait") -> bytes:
    """Render a block list to PDF bytes. Blocks use the same vocabulary as Word."""
    if not blocks or not isinstance(blocks, (list, tuple)):
        raise FileError('"blocks" must be a non-empty list')
    size = landscape(A4) if str(orientation).strip().lower().startswith("land") else A4
    base_font, bold_font = fonts()
    body, heads = _styles(base_font, bold_font)
    flow: list[Any] = []
    for n, block in enumerate(blocks, start=1):
        if not isinstance(block, dict):
            raise FileError(f'block {n}: must be an object with a "block" name')
        name = str(block.get("block") or block.get("type") or "").strip().lower()
        if name not in BLOCK_KINDS:
            raise FileError(f'block {n}: unknown block "{name}" (use {", ".join(BLOCK_KINDS)})')
        if name == "page_break":
            flow.append(PageBreak())
        elif name == "heading":
            level = max(0, min(int(block.get("level", 1) or 1), 4))
            flow.append(Paragraph(str(block.get("text", "")), heads[level]))
        elif name in ("para", "paragraph"):
            style = body
            align = block.get("align")
            if align is not None:
                key = str(align).strip().lower()
                if key not in ALIGNMENTS:
                    raise FileError(f"block {n}: align must be one of "
                                    f'{", ".join(sorted(ALIGNMENTS))}')
                style = ParagraphStyle(f"hfBody{n}", parent=body, alignment=ALIGNMENTS[key])
            flow.append(Paragraph(str(block.get("text", "")), style))
        elif name in ("bullets", "numbers"):
            items = block.get("items") or []
            if not isinstance(items, (list, tuple)) or not items:
                raise FileError(f'block {n}: "items" must be a non-empty list')
            flow.append(ListFlowable(
                [ListItem(Paragraph(str(item), body), leftIndent=12) for item in items],
                bulletType="1" if name == "numbers" else "bullet",
                bulletFontName=base_font, start="1" if name == "numbers" else None))
            flow.append(Spacer(1, 6))
        else:
            rows = block.get("rows") or []
            if not isinstance(rows, (list, tuple)) or not rows:
                raise FileError(f'block {n}: "rows" must be a non-empty list of rows')
            header = bool(block.get("header", True))
            data = [[Paragraph(str("" if cell is None else cell), body)
                     for cell in (row if isinstance(row, (list, tuple)) else [row])]
                    for row in rows]
            if sum(len(row) for row in data) > MAX_TABLE_CELLS:
                raise FileError(f"block {n}: too many table cells")
            table = Table(data, hAlign="LEFT", repeatRows=1 if header else 0)
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0),
                 colors.HexColor("#DDEBF7") if header else colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flow.append(table)
            flow.append(Spacer(1, 8))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=size, title=title or "", author=author or "",
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    try:
        doc.build(flow)
    except Exception as e:  # noqa: BLE001 - a bad flow is a message, not a traceback
        raise FileError(f"cannot render this PDF ({type(e).__name__}: {e})") from e
    return buffer.getvalue()


# ------------------------------------------------------------------- editing
def append_pdf(writer: PdfWriter, source: str | Path, pages: Any = None) -> int:
    reader = open_reader(source)
    if len(writer.pages) + len(reader.pages) > MAX_PAGES:
        raise FileError(f"the result would be over {MAX_PAGES} pages")
    if pages is None:
        writer.append(reader)
        return len(reader.pages)
    wanted = page_numbers(pages, len(reader.pages))
    for number in wanted:
        writer.add_page(reader.pages[number - 1])
    return len(wanted)


def append_bytes(writer: PdfWriter, data: bytes) -> int:
    reader = PdfReader(io.BytesIO(data))
    writer.append(reader)
    return len(reader.pages)


def keep_pages(writer: PdfWriter, pages: Any) -> int:
    """Reduce the writer to the given pages, in the order given."""
    total = len(writer.pages)
    wanted = page_numbers(pages, total)
    kept = [writer.pages[number - 1] for number in wanted]
    for number in range(total - 1, -1, -1):
        del writer.pages[number]
    for page in kept:
        writer.add_page(page)
    return len(kept)


def delete_pages(writer: PdfWriter, pages: Any) -> int:
    total = len(writer.pages)
    doomed = page_numbers(pages, total)
    if len(doomed) >= total:
        raise FileError("that would delete every page")
    for number in sorted(doomed, reverse=True):
        del writer.pages[number - 1]
    return len(doomed)


def rotate_pages(writer: PdfWriter, pages: Any, degrees: int = 90) -> int:
    turn = int(degrees)
    if turn % 90:
        raise FileError("rotation must be a multiple of 90 degrees")
    wanted = page_numbers(pages, len(writer.pages))
    for number in wanted:
        writer.pages[number - 1].rotate(turn)
    return len(wanted)


def set_metadata(writer: PdfWriter, title: str | None = None, author: str | None = None,
                 subject: str | None = None) -> dict:
    changed = {}
    for key, value in (("/Title", title), ("/Author", author), ("/Subject", subject)):
        if value is not None:
            changed[key] = str(value)
    if not changed:
        raise FileError("give title, author or subject")
    writer.add_metadata(changed)
    return changed
