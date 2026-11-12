"""python-pptx mechanics: read, create and edit .pptx decks.

Layout is where models fail, so the model never gets a coordinate. It names a *kind* of
slide (title, bullets, section, table, chart) and supplies the words; this module picks a
matching layout from the template by inspecting its placeholders, fills them, shrinks or
splits content that will not fit, and places every shape itself.

That is the whole reason a deck goes through a template: placeholders carry the theme's
fonts, colours and positions, so filling them cannot produce the overlapping text boxes
that freehand coordinates do.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.util import Emu, Inches, Pt

from . import config
from .errors import FileError

# What the model may ask for. Anything else is an error with this list in it.
SLIDE_KINDS = ("title", "bullets", "section", "two", "blank")
CHART_TYPES = {
    "bar": XL_CHART_TYPE.BAR_CLUSTERED,
    "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "line": XL_CHART_TYPE.LINE_MARKERS,
    "pie": XL_CHART_TYPE.PIE,
    "doughnut": XL_CHART_TYPE.DOUGHNUT,
}
TITLE_TYPES = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
BODY_TYPES = (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.SUBTITLE)

# Readability rules Python enforces, because the model will not.
MAX_BULLETS = 7          # more than this becomes a continuation slide
MAX_BULLET_CHARS = 180   # a longer bullet is trimmed
MAX_SLIDES = 100
CONTINUED = " (nastavak)"
WIDESCREEN = (Inches(13.333), Inches(7.5))


# --------------------------------------------------------------------- paths
def check_path(path: str | Path, *, must_exist: bool = True) -> Path:
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".ppt":
        raise FileError(f"{p.name}: the old binary .ppt format is not supported - "
                        f"open it in PowerPoint and save it as .pptx")
    if suffix in (".pptm", ".potm"):
        raise FileError(f"{p.name}: macro-enabled PowerPoint files are not supported "
                        f"(the macros would be lost) - save it as .pptx")
    if suffix not in config.SLIDES_SUFFIXES:
        raise FileError(f"{p.name}: not a PowerPoint file (expected "
                        f"{', '.join(config.SLIDES_SUFFIXES)})")
    if must_exist:
        if not p.is_file():
            raise FileError(f"no such file: {p}")
        size = p.stat().st_size
        if size > config.MAX_UPLOAD_BYTES:
            raise FileError(f"{p.name} is {size / 1048576:.1f} MB, over the "
                            f"{config.MAX_UPLOAD_BYTES // 1048576} MB limit")
    return p


def open_presentation(path: str | Path) -> Presentation:
    p = check_path(path)
    try:
        return Presentation(str(p))
    except FileError:
        raise
    except Exception as e:  # noqa: BLE001
        raise FileError(f"{p.name}: cannot read this presentation "
                        f"({type(e).__name__}: {e})") from e


def new_presentation(template: str | Path | None = None) -> Presentation:
    """A new deck. Without a template, python-pptx's 4:3 default is widened to 16:9.

    A template's own slides are kept and new ones appended, which is what a title page
    or a closing slide in a template is for.
    """
    if template is not None:
        return Presentation(str(check_path(template)))
    prs = Presentation()
    prs.slide_width, prs.slide_height = WIDESCREEN
    return prs


def save_presentation(prs: Presentation, target: str | Path) -> Path:
    dst = Path(target).expanduser()
    if dst.suffix.lower() not in config.SLIDES_SUFFIXES:
        raise FileError(f"{dst.name}: refusing to save a presentation with that extension")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(dst))
    except PermissionError as e:
        raise FileError(f"{dst.name} is open in another program (PowerPoint?) - "
                        f"close it and retry") from e
    except Exception as e:  # noqa: BLE001
        raise FileError(f"cannot write {dst.name} ({type(e).__name__}: {e})") from e
    return dst


# ------------------------------------------------------------------- layouts
def _types(layout) -> list:
    out = []
    for shape in layout.placeholders:
        try:
            out.append(shape.placeholder_format.type)
        except (AttributeError, ValueError):
            pass
    return out


def pick_layout(prs: Presentation, kind: str):
    """The layout in *this* template that fits the requested kind of slide.

    Chosen by which placeholders a layout exposes rather than by index, because index 1
    is "Title and Content" only in the default template; the name is a tie-breaker.
    """
    want = str(kind or "bullets").strip().lower()
    if want not in SLIDE_KINDS:
        raise FileError(f'unknown slide kind "{kind}" (use {", ".join(SLIDE_KINDS)})')
    scored = []
    for index, layout in enumerate(prs.slide_layouts):
        types = _types(layout)
        titles = sum(1 for t in types if t in TITLE_TYPES)
        bodies = sum(1 for t in types if t in BODY_TYPES)
        name = (layout.name or "").lower()
        if want == "title":
            score = ((PP_PLACEHOLDER.CENTER_TITLE in types) * 4 + titles * 2
                     + (PP_PLACEHOLDER.SUBTITLE in types) * 3 + ("title slide" in name) * 3)
        elif want == "bullets":
            score = (titles * 3 + (bodies == 1) * 4 + ("content" in name) * 2
                     - (bodies > 1) * 3 - (PP_PLACEHOLDER.CENTER_TITLE in types) * 2)
        elif want == "section":
            score = titles * 2 + ("section" in name) * 6 + (bodies == 1)
        elif want == "two":
            score = titles * 2 + (bodies >= 2) * 6 + ("two" in name or "comparison" in name) * 2
        else:  # blank: a layout with nothing on it, or the emptiest one available
            score = 6 - len(types) + ("blank" in name) * 6
        scored.append((score, -len(types), index, layout))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return scored[0][3]


def layout_names(prs: Presentation) -> list[str]:
    return [layout.name or f"layout {n}" for n, layout in enumerate(prs.slide_layouts)]


def _title_shape(slide):
    if slide.shapes.title is not None:
        return slide.shapes.title
    for shape in slide.placeholders:
        try:
            if shape.placeholder_format.type in TITLE_TYPES:
                return shape
        except (AttributeError, ValueError):
            continue
    return None


def _body_shapes(slide) -> list:
    bodies = []
    title = _title_shape(slide)
    for shape in slide.placeholders:
        if title is not None and shape is title:
            continue
        try:
            if shape.placeholder_format.type in BODY_TYPES:
                bodies.append(shape)
        except (AttributeError, ValueError):
            continue
    return bodies


def slide_title(slide) -> str:
    shape = _title_shape(slide)
    return shape.text_frame.text.strip() if shape is not None and shape.has_text_frame else ""


def _last_slide(prs: Presentation):
    return prs.slides[len(prs.slides._sldIdLst) - 1]


# ------------------------------------------------------------------ describe
def describe(path: str | Path) -> dict:
    """What is in a deck, as plain data: per slide the layout, title, bullets and notes."""
    p = check_path(path)
    prs = open_presentation(p)
    slides = []
    for n, slide in enumerate(prs.slides, start=1):
        bullets = []
        for shape in _body_shapes(slide):
            if shape.has_text_frame:
                bullets += [q.text.strip() for q in shape.text_frame.paragraphs if q.text.strip()]
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        slides.append({
            "index": n,
            "layout": slide.slide_layout.name or "?",
            "title": slide_title(slide)[: config.DIGEST_CELL_CHARS * 2],
            "bullets": [b[: config.DIGEST_CELL_CHARS * 2] for b in bullets[:6]],
            "shapes": len(slide.shapes),
            "notes": notes[:120],
        })
    return {
        "kind": "slides",
        "file": p.name,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "slide_count": len(prs.slides._sldIdLst),
        "slides": slides[:30],
        "layouts": layout_names(prs),
        "size": f"{prs.slide_width / 914400:.2f}x{prs.slide_height / 914400:.2f} in",
    }


def digest_text(desc: dict) -> str:
    lines = [f'file: {desc["file"]} (PowerPoint)',
             f'{desc["slide_count"]} slides, {desc["size"]}']
    for slide in desc["slides"][:12]:
        head = f'  slide {slide["index"]} [{slide["layout"]}]: {slide["title"] or "(no title)"}'
        if slide["bullets"]:
            head += f' - {len(slide["bullets"])} bullets: ' + " / ".join(slide["bullets"][:2])
        lines.append(head)
    if desc["slide_count"] > 12:
        lines.append(f'  ... and {desc["slide_count"] - 12} more slides')
    lines.append("layouts available: " + ", ".join(desc["layouts"][:8]))
    return "\n".join(lines)


# -------------------------------------------------------------------- slides
def _fill_bullets(shape, bullets: Sequence[Any]) -> int:
    """Write bullets into a body placeholder, at a font size that suits the count."""
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    size = Pt(20) if len(bullets) <= 4 else (Pt(18) if len(bullets) <= 6 else Pt(16))
    written = 0
    for n, item in enumerate(bullets):
        text, level = str(item), 0
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            level = max(0, min(int(item.get("level", 1) or 1) - 1, 4))
        paragraph = frame.paragraphs[0] if n == 0 else frame.add_paragraph()
        paragraph.text = text.strip()[:MAX_BULLET_CHARS]
        paragraph.level = level
        for run in paragraph.runs:
            run.font.size = size
        written += 1
    return written


def add_slide(prs: Presentation, kind: str = "bullets", title: str | None = None,
              subtitle: str | None = None, bullets: Sequence[Any] | None = None,
              notes: str | None = None) -> list[int]:
    """One slide, or more when the bullets do not fit. Returns the slide numbers written."""
    if len(prs.slides._sldIdLst) >= MAX_SLIDES:
        raise FileError(f"a deck of more than {MAX_SLIDES} slides is almost certainly a mistake")
    items = list(bullets or [])
    chunks = [items[i:i + MAX_BULLETS] for i in range(0, len(items), MAX_BULLETS)] or [[]]
    written = []
    for n, chunk in enumerate(chunks):
        slide = prs.slides.add_slide(pick_layout(prs, kind))
        head = str(title) if title is not None else None
        if head and n:
            head += CONTINUED
        title_shape = _title_shape(slide)
        if head and title_shape is not None:
            title_shape.text_frame.text = head
        bodies = _body_shapes(slide)
        filled = False
        if chunk and bodies:
            _fill_bullets(bodies[0], chunk)
            filled = True
        elif subtitle and bodies:
            bodies[0].text_frame.text = str(subtitle)
            filled = True
        # An untouched placeholder shows "Click to add text" in edit view, so drop it.
        for shape in (bodies[1:] if filled else bodies):
            if shape.has_text_frame and not shape.text_frame.text:
                shape._element.getparent().remove(shape._element)
        if notes is not None:
            slide.notes_slide.notes_text_frame.text = str(notes)
        written.append(len(prs.slides._sldIdLst))
    return written


def _content_box(prs: Presentation, slide) -> tuple[Emu, Emu, Emu, Emu]:
    """Left, top, width, height for a shape that has to sit under the title."""
    title = _title_shape(slide)
    top = Emu(int(title.top + title.height + Inches(0.3))) if title is not None else Inches(1.2)
    margin = Inches(0.7)
    width = Emu(int(prs.slide_width - 2 * margin))
    height = Emu(int(prs.slide_height - top - Inches(0.6)))
    return margin, top, width, height


def _clear_bodies(slide) -> None:
    for shape in _body_shapes(slide):
        shape._element.getparent().remove(shape._element)


def add_table_slide(prs: Presentation, rows: Sequence[Sequence[Any]],
                    title: str | None = None, notes: str | None = None) -> int:
    if not rows or not isinstance(rows, (list, tuple)):
        raise FileError('"rows" must be a non-empty list of rows')
    body = [r if isinstance(r, (list, tuple)) else [r] for r in rows]
    n_rows, n_cols = len(body), max(len(r) for r in body)
    if n_rows > 12 or n_cols > 8:
        raise FileError(f"a {n_rows}x{n_cols} table does not fit on a slide "
                        f"(up to 12 rows and 8 columns)")
    add_slide(prs, "bullets", title=title, notes=notes)
    slide = _last_slide(prs)
    _clear_bodies(slide)  # the body placeholder makes room for the table
    left, top, width, height = _content_box(prs, slide)
    table_height = Emu(min(int(height), int(Inches(0.4)) * n_rows))
    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, table_height).table
    for r, row in enumerate(body):
        for c in range(n_cols):
            value = row[c] if c < len(row) else ""
            cell = table.cell(r, c)
            cell.text = "" if value is None else str(value)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(14)
                    run.font.bold = r == 0
    return len(prs.slides._sldIdLst)


def add_chart_slide(prs: Presentation, kind: str, categories: Sequence[Any],
                    values: Sequence[Any], title: str | None = None,
                    series_name: str = "Serija", notes: str | None = None) -> int:
    name = str(kind).strip().lower()
    if name not in CHART_TYPES:
        raise FileError(f'unknown chart "{kind}" (use {", ".join(sorted(CHART_TYPES))})')
    cats = [str(c) for c in (categories or [])]
    if not cats:
        raise FileError('"categories" must not be empty')
    numbers = []
    for value in values or []:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise FileError(f"chart values must be numbers, got {value!r}")
        try:
            numbers.append(float(str(value).replace(",", ".")))
        except ValueError as e:
            raise FileError(f"chart values must be numbers, got {value!r}") from e
    if len(numbers) != len(cats):
        raise FileError(f"{len(cats)} categories but {len(numbers)} values - they must match")
    add_slide(prs, "bullets", title=title, notes=notes)
    slide = _last_slide(prs)
    _clear_bodies(slide)
    left, top, width, height = _content_box(prs, slide)
    data = CategoryChartData()
    data.categories = cats
    data.add_series(str(series_name), numbers)
    chart = slide.shapes.add_chart(CHART_TYPES[name], left, top, width, height, data).chart
    chart.has_legend = name in ("pie", "doughnut")
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.RIGHT
        chart.legend.include_in_layout = False
    return len(prs.slides._sldIdLst)


def delete_slide(prs: Presentation, index: int) -> str:
    """Remove slide ``index`` (1-based).

    python-pptx has no API for this, so the entry in the slide id list and its
    relationship are dropped by hand - the documented workaround.
    """
    slides = prs.slides
    total = len(slides._sldIdLst)
    if not 1 <= index <= total:
        raise FileError(f"no slide {index} (the deck has {total})")
    title = slide_title(slides[index - 1]) or f"slide {index}"
    entry = list(slides._sldIdLst)[index - 1]
    prs.part.drop_rel(entry.rId)
    slides._sldIdLst.remove(entry)
    return title


def set_notes(prs: Presentation, index: int, text: str) -> int:
    total = len(prs.slides._sldIdLst)
    if not 1 <= index <= total:
        raise FileError(f"no slide {index} (the deck has {total})")
    prs.slides[index - 1].notes_slide.notes_text_frame.text = str(text)
    return index


def replace_text(prs: Presentation, find: str, replace: str, match_case: bool = False) -> int:
    """Replace text in every text frame, run by run, so formatting survives."""
    if not find:
        raise FileError('"find" must not be empty')
    needle = str(find) if match_case else str(find).lower()
    new = str(replace)
    changed = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    text = run.text if match_case else run.text.lower()
                    if needle in text:
                        start = text.index(needle)
                        run.text = run.text[:start] + new + run.text[start + len(needle):]
                        changed += 1
    return changed
