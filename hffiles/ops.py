"""The operation contract: what a plan may ask for, and how a plan is executed.

A *plan* is data, not code::

    {"save_as": "budget-2026.xlsx",
     "ops": [{"op": "sort_rows", "sheet": "Prihodi", "column": "Iznos", "descending": true},
             {"op": "total_row", "columns": ["Iznos"]},
             {"op": "format_range", "range": "1", "bold": true, "fill": "lightblue"}]}

Every operation is looked up in ``REGISTRY``, its parameters are checked by name and
type, and only then does ``excel.py`` touch the workbook. Anything a model invents -
an unknown op, a missing argument, a path like ``../../Windows/x.xlsx`` - becomes a
plain error message before a single byte is written.

Rules that hold for every plan:

* A plan that only reads (``aggregate``, ``read_range``, ``find_rows``, ...) saves nothing.
* A plan that writes never touches the source. The result goes to ``files/out/``,
  unless the source already lives there, in which case it is the working copy of an
  ongoing edit and is updated in place.
* ``strict=True`` (the default) is all-or-nothing: the first failing operation stops
  the plan and nothing is saved.
* Editing an existing workbook rewrites it through openpyxl, which does not carry
  over charts, pivot tables or VBA drawings that were already in the file.
  ``excel.copy_workbook`` is the byte-for-byte copy for files like those.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openpyxl.utils import get_column_letter

from . import config, convert, excel, pdf, slides, word
from .errors import FileError

OP_KEYS = ("op", "action", "name", "type")          # what models call the verb
PARAM_KEYS = ("params", "args", "arguments")        # how they sometimes nest the rest
ALIASES = {
    "worksheet": "sheet", "sheet_name": "sheet", "tab": "sheet",
    "ref": "range", "cell_range": "range", "range_ref": "range",
    "cell_ref": "cell", "coordinate": "cell",
    "column_name": "column", "col": "column",
    "value_type": "type", "data_type": "type",
    "desc": "descending", "reverse": "descending",
    "has_headers": "has_header",
    "colour": "fill", "background": "fill", "bg_color": "fill",
    "text_color": "font_color", "font_colour": "font_color",
}


# --------------------------------------------------------------------- result
@dataclass
class OpResult:
    op: str
    ok: bool
    message: str
    value: Any = None


@dataclass
class PlanResult:
    results: list[OpResult] = field(default_factory=list)
    source: Path | None = None
    target: Path | None = None
    saved: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    pdf: Path | None = None  # set when the plan asked for a PDF export as well

    @property
    def ok(self) -> bool:
        return self.error is None and all(r.ok for r in self.results)

    def lines(self) -> list[str]:
        out = [f"note: {n}" for n in self.notes]
        out += [("" if r.ok else "FAILED ") + r.message for r in self.results]
        if self.error:
            out.append(f"FAILED {self.error}")
        if self.saved and self.target is not None:
            out.append(f"saved: {self.target}")
        if self.pdf is not None:
            out.append(f"PDF: {self.pdf}")
        elif not self.results and not self.error:
            out.append("nothing to do")
        return out

    def summary(self) -> str:
        return "\n".join(self.lines())


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class OpSpec:
    name: str
    run: Callable[[Any, dict], tuple[str, Any]]
    required: tuple[str, ...]
    optional: tuple[str, ...]
    mutates: bool
    help: str
    kind: str = "excel"  # which format this operation belongs to

    def signature(self) -> str:
        parts = [str(k) for k in self.required] + [f"{k}?" for k in self.optional]
        return f"{self.name}({', '.join(parts)})"


REGISTRY: dict[str, OpSpec] = {}


def op(name: str, *, required: tuple[str, ...] = (), optional: tuple[str, ...] = (),
       mutates: bool = True, sheet: bool = True, kind: str = "excel",
       help: str = "") -> Callable:
    """Register an operation. ``sheet`` adds the optional sheet parameter (Excel only).

    Operation names are unique across formats, so a plan's format follows from its
    operations and the model never has to say which one it means.
    """
    def register(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise RuntimeError(f"operation {name} is registered twice")
        REGISTRY[name] = OpSpec(name, fn, required,
                                (("sheet",) if sheet and kind == "excel" else ()) + optional,
                                mutates, help, kind)
        return fn
    return register


def reference(names: tuple[str, ...] | None = None, kind: str | None = None) -> str:
    """The op list as the model sees it: one line per operation, optionally one format."""
    chosen = [REGISTRY[n] for n in (names or tuple(REGISTRY)) if n in REGISTRY]
    if kind is not None:
        chosen = [spec for spec in chosen if spec.kind == kind]
    return "\n".join(f"- {spec.signature()}: {spec.help}" for spec in chosen)


def ops_for(kind: str) -> tuple[str, ...]:
    return tuple(name for name, spec in REGISTRY.items() if spec.kind == kind)


# ------------------------------------------------------------- param helpers
def _missing(key: str) -> FileError:
    return FileError(f'missing "{key}"')


def as_str(params: dict, key: str, default: str | None = None, *,
           required: bool = False) -> str | None:
    if params.get(key) is None:
        if required:
            raise _missing(key)
        return default
    value = params[key]
    if isinstance(value, (dict, list)):
        raise FileError(f'"{key}" must be text, not {type(value).__name__}')
    return str(value)


def as_bool(params: dict, key: str, default: bool | None = None) -> bool | None:
    if params.get(key) is None:
        return default
    value = params[key]
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "da", "on"):
        return True
    if text in ("false", "0", "no", "ne", "off"):
        return False
    raise FileError(f'"{key}" must be true or false, not {value!r}')


def as_number(params: dict, key: str, default: float | None = None, *,
              required: bool = False) -> float | None:
    if params.get(key) is None:
        if required:
            raise _missing(key)
        return default
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise FileError(f'"{key}" must be a number')
    try:
        return float(str(value).replace(",", "."))
    except ValueError as e:
        raise FileError(f'"{key}" must be a number, not {value!r}') from e


def as_int(params: dict, key: str, default: int | None = None, *,
           required: bool = False, minimum: int | None = None) -> int | None:
    value = as_number(params, key, None if default is None else float(default), required=required)
    if value is None:
        return None
    number = int(value)
    if minimum is not None and number < minimum:
        raise FileError(f'"{key}" must be at least {minimum}')
    return number


def as_list(params: dict, key: str, *, required: bool = False) -> list | None:
    if params.get(key) is None:
        if required:
            raise _missing(key)
        return None
    value = params[key]
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (str, int, float, bool)):
        return [value]  # a single value where a list was expected is harmless
    raise FileError(f'"{key}" must be a list')


def _match_args(params: dict) -> dict:
    """equals / contains / empty, shared by the find and delete-where operations."""
    return {
        "equals": params.get("equals"),
        "contains": params.get("contains"),
        "empty": as_bool(params, "empty", False),
        "match_case": as_bool(params, "match_case", False),
        "has_header": as_bool(params, "has_header", True),
    }


def _csv_in_workspace(name: str) -> Path:
    """Find a csv by name in files/in, files/out or the working directory - nowhere else."""
    base = config.safe_name(name, default="data.csv", suffixes=(".csv", ".txt", ".tsv"))
    for folder in (config.IN_DIR, config.OUT_DIR, Path.cwd()):
        candidate = folder / base
        if candidate.is_file():
            return candidate
    raise FileError(f'no csv named "{base}" - put it in {config.IN_DIR}')


# ------------------------------------------------------------- read-only ops
@op("list_sheets", mutates=False, sheet=False, help="names and sizes of every sheet")
def _list_sheets(ctx, p):
    parts = [f"{ws.title} ({ws.max_row}x{ws.max_column})" for ws in ctx.wb.worksheets]
    return "sheets: " + ", ".join(parts), [ws.title for ws in ctx.wb.worksheets]


@op("read_cell", required=("cell",), mutates=False, help="value of one cell")
def _read_cell(ctx, p):
    ws = ctx.sheet(p)
    ref = excel.check_cell(as_str(p, "cell", required=True))
    value = ws[ref].value
    return f"{ws.title}!{ref} = {excel.cell_text(value, 200)!r}", value


@op("read_range", optional=("range", "column", "has_header"), mutates=False,
    help='values of a range ("B2:D10"), a column letter or a column name')
def _read_range(ctx, p):
    ws = ctx.sheet(p)
    rows = excel.read_range(ws, p.get("range"), p.get("column"), as_bool(p, "has_header", True))
    shown = "; ".join(" | ".join(excel.cell_text(v) for v in r) for r in rows[:10])
    more = f" (+{len(rows) - 10} more rows)" if len(rows) > 10 else ""
    return f"{ws.title}: {len(rows)} rows: {shown}{more}", rows


@op("aggregate", required=("func",), optional=("range", "column", "has_header"), mutates=False,
    help="sum / avg / min / max / count over a column or range, computed in Python")
def _aggregate(ctx, p):
    ws = ctx.sheet(p)
    func = as_str(p, "func", required=True)
    value, skipped, empty = excel.aggregate(ws, func, p.get("range"), p.get("column"),
                                            as_bool(p, "has_header", True))
    where = p.get("column") or p.get("range")
    notes = []
    if skipped:
        notes.append(f"{skipped} formula cells skipped")
    if empty:
        # In a workbook read for its values, a formula Excel has never computed is
        # indistinguishable from an empty cell - so say how many were left out.
        notes.append(f"{empty} empty or uncomputed cells ignored")
    note = f" ({'; '.join(notes)})" if notes else ""
    return f"{func.lower()}({ws.title}!{where}) = {value}{note}", value


@op("count_rows", optional=("has_header",), mutates=False, help="how many data rows a sheet has")
def _count_rows(ctx, p):
    ws = ctx.sheet(p)
    has_header = as_bool(p, "has_header", True)
    rows = max(0, (ws.max_row or 0) - (1 if has_header else 0))
    return f"{ws.title}: {rows} data rows", rows


@op("find_rows", required=("column",),
    optional=("equals", "contains", "empty", "match_case", "has_header", "limit"), mutates=False,
    help="rows where a column equals / contains a value (or is empty)")
def _find_rows(ctx, p):
    ws = ctx.sheet(p)
    hits = excel.find_rows(ws, p["column"], limit=as_int(p, "limit", 50, minimum=1),
                           **_match_args(p))
    shown = "; ".join(f"r{n}: " + " | ".join(excel.cell_text(v) for v in row)
                      for n, row in hits[:10])
    return f"{len(hits)} rows match in {ws.title}: {shown}", [n for n, _ in hits]


@op("export_csv", optional=("name", "delimiter"), mutates=False,
    help="write one sheet to a csv file in files/out")
def _export_csv(ctx, p):
    ws = ctx.sheet(p)
    target = config.out_path(as_str(p, "name", f"{ws.title}.csv"), suffixes=(".csv",))
    path, rows = excel.export_csv(ws, target, as_str(p, "delimiter", ";"))
    return f"{rows} rows from {ws.title} -> {path}", str(path)


# ------------------------------------------------------------------ cell ops
@op("set_cell", required=("cell", "value"), optional=("type",),
    help="one cell; type is auto (default), text, number, date, bool or formula")
def _set_cell(ctx, p):
    ws = ctx.sheet(p)
    ref = as_str(p, "cell", required=True)
    written = excel.set_cell(ws, ref, p["value"], as_str(p, "type"))
    return f"{ws.title}!{ref.upper()} = {excel.cell_text(written, 60)!r}", written


@op("set_formula", required=("cell", "formula"), help='a formula, e.g. "=SUM(B2:B20)"')
def _set_formula(ctx, p):
    ws = ctx.sheet(p)
    ref = as_str(p, "cell", required=True)
    written = excel.set_cell(ws, ref, as_str(p, "formula", required=True), "formula")
    return f"{ws.title}!{ref.upper()} = {written}", written


@op("write_rows", required=("rows",), optional=("start",),
    help='a block of rows from a start cell (default A1): rows=[["Ime","Iznos"],["Ana",10]]')
def _write_rows(ctx, p):
    ws = ctx.sheet(p)
    rows = as_list(p, "rows", required=True)
    start = as_str(p, "start", "A1")
    cells = excel.write_rows(ws, start, rows)
    return f"{len(rows)} rows ({cells} cells) written to {ws.title} from {start.upper()}", cells


@op("append_row", required=("values",), help="one row added under the last used row")
def _append_row(ctx, p):
    ws = ctx.sheet(p)
    row = excel.append_row(ws, as_list(p, "values", required=True))
    return f"row {row} appended to {ws.title}", row


@op("add_column", required=("header",),
    optional=("values", "formula", "at", "number_format", "has_header"),
    help='new column; formula="=C{row}*0.25" is expanded for every data row')
def _add_column(ctx, p):
    ws = ctx.sheet(p)
    letter, filled = excel.add_column(ws, as_str(p, "header", required=True),
                                      as_list(p, "values"), as_str(p, "formula"),
                                      p.get("at"), as_str(p, "number_format"),
                                      as_bool(p, "has_header", True))
    # A header with nothing under it is almost never what was asked for, so say so
    # rather than reporting "0 cells" as if the work had been done.
    how = f"{filled} cells" if filled else "header only - no values or formula were given"
    return f'column {letter} "{p["header"]}" added to {ws.title} ({how})', letter


@op("total_row", required=("columns",), optional=("label", "as", "has_header"),
    help='bold totals row; as="formula" (default, =SUM) or as="value" (computed in Python)')
def _total_row(ctx, p):
    ws = ctx.sheet(p)
    how = (as_str(p, "as", "formula") or "formula").lower()
    if how not in ("formula", "value", "values", "number"):
        raise FileError('"as" must be formula or value')
    row, letters = excel.total_row(ws, as_list(p, "columns", required=True),
                                   as_str(p, "label", "Ukupno"), how == "formula",
                                   as_bool(p, "has_header", True))
    return f"totals row {row} in {ws.title} for {', '.join(letters)}", row


# ----------------------------------------------------------------- sheet ops
@op("create_sheet", required=("name",), optional=("index",), sheet=False,
    help="add an empty sheet")
def _create_sheet(ctx, p):
    name = as_str(p, "name", required=True)[:31]
    if name in ctx.wb.sheetnames:
        raise FileError(f'a sheet named "{name}" already exists')
    ctx.wb.create_sheet(name, as_int(p, "index"))
    return f'sheet "{name}" created', name


@op("delete_sheet", help="remove a sheet (not the last one)")
def _delete_sheet(ctx, p):
    ws = ctx.sheet(p)
    if len(ctx.wb.sheetnames) == 1:
        raise FileError("a workbook needs at least one sheet")
    title = ws.title
    ctx.wb.remove(ws)
    return f'sheet "{title}" deleted', title


@op("rename_sheet", required=("to",), help="rename a sheet")
def _rename_sheet(ctx, p):
    ws = ctx.sheet(p)
    new = as_str(p, "to", required=True)[:31]
    old = ws.title
    if new != old and new in ctx.wb.sheetnames:
        raise FileError(f'a sheet named "{new}" already exists')
    ws.title = new
    return f'sheet "{old}" renamed to "{new}"', new


@op("copy_sheet", optional=("to",), help="duplicate a sheet inside the workbook")
def _copy_sheet(ctx, p):
    ws = ctx.sheet(p)
    clone = ctx.wb.copy_worksheet(ws)
    to = as_str(p, "to")
    if to:
        clone.title = to[:31]
    return f'sheet "{ws.title}" copied to "{clone.title}"', clone.title


# ------------------------------------------------------- rows, columns, data
@op("insert_rows", required=("at",), optional=("count",), help="insert empty rows at a row number")
def _insert_rows(ctx, p):
    ws = ctx.sheet(p)
    at, count = as_int(p, "at", required=True, minimum=1), as_int(p, "count", 1, minimum=1)
    ws.insert_rows(at, count)
    return f"{count} rows inserted at {at} in {ws.title}", at


@op("delete_rows", required=("at",), optional=("count",), help="delete rows by row number")
def _delete_rows(ctx, p):
    ws = ctx.sheet(p)
    at, count = as_int(p, "at", required=True, minimum=1), as_int(p, "count", 1, minimum=1)
    ws.delete_rows(at, count)
    return f"{count} rows deleted at {at} in {ws.title}", at


@op("insert_columns", required=("at",), optional=("count", "has_header"),
    help="insert empty columns at a letter or header name")
def _insert_columns(ctx, p):
    ws = ctx.sheet(p)
    at = excel.column_index(ws, p["at"], as_bool(p, "has_header", True))
    count = as_int(p, "count", 1, minimum=1)
    ws.insert_cols(at, count)
    return f"{count} columns inserted at {get_column_letter(at)} in {ws.title}", at


@op("delete_columns", required=("at",), optional=("count", "has_header"),
    help="delete columns by letter or header name")
def _delete_columns(ctx, p):
    ws = ctx.sheet(p)
    at = excel.column_index(ws, p["at"], as_bool(p, "has_header", True))
    count = as_int(p, "count", 1, minimum=1)
    ws.delete_cols(at, count)
    return f"{count} columns deleted at {get_column_letter(at)} in {ws.title}", at


@op("delete_rows_where", required=("column",),
    optional=("equals", "contains", "empty", "match_case", "has_header"),
    help="delete data rows whose column equals / contains a value (or is empty)")
def _delete_rows_where(ctx, p):
    ws = ctx.sheet(p)
    n = excel.delete_rows_where(ws, p["column"], **_match_args(p))
    return f"{n} rows deleted from {ws.title}", n


@op("sort_rows", required=("column",), optional=("descending", "has_header"),
    help="sort the data rows by one column (values move, cell colours stay)")
def _sort_rows(ctx, p):
    ws = ctx.sheet(p)
    descending = as_bool(p, "descending", False)
    n = excel.sort_rows(ws, p["column"], descending, as_bool(p, "has_header", True))
    return (f"{n} rows in {ws.title} sorted by {p['column']} "
            f"({'descending' if descending else 'ascending'})"), n


@op("find_replace", required=("find", "replace"), optional=("match_case",),
    help="replace text in value cells (formulas are left alone)")
def _find_replace(ctx, p):
    ws = ctx.sheet(p)
    changed, formulas = excel.find_replace(ws, as_str(p, "find", required=True),
                                           as_str(p, "replace", "") or "",
                                           as_bool(p, "match_case", False))
    note = f", {formulas} formula cells left alone" if formulas else ""
    return f"{changed} cells changed in {ws.title}{note}", changed


@op("import_csv", required=("path",), optional=("start", "delimiter"),
    help="read a csv from files/in into a sheet")
def _import_csv(ctx, p):
    ws = ctx.sheet(p)
    src = _csv_in_workspace(as_str(p, "path", required=True))
    cells = excel.import_csv(ws, src, as_str(p, "start", "A1"), as_str(p, "delimiter"))
    return f"{src.name} -> {ws.title} ({cells} cells)", cells


# ----------------------------------------------------------------- cosmetics
@op("format_range", optional=("range", "column", "bold", "italic", "size", "font_color",
                              "fill", "number_format", "align", "wrap", "has_header"),
    help='bold / italic / fill / number_format ("#,##0.00") / align on a range, row ("1") or column')
def _format_range(ctx, p):
    ws = ctx.sheet(p)
    ref, touched = excel.format_range(
        ws, p.get("range"), p.get("column"), as_bool(p, "bold"), as_bool(p, "italic"),
        as_number(p, "size"), p.get("font_color"), p.get("fill"),
        as_str(p, "number_format"), as_str(p, "align"), as_bool(p, "wrap"),
        as_bool(p, "has_header", True))
    return f"{ws.title}!{ref}: {touched} cells formatted", ref


@op("column_width", required=("column", "width"), optional=("has_header",),
    help="set one column's width in characters")
def _column_width(ctx, p):
    ws = ctx.sheet(p)
    letter = excel.column_width(ws, p["column"], as_number(p, "width", required=True),
                                as_bool(p, "has_header", True))
    return f"{ws.title}!{letter} width {p['width']}", letter


@op("autofit", optional=("max_width",), help="widen every column to fit its content")
def _autofit(ctx, p):
    ws = ctx.sheet(p)
    n = excel.autofit(ws, as_number(p, "max_width", 60))
    return f"{n} columns in {ws.title} widened", n


@op("freeze_panes", optional=("cell",), help='freeze rows/columns above and left of a cell ("A2")')
def _freeze_panes(ctx, p):
    ws = ctx.sheet(p)
    ref = excel.check_cell(as_str(p, "cell", "A2"))
    ws.freeze_panes = ref
    return f"{ws.title}: panes frozen at {ref}", ref


@op("merge_cells", required=("range",), help="merge a range into one cell")
def _merge_cells(ctx, p):
    ws = ctx.sheet(p)
    ref = excel.range_ref(excel.resolve_range(ws, as_str(p, "range", required=True),
                                              has_header=False))
    ws.merge_cells(ref)
    return f"{ws.title}!{ref} merged", ref


@op("unmerge_cells", required=("range",), help="undo a merge")
def _unmerge_cells(ctx, p):
    ws = ctx.sheet(p)
    ref = excel.range_ref(excel.resolve_range(ws, as_str(p, "range", required=True),
                                              has_header=False))
    try:
        ws.unmerge_cells(ref)
    except (KeyError, ValueError) as e:
        raise FileError(f"{ws.title}!{ref} is not a merged range") from e
    return f"{ws.title}!{ref} unmerged", ref


@op("auto_filter", optional=("range",), help="put Excel's filter arrows on the used range")
def _auto_filter(ctx, p):
    ws = ctx.sheet(p)
    ref = as_str(p, "range") or ws.dimensions
    ws.auto_filter.ref = ref
    return f"{ws.title}: filter on {ref}", ref


@op("add_chart", required=("kind", "data"),
    optional=("categories", "title", "anchor", "titles_from_data"),
    help='chart (bar / column / line / pie): data="B1:B20", categories="A2:A20"')
def _add_chart(ctx, p):
    ws = ctx.sheet(p)
    where = excel.add_chart(ws, as_str(p, "kind", required=True),
                            as_str(p, "data", required=True), as_str(p, "categories"),
                            as_str(p, "title"), as_str(p, "anchor"),
                            as_bool(p, "titles_from_data", True))
    return f"{p['kind']} chart added to {ws.title} at {where}", where


# ------------------------------------------------------------- Word (.docx)
@op("add_heading", required=("text",), optional=("level",), kind="word",
    help="heading; level 1-4 (0 is the document title)")
def _add_heading(ctx, p):
    text = word.add_heading(ctx.doc, as_str(p, "text", required=True), as_int(p, "level", 1))
    return f"heading added: {text[:60]!r}", text


@op("add_paragraph", optional=("text", "style", "bold", "italic", "align", "size"), kind="word",
    help="paragraph of text")
def _add_paragraph(ctx, p):
    text = word.add_paragraph(ctx.doc, as_str(p, "text", "") or "", as_str(p, "style"),
                              as_bool(p, "bold"), as_bool(p, "italic"), as_str(p, "align"),
                              as_number(p, "size"))
    return f"paragraph added ({len(text)} chars)", len(text)


@op("add_bullets", required=("items",), optional=("numbered", "level"), kind="word",
    help="bullet or numbered list; items are strings")
def _add_bullets(ctx, p):
    n = word.add_bullets(ctx.doc, as_list(p, "items", required=True),
                         as_bool(p, "numbered", False), as_int(p, "level", 1))
    return f"{n} list items added", n


@op("add_table", required=("rows",), optional=("header", "style"), kind="word",
    help='table from rows: rows=[["Stavka","Cijena"],["Kruh",1.5]]')
def _add_doc_table(ctx, p):
    rows, cols = word.add_table(ctx.doc, as_list(p, "rows", required=True),
                                as_bool(p, "header", True), as_str(p, "style", "Table Grid"))
    return f"{rows}x{cols} table added", rows


@op("add_page_break", kind="word", help="start a new page")
def _add_page_break(ctx, p):
    word.add_page_break(ctx.doc)
    return "page break added", True


@op("replace_text", required=("find", "replace"), optional=("match_case",), kind="word",
    help="replace text in the body, tables, header and footer")
def _replace_text(ctx, p):
    n = word.replace_text(ctx.doc, as_str(p, "find", required=True),
                          as_str(p, "replace", "") or "", as_bool(p, "match_case", False))
    return f"{n} replacements made", n


@op("delete_paragraphs_where", required=("contains",), optional=("match_case",), kind="word",
    help="delete every paragraph containing a piece of text")
def _delete_paragraphs_where(ctx, p):
    n = word.delete_paragraphs_where(ctx.doc, as_str(p, "contains", required=True),
                                     as_bool(p, "match_case", False))
    return f"{n} paragraphs deleted", n


@op("set_properties", optional=("title", "author", "subject"), kind="word",
    help="document title, author or subject")
def _set_properties(ctx, p):
    changed = word.set_properties(ctx.doc, as_str(p, "title"), as_str(p, "author"),
                                  as_str(p, "subject"))
    return "properties set: " + ", ".join(f"{k}={v!r}" for k, v in changed.items()), changed


@op("read_outline", mutates=False, kind="word", help="the document's headings")
def _read_outline(ctx, p):
    found = word.outline(ctx.doc)
    shown = " / ".join(f'H{i["level"]} {i["text"]}' for i in found[:12])
    return f"{len(found)} headings: {shown}", found


@op("read_document", optional=("limit",), mutates=False, kind="word",
    help="the document's text, up to a character limit")
def _read_document(ctx, p):
    text = word.read_text(ctx.doc, as_int(p, "limit", 4000, minimum=1))
    return f"{len(text)} chars read: {text[:300]!r}", text


@op("count_words", mutates=False, kind="word", help="how many words the document has")
def _count_words(ctx, p):
    n = word.count_words(ctx.doc)
    return f"{n} words", n


# ---------------------------------------------------- PowerPoint (.pptx)
@op("add_slide", optional=("kind", "title", "subtitle", "bullets", "notes"), kind="slides",
    help='slide; kind is title, bullets (default), section, two or blank. Python picks '
         'the layout and splits bullets that do not fit')
def _add_slide(ctx, p):
    kind = as_str(p, "kind", "bullets")
    written = slides.add_slide(ctx.prs, kind, as_str(p, "title"), as_str(p, "subtitle"),
                               as_list(p, "bullets"), as_str(p, "notes"))
    extra = f" (+{len(written) - 1} continuation)" if len(written) > 1 else ""
    return f"slide {written[0]} added [{kind}]{extra}", written


@op("add_table_slide", required=("rows",), optional=("title", "notes"), kind="slides",
    help="slide holding a table, up to 12 rows and 8 columns")
def _add_table_slide(ctx, p):
    n = slides.add_table_slide(ctx.prs, as_list(p, "rows", required=True),
                               as_str(p, "title"), as_str(p, "notes"))
    return f"slide {n} added with a table", n


@op("add_chart_slide", required=("kind", "categories", "values"),
    optional=("title", "series_name", "notes"), kind="slides",
    help='slide holding a chart: kind bar/column/line/pie, categories=["Q1",...], '
         "values=[12,...]")
def _add_chart_slide(ctx, p):
    n = slides.add_chart_slide(ctx.prs, as_str(p, "kind", required=True),
                               as_list(p, "categories", required=True),
                               as_list(p, "values", required=True), as_str(p, "title"),
                               as_str(p, "series_name", "Serija") or "Serija",
                               as_str(p, "notes"))
    return f"slide {n} added with a {p['kind']} chart", n


@op("delete_slide", required=("index",), kind="slides", help="remove slide number N")
def _delete_slide(ctx, p):
    title = slides.delete_slide(ctx.prs, as_int(p, "index", required=True, minimum=1))
    return f"slide deleted: {title!r}", title


@op("set_notes", required=("index", "text"), kind="slides", help="speaker notes for slide N")
def _set_notes(ctx, p):
    n = slides.set_notes(ctx.prs, as_int(p, "index", required=True, minimum=1),
                         as_str(p, "text", required=True))
    return f"notes set on slide {n}", n


@op("replace_slide_text", required=("find", "replace"), optional=("match_case",), kind="slides",
    help="replace text on every slide")
def _replace_slide_text(ctx, p):
    n = slides.replace_text(ctx.prs, as_str(p, "find", required=True),
                            as_str(p, "replace", "") or "", as_bool(p, "match_case", False))
    return f"{n} replacements made", n


@op("list_slides", mutates=False, kind="slides", help="slide numbers and titles")
def _list_slides(ctx, p):
    titles = [slides.slide_title(s) or "(no title)" for s in ctx.prs.slides]
    return f"{len(titles)} slides: " + " / ".join(titles[:12]), titles


# --------------------------------------------------------------- PDF (.pdf)
@op("write_blocks", required=("blocks",), optional=("title", "author", "orientation"),
    kind="pdf",
    help='new pages from blocks: [{"block":"heading","text":"..."},'
         '{"block":"para","text":"..."},{"block":"bullets","items":[...]},'
         '{"block":"table","rows":[[...]]}]')
def _write_blocks(ctx, p):
    data = pdf.render_blocks(as_list(p, "blocks", required=True), title=as_str(p, "title"),
                             author=as_str(p, "author"),
                             orientation=as_str(p, "orientation", "portrait") or "portrait")
    n = pdf.append_bytes(ctx.writer, data)
    return f"{n} pages rendered and added", n


@op("merge_pdf", required=("path",), optional=("pages",), kind="pdf",
    help="append another PDF from files/in (optionally only some of its pages)")
def _merge_pdf(ctx, p):
    name = config.safe_name(as_str(p, "path", required=True), default="dodatak.pdf",
                            suffixes=config.PDF_SUFFIXES)
    for folder in (config.IN_DIR, config.OUT_DIR):
        candidate = folder / name
        if candidate.is_file():
            n = pdf.append_pdf(ctx.writer, candidate, p.get("pages"))
            return f"{n} pages from {name} appended", n
    raise FileError(f'no PDF named "{name}" - put it in {config.IN_DIR}')


@op("keep_pages", required=("pages",), kind="pdf", help='keep only these pages ("1-3,7")')
def _keep_pages(ctx, p):
    n = pdf.keep_pages(ctx.writer, p["pages"])
    return f"{n} pages kept", n


@op("delete_pages", required=("pages",), kind="pdf", help='delete these pages ("2,5-6")')
def _delete_pages(ctx, p):
    n = pdf.delete_pages(ctx.writer, p["pages"])
    return f"{n} pages deleted", n


@op("rotate_pages", required=("pages",), optional=("degrees",), kind="pdf",
    help="rotate pages by a multiple of 90 degrees")
def _rotate_pages(ctx, p):
    degrees = as_int(p, "degrees", 90)
    n = pdf.rotate_pages(ctx.writer, p["pages"], degrees)
    return f"{n} pages rotated by {degrees}", n


@op("set_pdf_metadata", optional=("title", "author", "subject"), kind="pdf",
    help="PDF title, author or subject")
def _set_pdf_metadata(ctx, p):
    changed = pdf.set_metadata(ctx.writer, as_str(p, "title"), as_str(p, "author"),
                               as_str(p, "subject"))
    return "metadata set: " + ", ".join(f"{k}={v!r}" for k, v in changed.items()), changed


@op("read_pdf_text", optional=("pages", "limit"), mutates=False, kind="pdf",
    help="text of the whole PDF or of some pages")
def _read_pdf_text(ctx, p):
    text = pdf.extract_text(ctx.reader, p.get("pages"), as_int(p, "limit", 6000, minimum=1))
    if not text.strip():
        return "no extractable text (a scanned PDF?)", ""
    return f"{len(text)} chars read: {text[:300]!r}", text


@op("count_pages", mutates=False, kind="pdf", help="how many pages the PDF has")
def _count_pages(ctx, p):
    n = len(ctx.writer.pages)
    return f"{n} pages", n


# ------------------------------------------------------------------ planning
@dataclass
class Plan:
    steps: list[tuple[OpSpec, dict]]
    save_as: str | None = None
    sheet: str | None = None
    warnings: list[str] = field(default_factory=list)
    export_pdf: bool = False

    @property
    def mutates(self) -> bool:
        return any(spec.mutates for spec, _ in self.steps)

    @property
    def kind(self) -> str:
        """Which format this plan is for, taken from its operations."""
        kinds = {spec.kind for spec, _ in self.steps}
        if len(kinds) > 1:
            raise FileError("one plan cannot mix "
                            + " and ".join(sorted(config.KIND_LABELS[k] for k in kinds))
                            + " operations")
        return kinds.pop() if kinds else "excel"


def _normalize(raw: Any) -> tuple[list, Any, Any, Any]:
    if isinstance(raw, str):
        raise FileError("the plan is text, not JSON")
    if isinstance(raw, list):
        return raw, None, None, None
    if not isinstance(raw, dict):
        raise FileError(f"a plan must be an object or a list, not {type(raw).__name__}")
    if any(key in raw for key in ("ops", "operations", "steps")):
        ops = raw.get("ops") or raw.get("operations") or raw.get("steps") or []
        if not isinstance(ops, list):
            raise FileError('"ops" must be a list')
        return (ops, raw.get("save_as") or raw.get("filename") or raw.get("file_name"),
                raw.get("sheet"), raw.get("export_pdf") or raw.get("as_pdf"))
    if any(key in raw for key in OP_KEYS):
        return [raw], None, None, None  # a single operation, unwrapped
    raise FileError('a plan needs an "ops" list')


def parse_plan(raw: Any) -> Plan:
    """Validate a plan without touching any file. Raises FileError on anything unknown."""
    ops, save_as, sheet, export_pdf = _normalize(raw)
    if len(ops) > config.MAX_PLAN_OPS:
        raise FileError(f"{len(ops)} operations is over the {config.MAX_PLAN_OPS} limit")
    steps: list[tuple[OpSpec, dict]] = []
    warnings: list[str] = []
    for n, item in enumerate(ops, start=1):
        if not isinstance(item, dict):
            raise FileError(f"op {n}: must be an object, not {type(item).__name__}")
        name = next((str(item[k]) for k in OP_KEYS if isinstance(item.get(k), str)
                     and str(item[k]).strip().lower().replace(" ", "_").replace("-", "_")
                     in REGISTRY), "")
        if not name:  # keep the model's own wording in the error message
            name = next((str(item[k]) for k in OP_KEYS if isinstance(item.get(k), str)), "")
        name = name.strip().lower().replace(" ", "_").replace("-", "_")
        if not name:
            raise FileError(f'op {n}: no "op" name')
        spec = REGISTRY.get(name)
        if spec is None:
            near = difflib.get_close_matches(name, list(REGISTRY), n=3, cutoff=0.6)
            raise FileError(f'op {n}: unknown operation "{name}"'
                            + (f" - did you mean {', '.join(near)}?" if near else ""))
        params: dict = {}
        for key, value in item.items():
            if key in OP_KEYS and isinstance(value, str) and value.strip().lower().replace(
                    " ", "_").replace("-", "_") == name:
                continue  # that was the verb itself
            if key in PARAM_KEYS and isinstance(value, dict):
                for nested, inner in value.items():  # aliases apply in here too
                    params[ALIASES.get(nested, nested)] = inner
                continue
            params[ALIASES.get(key, key)] = value
        allowed = set(spec.required) | set(spec.optional)
        unknown = sorted(k for k in params if k not in allowed)
        if unknown:
            # Small models decorate operations with parameters that do not exist. An extra
            # key is noise rather than a changed instruction, so it is dropped - but said
            # out loud, because a dropped key can also be a request that went unheard.
            for key in unknown:
                params.pop(key)
            warnings.append(f"op {n} ({name}): ignored unknown parameter(s) "
                            f"{', '.join(unknown)}; takes {spec.signature()}")
        missing = [k for k in spec.required if params.get(k) is None]
        if missing:
            raise FileError(f"op {n} ({name}): missing {', '.join(missing)}; "
                            f"takes {spec.signature()}")
        steps.append((spec, params))
    plan = Plan(steps, save_as if isinstance(save_as, str) else None,
                sheet if isinstance(sheet, str) else None, warnings,
                export_pdf=bool(export_pdf))
    plan.kind  # a plan that mixes formats is rejected here, before anything is opened
    return plan


class Context:
    """What an operation handler is allowed to see: the open file, and nothing else.

    One of ``wb`` / ``doc`` / ``prs`` / ``writer`` is set, depending on the plan's format.
    """

    def __init__(self, kind: str, *, wb=None, doc=None, prs=None, writer=None,
                 reader=None) -> None:
        self.kind = kind
        self.wb = wb
        self.doc = doc
        self.prs = prs
        self.writer = writer
        self.reader = reader

    def sheet(self, params: dict):
        return excel.sheet(self.wb, params.get("sheet"))


def resolve_target(source: Path | None, target: str | Path | None, save_as: str | None,
                   kind: str = "excel") -> Path:
    """Where a mutating plan writes.

    The caller's ``target`` wins. Otherwise a new file under files/out, named after the
    plan's ``save_as`` (sanitised) or after the source. A source that already lives in
    files/out is the working copy of an edit in progress and is updated in place, so a
    conversation of small changes does not leave twenty near-identical files behind.
    """
    suffixes = config.KIND_SUFFIXES[kind]
    default = config.DEFAULT_NAMES[kind]
    if target is not None:
        return Path(target).expanduser()
    if source is None:
        return config.out_path(save_as or default, suffixes=suffixes, default=default)
    if save_as:
        return config.out_path(save_as, suffixes=suffixes, default=default)
    if config.is_within(source, config.OUT_DIR):
        return source
    return config.out_path(source.name, suffixes=suffixes, default=default)


def apply_plan(raw: Any, *, source: str | Path | None = None,
               target: str | Path | None = None, strict: bool = True) -> PlanResult:
    """Validate a plan, run it, and save when it changed anything.

    ``source=None`` creates a new workbook. Read-only plans save nothing. With
    ``strict`` the first failure stops the plan and nothing is written.
    """
    src = Path(source).expanduser() if source is not None else None
    result = PlanResult(source=src)
    try:
        plan = parse_plan(raw)
        kind = plan.kind
        if src is not None:
            found = config.kind_or_fail(src)
            if found != kind:
                raise FileError(f"those are {config.KIND_LABELS[kind]} operations, but "
                                f"{src.name} is a {config.KIND_LABELS[found]} file")
    except FileError as e:
        result.error = str(e)
        return result
    result.notes = list(plan.warnings)
    if not plan.steps:
        result.error = "the plan has no operations"
        return result

    mutates = plan.mutates
    if mutates:
        result.target = resolve_target(src, target, plan.save_as, kind)
    try:
        ctx = _open(kind, src, plan, mutates)
    except FileError as e:
        result.error = str(e)
        return result
    try:
        for spec, params in plan.steps:
            try:
                message, value = spec.run(ctx, params)
                result.results.append(OpResult(spec.name, True, message, value))
            except FileError as e:
                result.results.append(OpResult(spec.name, False, f"{spec.name}: {e}"))
                if strict:
                    return result
            except Exception as e:  # noqa: BLE001 - openpyxl surprises stay inside the plan
                result.results.append(OpResult(spec.name, False,
                                               f"{spec.name}: {type(e).__name__}: {e}"))
                if strict:
                    return result
        if mutates and result.target is not None:
            try:
                _save(ctx, result.target)
                result.saved = True
            except FileError as e:
                result.error = str(e)
    finally:
        if ctx.wb is not None and getattr(ctx.wb, "read_only", False):
            ctx.wb.close()
    if result.saved and plan.export_pdf and result.target is not None:
        try:
            result.pdf = convert.to_pdf(result.target)
        except FileError as e:
            result.notes.append(f"PDF export failed: {e}")
    return result


def _open(kind: str, src: Path | None, plan: Plan, mutates: bool) -> Context:
    """Open the source (or start an empty file) in whatever way the format needs."""
    if kind == "excel":
        # data_only=True would drop every formula from a file we are about to rewrite.
        wb = excel.open_workbook(src, data_only=not mutates) if src else \
            excel.new_workbook(plan.sheet)
        return Context(kind, wb=wb)
    if kind == "word":
        return Context(kind, doc=word.open_document(src) if src else word.new_document())
    if kind == "slides":
        return Context(kind, prs=slides.open_presentation(src) if src
                       else slides.new_presentation())
    if src is None and not mutates:
        raise FileError("there is no PDF to read")
    return Context(kind, writer=pdf.new_writer(src),
                   reader=pdf.open_reader(src) if src is not None else None)


def _save(ctx: Context, target: Path) -> Path:
    if ctx.kind == "excel":
        return excel.save_workbook(ctx.wb, target)
    if ctx.kind == "word":
        return word.save_document(ctx.doc, target)
    if ctx.kind == "slides":
        return slides.save_presentation(ctx.prs, target)
    return pdf.save_writer(ctx.writer, target)


def create_workbook(rows_by_sheet: dict[str, list[list[Any]]],
                    target: str | Path | None = None, *, autofit_columns: bool = True,
                    bold_header: bool = True) -> PlanResult:
    """Build a workbook from plain rows: the short path for code that has the data already."""
    steps: list[dict] = []
    for n, (name, rows) in enumerate(rows_by_sheet.items()):
        steps.append({"op": "rename_sheet", "to": name} if n == 0
                     else {"op": "create_sheet", "name": name})
        if rows:
            steps.append({"op": "write_rows", "sheet": name, "start": "A1", "rows": rows})
            if bold_header:
                steps.append({"op": "format_range", "sheet": name, "range": "1", "bold": True})
        if autofit_columns:
            steps.append({"op": "autofit", "sheet": name})
    return apply_plan({"ops": steps}, target=target)
