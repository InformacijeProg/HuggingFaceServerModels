"""openpyxl mechanics: inspect, create, copy and edit .xlsx/.xlsm workbooks.

Everything here is plain Python working on a real workbook object. No function here
takes instructions from text: ``ops.py`` is the only caller, and it calls these with
parameters it has already validated.

Two openpyxl facts shape the whole module:

* openpyxl does not evaluate formulas. ``data_only=True`` returns the value Excel
  cached when it last saved the file, which is ``None`` in a workbook only hfserve
  has ever written. Reading therefore prefers cached values and falls back to the
  formula text, and aggregates report how many formula cells they had to skip.
* A workbook loaded with ``data_only=True`` has lost its formulas, so saving it would
  replace them with constants. Mutating plans are always loaded with ``data_only=False``
  (enforced in ``ops.apply_plan``).
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.utils.cell import coordinate_from_string, range_boundaries
from openpyxl.worksheet.worksheet import Worksheet

from . import config
from .errors import FileError

CELL_RE = re.compile(r"[A-Za-z]{1,3}[0-9]{1,7}$")
RANGE_RE = re.compile(r"[A-Za-z]{1,3}[0-9]{1,7}:[A-Za-z]{1,3}[0-9]{1,7}$")
COLS_RE = re.compile(r"([A-Za-z]{1,3})(?::([A-Za-z]{1,3}))?$")
ROWS_RE = re.compile(r"([0-9]{1,7})(?::([0-9]{1,7}))?$")
NUMBER_RE = re.compile(r"-?[0-9]+([.][0-9]+)?$")

COLORS = {
    "yellow": "FFFF00", "red": "FF0000", "green": "00B050", "blue": "0070C0",
    "orange": "FFC000", "grey": "D9D9D9", "gray": "D9D9D9", "white": "FFFFFF",
    "black": "000000", "lightgreen": "C6EFCE", "lightred": "FFC7CE",
    "lightblue": "DDEBF7", "lightyellow": "FFEB9C",
}
CHART_TYPES = {"bar": BarChart, "column": BarChart, "line": LineChart, "pie": PieChart}
AGGREGATES = ("sum", "avg", "min", "max", "count", "count_empty")


# --------------------------------------------------------------------- paths
def check_path(path: str | Path, *, must_exist: bool = True) -> Path:
    """Validate an Excel path: right extension, exists, not absurdly large."""
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".xls":
        raise FileError(f"{p.name}: the old binary .xls format is not supported - "
                        f"open it in Excel and save it as .xlsx")
    if suffix not in config.EXCEL_SUFFIXES:
        raise FileError(f"{p.name}: not an Excel file (expected "
                        f"{', '.join(config.EXCEL_SUFFIXES)})")
    if must_exist:
        if not p.is_file():
            raise FileError(f"no such file: {p}")
        size = p.stat().st_size
        if size > config.MAX_UPLOAD_BYTES:
            raise FileError(f"{p.name} is {size / 1048576:.1f} MB, over the "
                            f"{config.MAX_UPLOAD_BYTES // 1048576} MB limit")
    return p


def upload(path: str | Path) -> Path:
    """Copy a user-chosen workbook into files/in/ and return the copy.

    The original is never opened for writing again: later steps read this copy and
    write into files/out/, so no plan - good or bad - can damage the user's file.
    """
    src = check_path(path)
    config.ensure_dirs()
    dst = config.unique_path(config.IN_DIR / config.safe_name(src.name))
    shutil.copy2(src, dst)
    return dst


def copy_workbook(source: str | Path, target: str | Path | None = None) -> Path:
    """Byte-for-byte copy: keeps macros, charts and pivot tables openpyxl cannot rewrite."""
    src = check_path(source)
    dst = Path(target).expanduser() if target else config.out_path(src.name)
    if dst.resolve() == src.resolve():
        raise FileError("source and target are the same file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# --------------------------------------------------------------- open / save
def open_workbook(path: str | Path, *, data_only: bool = False,
                  read_only: bool = False) -> Workbook:
    p = check_path(path)
    try:
        return load_workbook(p, data_only=data_only, read_only=read_only,
                             keep_vba=p.suffix.lower() in config.MACRO_SUFFIXES)
    except FileError:
        raise
    except Exception as e:  # noqa: BLE001 - openpyxl raises many types on a bad file
        raise FileError(f"{p.name}: cannot read this workbook ({type(e).__name__}: {e})") from e


def new_workbook(sheet_name: str | None = None) -> Workbook:
    wb = Workbook()
    if sheet_name:
        wb.active.title = str(sheet_name)[:31]
    return wb


def save_workbook(wb: Workbook, target: str | Path) -> Path:
    dst = Path(target).expanduser()
    if dst.suffix.lower() not in config.EXCEL_SUFFIXES:
        raise FileError(f"{dst.name}: refusing to save a workbook with that extension")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(dst)
    except PermissionError as e:
        raise FileError(f"{dst.name} is open in another program (Excel?) - close it and retry") from e
    except Exception as e:  # noqa: BLE001
        raise FileError(f"cannot write {dst.name} ({type(e).__name__}: {e})") from e
    return dst


def sheet(wb: Workbook, name: str | None = None) -> Worksheet:
    """The named sheet, or the active one when no name was given."""
    if not name:
        return wb.active
    if name in wb.sheetnames:
        return wb[name]
    # Models get capitalisation and stray spaces wrong far more often than the name itself.
    want = str(name).strip().lower()
    for title in wb.sheetnames:
        if title.strip().lower() == want:
            return wb[title]
    raise FileError(f'no sheet named "{name}" (have: {", ".join(wb.sheetnames)})')


# ------------------------------------------------------------------ describe
def cell_text(value: Any, limit: int = config.DIGEST_CELL_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "~"


def value_kind(value: Any) -> str:
    if value is None or value == "":
        return "empty"
    if isinstance(value, str):
        return "formula" if value.startswith("=") else "text"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (dt.datetime, dt.date)):
        return "date"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def _sample(ws: Worksheet, rows: int) -> list[list[Any]]:
    out: list[list[Any]] = []
    for row in ws.iter_rows(values_only=True):
        out.append(list(row[: config.DIGEST_MAX_COLS]))
        if len(out) >= rows:
            break
    return out


def describe(path: str | Path) -> dict:
    """What is in a workbook, as plain data: sheets, size, headers, column kinds, sample rows.

    This - not the file - is what the model is shown. It is context for turning a
    request into a plan; nothing in it is ever executed.
    """
    p = check_path(path)
    wb = open_workbook(p, data_only=True, read_only=True)
    formulas: Workbook | None = None
    try:
        sheets = []
        for ws in wb.worksheets[: config.DIGEST_MAX_SHEETS]:
            rows = _sample(ws, config.DIGEST_SAMPLE_ROWS + 1)
            body = rows[1:]
            if rows and body and not any(v not in (None, "") for r in body for v in r):
                # No cached values (a file only hfserve has written): show formulas instead.
                if formulas is None:
                    formulas = open_workbook(p, data_only=False, read_only=True)
                if ws.title in formulas.sheetnames:
                    rows = _sample(formulas[ws.title], config.DIGEST_SAMPLE_ROWS + 1)
            n_rows = ws.max_row if ws.max_row is not None else len(rows)
            n_cols = ws.max_column if ws.max_column is not None else max((len(r) for r in rows), default=0)
            first = rows[0] if rows else []
            has_header = bool(first) and any(v not in (None, "") for v in first) and all(
                isinstance(v, str) and not v.startswith("=") for v in first if v not in (None, ""))
            headers = [cell_text(v) for v in first] if has_header else []
            kinds = []
            for col in range(len(first)):
                seen = {value_kind(r[col]) for r in rows[1:] if col < len(r)} - {"empty"}
                kinds.append(sorted(seen)[0] if len(seen) == 1 else ("mixed" if seen else "empty"))
            sheets.append({
                "name": ws.title,
                "rows": n_rows,
                "cols": n_cols,
                "headers": headers,
                "column_kinds": kinds,
                "sample": [[cell_text(v) for v in r] for r in (rows[1:] if has_header else rows)],
            })
        return {
            "kind": "excel",
            "file": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sheets": sheets,
            "sheet_names": list(wb.sheetnames),
            "truncated_sheets": max(0, len(wb.sheetnames) - config.DIGEST_MAX_SHEETS),
        }
    finally:
        wb.close()
        if formulas is not None:
            formulas.close()


def digest_text(desc: dict) -> str:
    """The prompt-sized rendering of ``describe``: a few hundred characters, not a dump."""
    lines = [f'file: {desc["file"]}']
    for sh in desc["sheets"]:
        lines.append(f'sheet "{sh["name"]}": {sh["rows"]} rows x {sh["cols"]} cols')
        if sh["headers"]:
            cols = ", ".join(
                f"{get_column_letter(i + 1)}={h}"
                + (f" ({sh['column_kinds'][i]})" if i < len(sh["column_kinds"]) else "")
                for i, h in enumerate(sh["headers"]) if h)
            lines.append(f"  columns: {cols}")
        start = 2 if sh["headers"] else 1
        for n, row in enumerate(sh["sample"]):
            if any(row):
                lines.append(f"  row {start + n}: " + " | ".join(row))
    if desc["truncated_sheets"]:
        lines.append(f'  ... and {desc["truncated_sheets"]} more sheets')
    return "\n".join(lines)


# -------------------------------------------------------------- cell values
def _number(text: str) -> int | float:
    value = float(text)
    return int(value) if value.is_integer() and "." not in text else value


def _looks_numeric(text: str) -> bool:
    """True for "1500" and "-3.5"; false for "007" and "0031", which are codes, not numbers."""
    s = text.strip()
    if not NUMBER_RE.match(s):
        return False
    digits = s.lstrip("-")
    return not (len(digits) > 1 and digits[0] == "0" and not digits.startswith("0."))


def parse_date(value: Any) -> dt.date | dt.datetime:
    if isinstance(value, (dt.datetime, dt.date)):
        return value
    text = str(value).strip()
    try:
        return dt.datetime.fromisoformat(text) if len(text) > 10 else dt.date.fromisoformat(text)
    except ValueError:
        pass
    parts = text.replace("/", ".").split(".")
    parts = [p for p in (q.strip() for q in parts) if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        d, m, y = (int(p) for p in parts)
        if y < 100:
            y += 2000
        try:
            return dt.date(y, m, d)
        except ValueError as e:
            raise FileError(f"not a date: {value!r}") from e
    raise FileError(f"not a date: {value!r} (use 2026-01-31 or 31.01.2026)")


def coerce(value: Any, type_hint: str | None = None) -> Any:
    """JSON value -> cell value. ``type_hint`` wins; otherwise numeric text becomes a number."""
    if isinstance(value, (list, dict)):
        raise FileError(f"cannot write a {type(value).__name__} into a single cell: {value!r}")
    hint = (type_hint or "auto").strip().lower()
    if hint == "auto":
        if isinstance(value, str) and _looks_numeric(value):
            return _number(value.strip())
        return value
    if hint == "text":
        return "" if value is None else str(value)
    if hint in ("number", "int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise FileError(f"not a number: {value!r}")
        try:
            return value if isinstance(value, (int, float)) else _number(str(value).strip().replace(",", "."))
        except ValueError as e:
            raise FileError(f"not a number: {value!r}") from e
    if hint == "date":
        return parse_date(value)
    if hint in ("bool", "boolean"):
        return value if isinstance(value, bool) else str(value).strip().lower() in ("true", "1", "yes", "da")
    if hint == "formula":
        text = str(value).strip()
        return text if text.startswith("=") else "=" + text
    raise FileError(f'unknown value type "{type_hint}" '
                    f"(use auto, text, number, date, bool or formula)")


def set_cell(ws: Worksheet, ref: str, value: Any, type_hint: str | None = None) -> Any:
    ref = check_cell(ref)
    written = coerce(value, type_hint)
    try:
        cell = ws[ref]
        cell.value = written
    except AttributeError as e:  # MergedCell.value is read-only
        raise FileError(f"{ws.title}!{ref} is inside a merged range - unmerge it first") from e
    if isinstance(written, (dt.date, dt.datetime)):
        cell.number_format = config.DATE_FORMAT
    return written


# ------------------------------------------------------- cells, rows, ranges
def check_cell(ref: Any) -> str:
    text = str(ref).strip().replace("$", "").upper()
    if not CELL_RE.match(text):
        raise FileError(f'not a cell reference: "{ref}" (expected e.g. B2)')
    return text


def _header_index(ws: Worksheet, text: str) -> int | None:
    want = text.strip().lower()
    for cell in next(ws.iter_rows(min_row=1, max_row=1), ()):
        if isinstance(cell.value, str) and cell.value.strip().lower() == want:
            return cell.column
    return None


def column_index(ws: Worksheet, column: Any, has_header: bool = True) -> int:
    """Resolve "C", 3 or a header name from row 1 to a 1-based column index.

    Header names are tried first: a model asked to total "Iznos" writes ``"Iznos"``, and
    headers win over the letter reading of the same text. A name is still accepted when
    ``has_header`` is false, because a model that sets that flag by mistake should not
    get an error that lists the very header it just named.
    """
    if column is None or isinstance(column, bool):
        raise FileError(f"not a column: {column!r}")
    if isinstance(column, int):
        if 1 <= column <= 16384:
            return column
        raise FileError(f"column out of range: {column}")
    text = str(column).strip()
    if not text:
        raise FileError("empty column reference")
    if has_header:
        found = _header_index(ws, text)
        if found is not None:
            return found
    if ":" not in text and COLS_RE.match(text):
        return column_index_from_string(text.upper())
    if text.isdigit():
        return int(text)
    found = _header_index(ws, text)
    if found is not None:
        return found
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1), ()) if c.value]
    raise FileError(f'no column "{column}" in "{ws.title}"'
                    + (f' (headers: {", ".join(str(h) for h in headers[:12])})' if headers else ""))


def data_bounds(ws: Worksheet, has_header: bool = True) -> tuple[int, int]:
    """(first data row, last row). Raises when the sheet holds nothing but a header."""
    first = 2 if has_header else 1
    last = ws.max_row or 0
    if last < first:
        raise FileError(f'"{ws.title}" has no data rows')
    return first, last


def resolve_range(ws: Worksheet, ref: Any = None, column: Any = None,
                  has_header: bool = True) -> tuple[int, int, int, int]:
    """(min_col, min_row, max_col, max_row) from a range, cell, column, row or header name."""
    if column is not None and ref is None:
        idx = column_index(ws, column, has_header)
        first, last = data_bounds(ws, has_header)
        return idx, first, idx, last
    text = str(ref or "").strip().replace("$", "")
    if not text:
        raise FileError("no range given")
    if RANGE_RE.match(text.upper()) or CELL_RE.match(text.upper()):
        return range_boundaries(text.upper())
    m = COLS_RE.match(text)
    if m:  # "C" or "C:E": the used height of those columns
        lo = column_index_from_string(m.group(1).upper())
        hi = column_index_from_string((m.group(2) or m.group(1)).upper())
        return min(lo, hi), 1, max(lo, hi), max(ws.max_row or 1, 1)
    m = ROWS_RE.match(text)
    if m:  # "1" or "2:10": the used width of those rows
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        return 1, min(lo, hi), max(ws.max_column or 1, 1), max(lo, hi)
    idx = column_index(ws, text, has_header)  # a header name
    first, last = data_bounds(ws, has_header)
    return idx, first, idx, last


def range_ref(bounds: tuple[int, int, int, int]) -> str:
    c1, r1, c2, r2 = bounds
    return f"{get_column_letter(c1)}{r1}:{get_column_letter(c2)}{r2}"


def read_range(ws: Worksheet, ref: Any = None, column: Any = None,
               has_header: bool = True, max_cells: int = 2000) -> list[list[Any]]:
    c1, r1, c2, r2 = resolve_range(ws, ref, column, has_header)
    if (c2 - c1 + 1) * (r2 - r1 + 1) > max_cells:
        raise FileError(f"{range_ref((c1, r1, c2, r2))} is more than {max_cells} cells - "
                        f"ask for a smaller range")
    return [list(row) for row in ws.iter_rows(min_col=c1, min_row=r1, max_col=c2,
                                              max_row=r2, values_only=True)]


# ------------------------------------------------------------------- writing
def write_rows(ws: Worksheet, start: str, rows: Sequence[Any]) -> int:
    """Write a block of rows with its top-left cell at ``start``. Returns cells written."""
    letter, row0 = coordinate_from_string(check_cell(start))
    col0 = column_index_from_string(letter)
    if not isinstance(rows, (list, tuple)):
        raise FileError('"rows" must be a list of rows')
    if len(rows) > config.MAX_WRITE_ROWS:
        raise FileError(f"{len(rows)} rows is over the {config.MAX_WRITE_ROWS} row limit")
    cells = sum(len(r) if isinstance(r, (list, tuple)) else 1 for r in rows)
    if cells > config.MAX_WRITE_CELLS:
        raise FileError(f"{cells} cells is over the {config.MAX_WRITE_CELLS} cell limit")
    written = 0
    for dr, row in enumerate(rows):
        values = row if isinstance(row, (list, tuple)) else [row]
        for dc, value in enumerate(values):
            if value is None:
                continue
            set_cell(ws, f"{get_column_letter(col0 + dc)}{row0 + dr}", value)
            written += 1
    return written


def append_row(ws: Worksheet, values: Sequence[Any]) -> int:
    if not isinstance(values, (list, tuple)):
        values = [values]
    row = (ws.max_row or 0) + 1
    if row == 2 and all(c.value is None for c in next(ws.iter_rows(max_row=1), ())):
        row = 1  # a brand new, empty sheet
    write_rows(ws, f"A{row}", [list(values)])
    return row


def add_column(ws: Worksheet, header: str, values: Sequence[Any] | None = None,
               formula: str | None = None, at: Any = None, number_format: str | None = None,
               has_header: bool = True) -> tuple[str, int]:
    """New column with a header and either explicit values or a per-row formula template.

    ``formula="=C{row}*0.25"`` is expanded by Python for every data row, so the sheet
    gets real per-row formulas without the model having to count rows.
    """
    if at is not None:
        idx = column_index(ws, at, has_header)
        ws.insert_cols(idx)
    else:
        idx = (ws.max_column or 0) + 1
    letter = get_column_letter(idx)
    first, last = (2, ws.max_row or 1) if has_header else (1, ws.max_row or 1)
    if has_header:
        set_cell(ws, f"{letter}1", header, "text")
    filled = 0
    if values is not None:
        if not isinstance(values, (list, tuple)):
            raise FileError('"values" must be a list')
        for n, value in enumerate(values):
            set_cell(ws, f"{letter}{first + n}", value)
            filled += 1
    elif formula:
        template = formula if formula.startswith("=") else "=" + formula
        for row in range(first, last + 1):
            set_cell(ws, f"{letter}{row}", template.replace("{row}", str(row)), "formula")
            filled += 1
    if number_format and filled:
        for row in range(first, first + filled):
            ws[f"{letter}{row}"].number_format = number_format
    return letter, filled


def total_row(ws: Worksheet, columns: Sequence[Any], label: str = "Ukupno",
              as_formula: bool = True, has_header: bool = True) -> tuple[int, list[str]]:
    """Append a bold totals row.

    ``as_formula`` (the default) writes =SUM(...), which keeps the sheet live in Excel;
    the computed alternative writes the number Python worked out instead.
    """
    first, last = data_bounds(ws, has_header)
    row = last + 1
    indices = [column_index(ws, c, has_header) for c in columns]
    if label and 1 not in indices:
        set_cell(ws, f"A{row}", label, "text")
        ws[f"A{row}"].font = Font(bold=True)
    done = []
    for idx in indices:
        letter = get_column_letter(idx)
        if as_formula:
            set_cell(ws, f"{letter}{row}", f"=SUM({letter}{first}:{letter}{last})", "formula")
        else:
            total, _, _ = aggregate(ws, "sum", ref=f"{letter}{first}:{letter}{last}")
            set_cell(ws, f"{letter}{row}", total, "number")
        ws[f"{letter}{row}"].font = Font(bold=True)
        done.append(letter)
    return row, done


# ----------------------------------------------------------------- analysing
def aggregate(ws: Worksheet, func: str, ref: Any = None, column: Any = None,
              has_header: bool = True) -> tuple[Any, int, int]:
    """sum/avg/min/max/count over a range.

    Returns (value, formula cells skipped, empty cells). Both counts are worth showing:
    a workbook read for its values shows an uncomputed formula as an empty cell, so a
    sum that ignored three cells is a sum the user should know about.
    """
    name = str(func).strip().lower()
    if name in ("average", "mean"):
        name = "avg"
    if name not in AGGREGATES:
        raise FileError(f'unknown function "{func}" (use {", ".join(AGGREGATES)})')
    bounds = resolve_range(ws, ref, column, has_header)
    c1, r1, c2, r2 = bounds
    numbers: list[float] = []
    empty = skipped = filled = 0
    for row in ws.iter_rows(min_col=c1, min_row=r1, max_col=c2, max_row=r2, values_only=True):
        for value in row:
            if value is None or value == "":
                empty += 1
            elif isinstance(value, str) and value.startswith("="):
                skipped += 1  # a formula openpyxl cannot evaluate
            elif isinstance(value, bool):
                filled += 1
            elif isinstance(value, (int, float)):
                numbers.append(float(value))
                filled += 1
            else:
                filled += 1
    if name == "count":
        return filled, skipped, empty
    if name == "count_empty":
        return empty, skipped, 0
    if not numbers:
        raise FileError(f"no numbers in {range_ref(bounds)}"
                        + (f" ({skipped} formula cells have no cached value - open the "
                           f"file in Excel once and save it)" if skipped else ""))
    if name == "sum":
        value = sum(numbers)
    elif name == "avg":
        value = sum(numbers) / len(numbers)
    elif name == "min":
        value = min(numbers)
    else:
        value = max(numbers)
    return (int(value) if float(value).is_integer() else round(value, 10)), skipped, empty


def _match(value: Any, equals: Any, contains: Any, empty: bool, match_case: bool) -> bool:
    if empty:
        return value is None or str(value).strip() == ""
    if equals is not None:
        if isinstance(equals, str) and isinstance(value, str):
            return (equals == value) if match_case else (equals.strip().lower() == value.strip().lower())
        if isinstance(equals, (int, float)) and isinstance(value, (int, float)):
            return float(equals) == float(value)
        return cell_text(value, 200) == cell_text(equals, 200)
    if contains is not None:
        haystack, needle = cell_text(value, 10_000), str(contains)
        return needle in haystack if match_case else needle.lower() in haystack.lower()
    raise FileError("give equals, contains or empty")


def find_rows(ws: Worksheet, column: Any, equals: Any = None, contains: Any = None,
              empty: bool = False, match_case: bool = False, has_header: bool = True,
              limit: int = 50) -> list[tuple[int, list[Any]]]:
    idx = column_index(ws, column, has_header)
    first, last = data_bounds(ws, has_header)
    hits: list[tuple[int, list[Any]]] = []
    for n, row in enumerate(ws.iter_rows(min_row=first, max_row=last, values_only=True)):
        if idx - 1 < len(row) and _match(row[idx - 1], equals, contains, empty, match_case):
            hits.append((first + n, list(row)))
            if len(hits) >= limit:
                break
    return hits


# ----------------------------------------------------------------- reshaping
def _sort_key(value: Any):
    """Numbers, then dates, then text, then blanks: what a person expects from a sort."""
    if value is None or value == "":
        return (3, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, (int, float)):
        return (0, float(value), "")
    if isinstance(value, (dt.datetime, dt.date)):
        stamp = dt.datetime.combine(value, dt.time()) if not isinstance(value, dt.datetime) else value
        return (0, stamp.timestamp(), "")
    return (2, 0.0, str(value).strip().lower())


def sort_rows(ws: Worksheet, column: Any, descending: bool = False,
              has_header: bool = True) -> int:
    """Sort the data rows by one column.

    Values move, per-cell formatting does not: openpyxl styles belong to cells rather
    than to rows, so colours stay where they are while the text moves.
    """
    idx = column_index(ws, column, has_header)
    first, last = data_bounds(ws, has_header)
    width = ws.max_column or 1
    rows = [list(r) for r in ws.iter_rows(min_row=first, max_row=last, max_col=width,
                                          values_only=True)]
    rows.sort(key=lambda r: _sort_key(r[idx - 1] if idx - 1 < len(r) else None),
              reverse=descending)
    for dr, row in enumerate(rows):
        for dc in range(width):
            ws.cell(row=first + dr, column=dc + 1).value = row[dc] if dc < len(row) else None
    return len(rows)


def delete_rows_where(ws: Worksheet, column: Any, equals: Any = None, contains: Any = None,
                      empty: bool = False, match_case: bool = False,
                      has_header: bool = True) -> int:
    idx = column_index(ws, column, has_header)
    first, last = data_bounds(ws, has_header)
    doomed = [first + n for n, row in enumerate(
        ws.iter_rows(min_row=first, max_row=last, values_only=True))
        if idx - 1 < len(row) and _match(row[idx - 1], equals, contains, empty, match_case)]
    for row in reversed(doomed):  # bottom up, so the remaining indices stay valid
        ws.delete_rows(row)
    return len(doomed)


def find_replace(ws: Worksheet, find: str, replace: str, match_case: bool = False) -> tuple[int, int]:
    """Replace text in value cells. Formula cells are left alone and counted instead."""
    if not find:
        raise FileError('"find" must not be empty')
    pattern = re.compile(re.escape(str(find)), 0 if match_case else re.IGNORECASE)
    new = str(replace)
    changed = formulas = 0
    for row in ws.iter_rows():
        for cell in row:
            value = cell.value
            if not isinstance(value, str) or not pattern.search(value):
                continue
            if value.startswith("="):
                formulas += 1  # rewriting a formula by text search is how spreadsheets break
                continue
            cell.value = pattern.sub(lambda _m: new, value)
            changed += 1
    return changed, formulas


# ----------------------------------------------------------------- cosmetics
def color(value: Any) -> str:
    text = str(value).strip().lstrip("#")
    text = COLORS.get(text.lower().replace(" ", ""), text)
    if not re.fullmatch(r"[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8}", text):
        raise FileError(f"not a colour: {value!r} (use RRGGBB or a name like "
                        f"{', '.join(sorted(COLORS)[:5])})")
    return text.upper() if len(text) == 8 else "FF" + text.upper()


def format_range(ws: Worksheet, ref: Any = None, column: Any = None, bold: bool | None = None,
                 italic: bool | None = None, size: float | None = None, font_color: Any = None,
                 fill: Any = None, number_format: str | None = None, align: str | None = None,
                 wrap: bool | None = None, has_header: bool = True) -> tuple[str, int]:
    bounds = resolve_range(ws, ref, column, has_header)
    c1, r1, c2, r2 = bounds
    if align is not None and str(align).lower() not in ("left", "center", "centre", "right"):
        raise FileError(f'align must be left, center or right, not "{align}"')
    patt = PatternFill(start_color=color(fill), end_color=color(fill), fill_type="solid") if fill else None
    font_change = any(v is not None for v in (bold, italic, size, font_color))
    touched = 0
    for row in ws.iter_rows(min_col=c1, min_row=r1, max_col=c2, max_row=r2):
        for cell in row:
            if font_change:
                old = cell.font
                cell.font = Font(name=old.name,
                                 size=size if size is not None else old.size,
                                 bold=old.bold if bold is None else bool(bold),
                                 italic=old.italic if italic is None else bool(italic),
                                 color=color(font_color) if font_color is not None else old.color)
            if patt is not None:
                cell.fill = patt
            if number_format:
                cell.number_format = number_format
            if align is not None or wrap is not None:
                old_align = cell.alignment
                horizontal = old_align.horizontal
                if align is not None:
                    text = str(align).lower()
                    horizontal = "center" if text in ("center", "centre") else text
                cell.alignment = Alignment(horizontal=horizontal, vertical=old_align.vertical,
                                           wrap_text=old_align.wrap_text if wrap is None else bool(wrap))
            touched += 1
    return range_ref(bounds), touched


def column_width(ws: Worksheet, column: Any, width: float, has_header: bool = True) -> str:
    letter = get_column_letter(column_index(ws, column, has_header))
    ws.column_dimensions[letter].width = max(1.0, min(float(width), 255.0))
    return letter


def autofit(ws: Worksheet, max_width: float = 60, max_rows: int = 2000) -> int:
    """Width per column from the longest sampled value (openpyxl has no real autofit)."""
    widest: dict[int, int] = {}
    for n, row in enumerate(ws.iter_rows(values_only=True)):
        if n >= max_rows:
            break
        for col, value in enumerate(row, start=1):
            length = len(cell_text(value, 200))
            if length > widest.get(col, 0):
                widest[col] = length
    for col, length in widest.items():
        ws.column_dimensions[get_column_letter(col)].width = max(8.0, min(length + 2.0, float(max_width)))
    return len(widest)


def add_chart(ws: Worksheet, kind: str, data: str, categories: str | None = None,
              title: str | None = None, anchor: str | None = None,
              titles_from_data: bool = True) -> str:
    name = str(kind).strip().lower()
    if name not in CHART_TYPES:
        raise FileError(f'unknown chart "{kind}" (use {", ".join(sorted(CHART_TYPES))})')
    c1, r1, c2, r2 = resolve_range(ws, data, has_header=False)
    chart = CHART_TYPES[name]()
    if name == "column":
        chart.type = "col"
    chart.add_data(Reference(ws, min_col=c1, min_row=r1, max_col=c2, max_row=r2),
                   titles_from_data=bool(titles_from_data))
    if categories:
        k1, p1, k2, p2 = resolve_range(ws, categories, has_header=False)
        chart.set_categories(Reference(ws, min_col=k1, min_row=p1, max_col=k2, max_row=p2))
    if title:
        chart.title = str(title)
    where = check_cell(anchor) if anchor else f"{get_column_letter((ws.max_column or 1) + 2)}2"
    ws.add_chart(chart, where)
    return where


# ----------------------------------------------------------------------- csv
def export_csv(ws: Worksheet, target: str | Path, delimiter: str = ";") -> tuple[Path, int]:
    """Values only, UTF-8 with a BOM and ; as separator: what Croatian Excel opens cleanly."""
    dst = Path(target).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with dst.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        for row in ws.iter_rows(values_only=True):
            writer.writerow(["" if v is None else cell_text(v, 10_000) for v in row])
            rows += 1
    return dst, rows


DECIMAL_COMMA_RE = re.compile(r"-?[0-9]+,[0-9]+$")


def import_csv(ws: Worksheet, source: str | Path, start: str = "A1",
               delimiter: str | None = None, decimal_comma: bool | None = None) -> int:
    """Read a csv into a sheet, delimiter sniffed from the first line when not given.

    A semicolon-separated file is a Croatian Excel export, where "1,50" is one and a
    half, so comma decimals are converted to numbers for those files only - in a
    comma-separated file the same text is a thousands separator or two fields.
    """
    src = Path(source).expanduser()
    if not src.is_file():
        raise FileError(f"no such file: {src}")
    if src.stat().st_size > config.MAX_UPLOAD_BYTES:
        raise FileError(f"{src.name} is too large")
    lines = src.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    if delimiter is None:
        head = lines[0] if lines else ""
        delimiter = max((";", ",", "\t"), key=head.count) if head else ","
    if decimal_comma is None:
        decimal_comma = delimiter == ";"

    def value_of(text: str):
        if decimal_comma and DECIMAL_COMMA_RE.match(text.strip()):
            return _number(text.strip().replace(",", "."))
        return coerce(text)

    rows = [[value_of(value) for value in row] for row in csv.reader(lines, delimiter=delimiter)]
    return write_rows(ws, start, rows)
