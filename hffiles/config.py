"""Paths, limits and path hygiene for hffiles (sibling of hfrag/hfzakon, same HOME).

Environment variables:
  HFFILES_HOME   root for the file workspace (default: HFSERVE_HOME/files)

Workspace layout::

    files/in/    uploads, copied here untouched  (the user's original is never opened for writing)
    files/out/   everything hffiles produces     (the only folder a model-supplied name can reach)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from hfserve import config as serve_config

from .errors import FileError

FILES_DIR = Path(os.environ.get("HFFILES_HOME", serve_config.HOME_DIR / "files")).expanduser().resolve()
IN_DIR = FILES_DIR / "in"
OUT_DIR = FILES_DIR / "out"

# openpyxl reads/writes the OOXML formats only. The old binary .xls needs xlrd and
# cannot be written at all, so it is refused with a "save as .xlsx" hint instead.
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xltx", ".xltm")
MACRO_SUFFIXES = (".xlsm", ".xltm")
# python-docx and python-pptx write the modern formats only; the binary .doc/.ppt and the
# macro-enabled .docm/.pptm are refused by word.py / slides.py with a "save as" hint.
WORD_SUFFIXES = (".docx",)
SLIDES_SUFFIXES = (".pptx",)
PDF_SUFFIXES = (".pdf",)

KINDS = ("excel", "word", "slides", "pdf")
KIND_SUFFIXES = {"excel": EXCEL_SUFFIXES, "word": WORD_SUFFIXES,
                 "slides": SLIDES_SUFFIXES, "pdf": PDF_SUFFIXES}
KIND_LABELS = {"excel": "Excel", "word": "Word", "slides": "PowerPoint", "pdf": "PDF"}
DEFAULT_NAMES = {"excel": "rezultat.xlsx", "word": "rezultat.docx",
                 "slides": "rezultat.pptx", "pdf": "rezultat.pdf"}
# Every extension the GUI's attach dialog and the CLI accept.
ALL_SUFFIXES = tuple(s for kind in KINDS for s in KIND_SUFFIXES[kind])

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
# A plan is a handful of operations; a thousand of them is a runaway model, not a request.
MAX_PLAN_OPS = 80
# Guard rails for a single operation (a model that writes 10 million cells is a bug).
MAX_WRITE_ROWS = 100_000
MAX_WRITE_CELLS = 500_000

# How much of a workbook goes into the prompt. Small models drown in long context,
# so this stays deliberately tight: enough to see the shape of the sheet, no more.
DIGEST_SAMPLE_ROWS = 5
DIGEST_MAX_COLS = 24
DIGEST_MAX_SHEETS = 8
DIGEST_CELL_CHARS = 40

DEFAULT_OUT_NAME = "rezultat.xlsx"
DATE_FORMAT = "dd.mm.yyyy"

_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def ensure_dirs() -> None:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str | None, *, default: str = DEFAULT_OUT_NAME,
               suffixes: tuple[str, ...] = EXCEL_SUFFIXES) -> str:
    """Turn anything a model wrote into a plain file name.

    Drops every directory part, so ``../../Windows/x.xlsx`` or ``C:/secret.xlsx``
    become ``x.xlsx`` / ``secret.xlsx`` and stay inside the workspace. Unknown
    extensions get the first allowed one appended rather than being trusted.
    """
    raw = (name or "").strip().replace("\\", "/")
    stem = Path(raw).name  # kills every traversal and drive letter
    stem = _BAD_CHARS.sub("", stem).strip(" .")
    if not stem:
        stem = default
    path = Path(stem)
    if path.suffix.lower() not in suffixes:
        path = path.with_name(path.name + suffixes[0])
    return path.name[:120]


def unique_path(path: Path) -> Path:
    """``out.xlsx`` -> ``out.xlsx`` / ``out-1.xlsx`` / ``out-2.xlsx``: never overwrite silently."""
    if not path.exists():
        return path
    for n in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileError(f"too many files named like {path.name}")


def is_within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except (ValueError, OSError):
        return False


def out_path(name: str | None, *, suffixes: tuple[str, ...] = EXCEL_SUFFIXES,
             default: str = DEFAULT_OUT_NAME) -> Path:
    """A fresh, non-colliding path under files/out/ for a (possibly model-supplied) name."""
    ensure_dirs()
    return unique_path(OUT_DIR / safe_name(name, default=default, suffixes=suffixes))


def kind_of(path: str | Path) -> str | None:
    """"excel" / "word" / "slides" / "pdf" from the extension, or None when unsupported."""
    suffix = Path(path).suffix.lower()
    for kind in KINDS:
        if suffix in KIND_SUFFIXES[kind]:
            return kind
    return None


def kind_or_fail(path: str | Path) -> str:
    kind = kind_of(path)
    if kind is None:
        raise FileError(f"{Path(path).name}: unsupported file type (expected "
                        f"{', '.join(ALL_SUFFIXES)})")
    return kind
