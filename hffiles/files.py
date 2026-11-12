"""One entry point per question about a file, whatever its format.

``excel.py``, ``word.py``, ``slides.py`` and ``pdf.py`` each know one format. This module
is the switchboard the GUI and the CLI talk to, so neither has to care which library is
involved: ``files.describe(path)`` works for a workbook, a document, a deck or a PDF, and
the dict that comes back always carries its ``kind``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import config, convert, excel, pdf, slides, word
from .errors import FileError

MODULES = {"excel": excel, "word": word, "slides": slides, "pdf": pdf}


def kind_of(path: str | Path) -> str:
    """"excel" / "word" / "slides" / "pdf", or a FileError naming what is supported."""
    return config.kind_or_fail(path)


def module_for(path_or_kind: str | Path):
    kind = path_or_kind if path_or_kind in config.KINDS else kind_of(path_or_kind)
    return MODULES[kind]


def check_path(path: str | Path, *, must_exist: bool = True) -> Path:
    return module_for(path).check_path(path, must_exist=must_exist)


def describe(path: str | Path) -> dict:
    """What is in the file, as plain data - the only thing a model is ever shown."""
    return module_for(path).describe(path)


def digest_text(desc: dict) -> str:
    """The prompt-sized rendering of any ``describe`` result."""
    kind = str(desc.get("kind") or "excel")
    if kind not in MODULES:
        raise FileError(f"unknown file kind {kind!r}")
    return MODULES[kind].digest_text(desc)


def upload(path: str | Path) -> Path:
    """Copy a user-chosen file into files/in/ and return the copy.

    The original is never opened for writing again: later steps read this copy and write
    into files/out/, so no plan - good or bad - can damage the user's file.
    """
    src = check_path(path)
    kind = kind_of(src)
    config.ensure_dirs()
    dst = config.unique_path(config.IN_DIR / config.safe_name(
        src.name, default=config.DEFAULT_NAMES[kind], suffixes=config.KIND_SUFFIXES[kind]))
    shutil.copy2(src, dst)
    return dst


def copy_file(source: str | Path, target: str | Path | None = None) -> Path:
    """Byte-for-byte copy into files/out (or a chosen path).

    For every format this keeps what a rewrite through the libraries cannot: a workbook's
    pivot tables and macros, a deck's animations, a PDF's exact bytes.
    """
    src = check_path(source)
    kind = kind_of(src)
    dst = (Path(target).expanduser() if target
           else config.out_path(src.name, suffixes=config.KIND_SUFFIXES[kind],
                                default=config.DEFAULT_NAMES[kind]))
    if dst.resolve() == src.resolve():
        raise FileError("source and target are the same file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def to_pdf(source: str | Path, target: str | Path | None = None, *,
           prefer: str | None = None) -> Path:
    """Render any supported file to PDF (convert.py decides which renderer is used)."""
    return convert.to_pdf(source, target, prefer=prefer)


def pdf_backends() -> dict:
    return convert.backends()


def summary(desc: dict) -> str:
    """One short line for a status bar: "prihodi.xlsx - Excel, 2 list(a), 41 redaka"."""
    kind = str(desc.get("kind") or "excel")
    label = config.KIND_LABELS.get(kind, kind)
    if kind == "excel":
        sheets = desc.get("sheets") or []
        rows = sum(int(s.get("rows") or 0) for s in sheets)
        detail = f'{len(desc.get("sheet_names") or sheets)} list(a), {rows} redaka'
    elif kind == "word":
        detail = f'{desc.get("paragraphs", 0)} odlomaka, {desc.get("words", 0)} rijeci'
    elif kind == "slides":
        detail = f'{desc.get("slide_count", 0)} slajdova'
    else:
        detail = f'{desc.get("pages", 0)} stranica'
        if not desc.get("has_text", True):
            detail += ", bez teksta (skenirano?)"
    return f'{desc.get("file", "?")} - {label}, {detail}'
