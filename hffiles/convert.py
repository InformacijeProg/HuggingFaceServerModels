"""Export a workbook, document or deck to PDF.

There is no honest pure-Python way to do this: a PDF of a .docx has to be *rendered*, and
only the programs that own those formats know how their own layout engine breaks pages. So
this module drives a renderer that is installed on the machine:

1. **Microsoft Office** through COM (pywin32). Highest fidelity, Windows only, and it also
   recalculates a workbook's formulas on the way out.
2. **LibreOffice** ``soffice --headless --convert-to pdf``. Cross-platform, good fidelity.

With neither installed the export fails with a message saying so, rather than producing a
re-typeset lookalike that silently differs from the file the user has.

COM is apartment-threaded, so each call initialises and uninitialises COM for its own
thread - which is what lets the GUI run this in a worker.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config
from .errors import FileError

# Export constants from the Word, Excel and PowerPoint type libraries.
WD_EXPORT_PDF = 17
XL_TYPE_PDF = 0
PP_SAVE_AS_PDF = 32

SOFFICE_CANDIDATES = (
    "C:/Program Files/LibreOffice/program/soffice.exe",
    "C:/Program Files (x86)/LibreOffice/program/soffice.exe",
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)
TIMEOUT = 300
OFFICE_PROG_IDS = {"word": "Word.Application", "excel": "Excel.Application",
                   "slides": "PowerPoint.Application"}


def find_libreoffice() -> Path | None:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return Path(found)
    for candidate in SOFFICE_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def office_available() -> bool:
    """True when pywin32 is importable and an Office application is registered."""
    if os.name != "nt":
        return False
    try:
        import win32com.client  # noqa: F401
        import winreg
    except ImportError:
        return False
    for prog_id in OFFICE_PROG_IDS.values():
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, prog_id))
            return True
        except OSError:
            continue
    return False


def backends() -> dict:
    """What this machine can render with; shown by ``hffiles where`` and the GUI."""
    soffice = find_libreoffice()
    return {"office": office_available(), "libreoffice": str(soffice) if soffice else None}


# ------------------------------------------------------------------ backends
# Run in a dedicated child process: DispatchEx starts a *new* Office instance instead of
# attaching to the one the user has open (quitting that would close their documents), and a
# child keeps Office's COM apartment, its slow asynchronous shutdown and any RPC failure out
# of this process entirely. It also gives the export a hard timeout.
OFFICE_SCRIPT = """
import sys

import pythoncom
import win32com.client

PROG = {"word": "Word.Application", "excel": "Excel.Application",
        "slides": "PowerPoint.Application"}
kind, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
pythoncom.CoInitialize()
app = win32com.client.DispatchEx(PROG[kind])
try:
    try:
        app.DisplayAlerts = False
    except Exception:
        pass
    if kind == "word":
        doc = app.Documents.Open(src, ReadOnly=True, AddToRecentFiles=False)
        try:
            doc.ExportAsFixedFormat(OutputFileName=dst, ExportFormat=17)
        finally:
            doc.Close(0)
    elif kind == "excel":
        book = app.Workbooks.Open(src, ReadOnly=True, UpdateLinks=0)
        try:
            book.ExportAsFixedFormat(0, dst)
        finally:
            book.Close(False)
    else:
        deck = app.Presentations.Open(src, WithWindow=False, ReadOnly=True)
        try:
            deck.SaveAs(dst, 32)
        finally:
            deck.Close()
finally:
    try:
        app.Quit()
    except Exception:
        pass
"""
CREATE_NO_WINDOW = 0x08000000  # no console flash when the GUI exports


def _via_office(src: Path, dst: Path, kind: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run([sys.executable, "-c", OFFICE_SCRIPT, kind, str(src), str(dst)],
                              capture_output=True, text=True, timeout=TIMEOUT, check=False,
                              creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired as e:
        raise FileError(f"{config.KIND_LABELS[kind]} did not finish exporting {src.name} "
                        f"within {TIMEOUT}s") from e
    if done.returncode != 0 or not dst.is_file():
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        raise FileError(f"{config.KIND_LABELS[kind]} could not export {src.name} to PDF"
                        + (f": {detail[-1][:200]}" if detail else "")
                        + ". Is the file open in Office?")


def _via_libreoffice(src: Path, dst: Path, soffice: Path) -> None:
    out_dir = dst.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(soffice), "--headless", "--norestore", "--convert-to", "pdf",
               "--outdir", str(out_dir), str(src)]
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:
        raise FileError(f"LibreOffice did not finish converting {src.name} "
                        f"within {TIMEOUT}s") from e
    produced = out_dir / f"{src.stem}.pdf"
    if not produced.is_file():
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        raise FileError(f"LibreOffice could not convert {src.name}"
                        + (f": {detail[-1][:200]}" if detail else ""))
    if produced != dst:
        produced.replace(dst)


# -------------------------------------------------------------------- public
def to_pdf(source: str | Path, target: str | Path | None = None, *,
           prefer: str | None = None) -> Path:
    """Render ``source`` to PDF and return the path written (files/out by default).

    ``prefer`` forces a backend ("office" or "libreoffice") instead of taking whichever is
    installed. A PDF source is simply copied.
    """
    src = Path(source).expanduser()
    if not src.is_file():
        raise FileError(f"no such file: {src}")
    kind = config.kind_or_fail(src)
    dst = (Path(target).expanduser() if target is not None
           else config.out_path(f"{src.stem}.pdf", suffixes=config.PDF_SUFFIXES,
                                default=config.DEFAULT_NAMES["pdf"]))
    if dst.suffix.lower() not in config.PDF_SUFFIXES:
        raise FileError(f"{dst.name}: the target of a PDF export must end in .pdf")
    if kind == "pdf":
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst

    have = backends()
    wanted = (prefer or "").strip().lower() or None
    if wanted not in (None, "office", "libreoffice"):
        raise FileError('prefer must be "office" or "libreoffice"')
    if wanted == "office" and not have["office"]:
        raise FileError("Microsoft Office is not available on this machine")
    if wanted == "libreoffice" and not have["libreoffice"]:
        raise FileError("LibreOffice is not installed")

    if wanted == "office" or (wanted is None and have["office"]):
        _via_office(src, dst, kind)
    elif wanted == "libreoffice" or have["libreoffice"]:
        _via_libreoffice(src, dst, Path(have["libreoffice"]))
    else:
        raise FileError(
            f"cannot export {src.name} to PDF: no renderer found. Install LibreOffice "
            f"(free, cross-platform) or Microsoft Office, or save the file as PDF from the "
            f"program itself. A PDF rebuilt in Python would not match this file's layout.")
    if not dst.is_file():
        raise FileError(f"the PDF export produced no file at {dst}")
    return dst
