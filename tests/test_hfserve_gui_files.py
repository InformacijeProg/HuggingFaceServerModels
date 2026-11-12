"""The GUI's file path: attach a workbook, let a plan run, keep working on the result.

A real window is built (no mainloop) with every outbound call stubbed: no server is
contacted or spawned, and the legislation refresh is disabled, so nothing here touches
the network. Skipped where Tk cannot open a display.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")

from hfserve import gui, process  # noqa: E402 - after importorskip
from hffiles import config as fconfig  # noqa: E402
from hffiles import convert  # noqa: E402
from hffiles import excel, files as hffiles_files, ops  # noqa: E402

PLAN = '{"ops": [{"op": "add_column", "header": "PDV", "formula": "=C{row}*0.25"}]}'


@pytest.fixture
def app(files_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """An App with a real widget tree and no outside world."""
    import hfzakon.refresh as zrefresh

    monkeypatch.setattr(zrefresh, "maybe_run", lambda progress=None: None)
    monkeypatch.setattr(process, "server_health", lambda *a, **k: {"model": "test/model"})
    monkeypatch.setattr(process, "port_in_use", lambda *a, **k: False)
    monkeypatch.setattr(process, "spawn_server", lambda *a, **k: pytest.fail("spawned a server"))
    # The connect worker would talk HTTP; post the event it ends with instead.
    monkeypatch.setattr(gui.App, "_connect",
                        lambda self: self.events.put(("connected", ["test/model"], "test/model")))
    try:
        root = tk.Tk()
    except tk.TclError as e:  # no display
        pytest.skip(f"tkinter unavailable: {e}")
    window = gui.App(root, "127.0.0.1", 65000, model="test/model")
    yield window
    try:
        window.client.close()  # the connect worker opened a socket at a dead port
    except Exception:  # noqa: BLE001
        pass
    vars(window).clear()  # Tk variables must be collected while Tk is still alive
    gc.collect()
    root.destroy()


def _wait_for(window, kind: str, timeout: float = 15.0):
    """The next event of one kind; other events (e.g. the failed connect) are ignored."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = window.events.get(timeout=0.1)
        except Exception:
            continue
        if event[0] == kind:
            return event
    pytest.fail(f"no {kind} event within {timeout}s")


def _attach(window, path: Path) -> Path:
    """Do what the attach worker does, then hand the event to the window."""
    copy = hffiles_files.upload(path)
    desc = hffiles_files.describe(copy)
    window._handle(("file_attached", copy, desc, hffiles_files.digest_text(desc),
                    hffiles_files.summary(desc)))
    return copy


def test_the_file_row_starts_empty(app):
    assert "nema" in app.l_file["text"]
    assert str(app.b_detach["state"]) == "disabled"
    assert app.file_path is None


def test_attach_shows_the_digest_and_enables_detach(app, make_xlsx):
    _attach(app, make_xlsx())
    assert app.file_path is not None
    assert str(app.b_detach["state"]) == "normal"
    assert "C=Iznos (number)" in app.transcript.get("1.0", "end")  # the model's view, shown
    app.detach_file()
    assert app.file_path is None and str(app.b_detach["state"]) == "disabled"


def test_a_plan_is_executed_and_its_result_becomes_the_working_file(app, make_xlsx):
    uploaded = _attach(app, make_xlsx())
    before = uploaded.read_bytes()

    app.busy = True
    app.plan_pending = True
    assert app._run_plan(PLAN) is True
    _kind, result, new_desc = _wait_for(app, "plan_result")
    app._on_plan_result(result, new_desc)

    assert result.ok and result.saved
    assert result.target.parent == fconfig.OUT_DIR
    assert uploaded.read_bytes() == before  # the upload is never written to
    assert app.file_path == result.target  # a follow-up request builds on the result
    assert not app.busy
    assert "PDV" in app.transcript.get("1.0", "end")


def test_an_answer_without_a_plan_is_reported_and_frees_the_window(app, make_xlsx):
    _attach(app, make_xlsx())
    app.busy = True
    app.plan_pending = True
    assert app._run_plan("Naravno, sortirat cu tablicu!") is False
    assert "nije vratio plan" in app.transcript.get("1.0", "end")


def test_a_failing_plan_writes_nothing(app, make_xlsx):
    _attach(app, make_xlsx())
    app.busy = True
    assert app._run_plan('{"ops": [{"op": "sort_rows", "column": "Nema"}]}') is True
    _kind, result, new_desc = _wait_for(app, "plan_result")
    app._on_plan_result(result, new_desc)
    assert not result.ok and not result.saved
    assert list(fconfig.OUT_DIR.iterdir()) == []
    assert not app.busy


def test_plan_mode_keeps_the_json_out_of_the_chat_history(app, make_xlsx):
    _attach(app, make_xlsx())
    app.history = [{"role": "user", "content": "dodaj stupac"}]
    app.answer_text = PLAN
    app.plan_pending = True
    app.busy = True
    app._on_done({"done_reason": "stop"})
    assert app.history == []
    assert app.busy  # still working: the plan runs in a worker
    _wait_for(app, "plan_result")


def test_an_attach_error_is_shown_without_breaking_the_chat(app):
    app.busy = True
    app._handle(("file_error", "FileError: no such file: x.xlsx"))
    assert "no such file" in app.transcript.get("1.0", "end")
    assert not app.busy


def test_the_label_survives_an_empty_sheet(app, files_dir):
    created = ops.create_workbook({"Prazno": []})
    desc = hffiles_files.describe(created.target)
    app._handle(("file_attached", created.target, desc, "", hffiles_files.summary(desc)))
    assert "Excel" in app.l_file["text"]

# --------------------------------------------------- the download / PDF buttons
def test_the_save_buttons_are_off_until_something_is_attached(app):
    assert str(app.b_save["state"]) == "disabled"
    assert str(app.b_pdf["state"]) == "disabled"


def test_word_and_pdf_files_attach_too(app, tmp_path):
    from docx import Document

    source = tmp_path / "dopis.docx"
    doc = Document()
    doc.add_heading("Naslov", level=1)
    doc.add_paragraph("Odlomak")
    doc.save(str(source))
    _attach(app, source)
    assert app.file_desc["kind"] == "word"
    assert "Word" in app.l_file["text"]
    assert str(app.b_pdf["state"]) == "normal"
    assert "H1 Naslov" in app.transcript.get("1.0", "end")


def test_save_copy_writes_where_the_user_points(app, tmp_path, make_xlsx, monkeypatch):
    _attach(app, make_xlsx())
    target = tmp_path / "preuzeto.xlsx"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: str(target))
    app.save_copy()
    assert target.is_file()
    assert target.read_bytes() == app.file_path.read_bytes()
    assert str(target) in app.transcript.get("1.0", "end")


def test_save_copy_does_nothing_when_the_dialog_is_cancelled(app, tmp_path, make_xlsx,
                                                            monkeypatch):
    _attach(app, make_xlsx())
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: "")
    app.save_copy()  # must not raise
    assert "spremljeno" not in app.transcript.get("1.0", "end")


@pytest.mark.skipif(not any(convert.backends().values()),
                    reason="no Office or LibreOffice installed")
def test_save_pdf_exports_and_frees_the_window(app, tmp_path, make_xlsx, monkeypatch):
    _attach(app, make_xlsx())
    target = tmp_path / "izvoz.pdf"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **k: str(target))
    app.save_pdf()
    assert app.busy  # the export runs in a worker
    kind, path = _wait_for(app, "pdf_ready", timeout=180)
    app._handle((kind, path))
    assert target.is_file() and target.stat().st_size > 500
    assert not app.busy
    assert "PDF" in app.transcript.get("1.0", "end")
