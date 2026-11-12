"""Shared fixtures.

registry.py and process.py read ``config.MODELS_DIR`` / ``config.PID_FILE`` / ... through
the ``config`` module at call time, so patching the attributes on the module object is
enough to redirect every test into a throw-away directory. Nothing here touches the real
``models/`` or ``.hfserve/`` folders, and nothing talks to the network.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

import psutil
import pytest

from hfserve import config

SLEEPER = "import time; time.sleep(60)"

# On Windows a venv's python.exe is a launcher that runs the base interpreter as a *child*
# process, which would turn every spawned sleeper into a two-process tree. The children
# only need the standard library, so spawn them with the base interpreter directly.
CHILD_PYTHON = getattr(sys, "_base_executable", None) or sys.executable


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point HOME/MODELS/STATE/PID/LOG at tmp_path. Returns the (not yet created) state dir."""
    home = tmp_path / "home"
    state = home / ".hfserve"
    monkeypatch.setattr(config, "HOME_DIR", home)
    monkeypatch.setattr(config, "MODELS_DIR", home / "models")
    monkeypatch.setattr(config, "STATE_DIR", state)
    monkeypatch.setattr(config, "PID_FILE", state / "server.pid")
    monkeypatch.setattr(config, "LOG_FILE", state / "server.log")
    return state


@pytest.fixture
def models_dir(state_dir: Path) -> Path:
    """An existing, empty models root (config.MODELS_DIR is already patched by state_dir)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR


@pytest.fixture
def make_model(models_dir: Path) -> Callable[..., Path]:
    """Create a fake model folder under the patched MODELS_DIR and return its path.

    ``repo_id`` may be ``org/name`` or a single segment. ``files`` maps file name to size in
    bytes. ``config_json=False`` writes no config.json (use ``manifest`` to keep it a model dir).
    """

    def _make(
        repo_id: str,
        *,
        model_type: str | None = "llama",
        architectures: tuple[str, ...] = ("LlamaForCausalLM",),
        auto_map: dict | None = None,
        manifest: dict | None = None,
        config_json: bool = True,
        chat_template_file: bool = False,
        tokenizer_chat_template: str | None = None,
        tokenizer_config: dict | None = None,
        files: dict[str, int] | None = None,
    ) -> Path:
        path = models_dir.joinpath(*repo_id.split("/"))
        path.mkdir(parents=True, exist_ok=True)
        if config_json:
            cfg: dict = {"architectures": list(architectures)}
            if model_type is not None:
                cfg["model_type"] = model_type
            if auto_map is not None:
                cfg["auto_map"] = auto_map
            (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        if manifest is not None:
            (path / config.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        if chat_template_file:
            (path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        tok = dict(tokenizer_config or {})
        if tokenizer_chat_template is not None:
            tok["chat_template"] = tokenizer_chat_template
        if tok or tokenizer_config is not None:
            (path / "tokenizer_config.json").write_text(json.dumps(tok), encoding="utf-8")
        for name, size in (files or {}).items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * size)
        return path

    return _make


@pytest.fixture
def free_port() -> int:
    """A loopback port that nothing listens on (bound briefly, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def spawn():
    """Spawn ``python -c <code> *args`` children; kill every one (and its children) at teardown."""
    procs: list[subprocess.Popen] = []

    def _spawn(*args: str, code: str = SLEEPER, **popen_kwargs) -> subprocess.Popen:
        p = subprocess.Popen([CHILD_PYTHON, "-c", code, *args], **popen_kwargs)
        procs.append(p)
        return p

    yield _spawn

    for p in procs:
        kids: list[psutil.Process] = []
        if p.poll() is None:
            try:
                kids = psutil.Process(p.pid).children(recursive=True)
            except psutil.Error:
                kids = []
        for k in kids:
            try:
                k.kill()
            except psutil.Error:
                pass
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=5)
        except Exception:
            pass
        for stream in (p.stdin, p.stdout, p.stderr):
            if stream is not None:
                stream.close()


@pytest.fixture
def files_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point hffiles' workspace at tmp_path (the real files/in and files/out stay untouched)."""
    from hffiles import config as fconfig

    root = tmp_path / "files"
    monkeypatch.setattr(fconfig, "FILES_DIR", root)
    monkeypatch.setattr(fconfig, "IN_DIR", root / "in")
    monkeypatch.setattr(fconfig, "OUT_DIR", root / "out")
    fconfig.ensure_dirs()
    return root


SAMPLE_ROWS = [
    ["Datum", "Opis", "Iznos", "Kategorija"],
    ["2026-01-03", "Placa", 1500, "prihod"],
    ["2026-01-05", "Kava", 12.5, "trosak"],
    ["2026-01-09", "Najam", 430, "trosak"],
]


@pytest.fixture
def make_xlsx(tmp_path: Path) -> Callable[..., Path]:
    """Write a workbook with openpyxl directly, so tests do not need hffiles to set up."""

    def _make(name: str = "prihodi.xlsx", rows: list[list] | None = None,
              sheet: str = "Prihodi", extra_sheets: dict[str, list[list]] | None = None) -> Path:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = sheet
        for row in (SAMPLE_ROWS if rows is None else rows):
            ws.append(row)
        for title, extra in (extra_sheets or {}).items():
            other = wb.create_sheet(title)
            for row in extra:
                other.append(row)
        path = tmp_path / name
        wb.save(path)
        return path

    return _make


@pytest.fixture
def train_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point hftrain's TRAIN/DATA/RUNS/QA dirs at tmp_path (nothing under training/ is touched)."""
    from hftrain import config as tconfig

    root = tmp_path / "training"
    monkeypatch.setattr(tconfig, "TRAIN_DIR", root)
    monkeypatch.setattr(tconfig, "DATA_DIR", root / "data")
    monkeypatch.setattr(tconfig, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(tconfig, "QA_DIR", root / "QA")
    return root
