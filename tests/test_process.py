"""Unit tests for hfserve.process: pid file, ports, health probes, process-tree kill, stop_server.

Everything runs against throw-away child processes and ephemeral loopback ports; the default
server port is never touched. Tests whose code path ends in the ``hfserve serve`` sweep
(which kills every such process on the machine) use the ``no_real_server`` guard fixture.

Windows note: a refused loopback connection takes ~2 s to fail (the stack retries the SYN
after the RST), so every probe of a dead port costs its full timeout; on top of that every
``httpx.get``/``httpx.post`` in process.py builds a fresh SSL context (~0.25 s). The tests
below shorten timeouts or stub the HTTP probe / the ~1 s process sweep wherever those are not
the thing under test, and keep them real where they are.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import pytest

from hfserve import config, process


@pytest.fixture(scope="module", autouse=True)
def fast_ssl_context(tmp_path_factory):
    """Make httpx's per-call SSL context cheap.

    Every ``httpx.get``/``httpx.post`` in process.py creates a client and therefore an SSL
    context, and parsing certifi's full CA bundle costs ~0.25 s per call on Windows. The
    probes are plain http:// so TLS is never negotiated. process.py passes
    ``trust_env=False`` (so SSL_CERT_FILE is ignored) and httpx then always calls
    ``certifi.where()`` at request time; pointing that at a one-certificate PEM for the
    duration of this module turns each probe into ~0.01 s.
    """
    try:
        import certifi
    except ImportError:  # certifi is an httpx dependency; fall back to the slow path
        yield
        return
    bundle = Path(certifi.where()).read_text(encoding="utf-8", errors="replace")
    end = "-----END CERTIFICATE-----"
    one_cert = tmp_path_factory.mktemp("certs") / "one-cert.pem"
    one_cert.write_text(bundle[: bundle.index(end) + len(end)] + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(certifi, "where", lambda: str(one_cert))
        yield

HOST = "127.0.0.1"
# The inner sys.executable is the base interpreter too (see conftest.CHILD_PYTHON).
SPAWN_CHILD = (
    "import subprocess, sys, time; "
    "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
    "print(c.pid, flush=True); time.sleep(60)"
)


def _unused_pid() -> int:
    for pid in range(2_000_000_000, 2_000_100_000, 4):
        if not psutil.pid_exists(pid):
            return pid
    pytest.skip("could not find an unused pid")


def _wait_until(pred, timeout: float = 5.0, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


def _cmdline_visible(p: subprocess.Popen) -> bool:
    """True once psutil can read the child's command line (cheap, unlike find_server_processes)."""
    try:
        return process._is_hfserve_server(psutil.Process(p.pid))
    except psutil.Error:
        return False


@pytest.fixture(scope="module")
def no_real_server():
    """Skip tests that sweep-kill ``hfserve serve`` processes while a real one is running.

    Checked once per module; only python-named processes get their command line read, which
    keeps it well under find_server_processes()' ~1 s.
    """
    hits = []
    for proc in psutil.process_iter(["pid", "name"]):
        if "python" in (proc.info["name"] or "").lower() and process._is_hfserve_server(proc):
            try:
                hits.append((proc.pid, " ".join(proc.cmdline())[:120]))
            except psutil.Error:
                hits.append((proc.pid, "?"))
    if hits:
        pytest.skip(f"an hfserve server is running {hits}; refusing to run tests that kill it")


@pytest.fixture
def no_http_probe(monkeypatch):
    """Stub the /api/health probe (1.5 s per dead port on Windows) where it is not under test."""
    monkeypatch.setattr(process, "is_server_running", lambda host, port: False)


@pytest.fixture
def no_sweep(monkeypatch):
    """Stub the ~1 s process sweep where it is not under test."""
    monkeypatch.setattr(process, "find_server_processes", lambda port=None: [])


@pytest.fixture
def fake_server():
    """Loopback HTTP server: GET /api/health and POST /api/shutdown, behaviour set per test."""
    state = {
        "health": {"app": config.APP_NAME, "status": "ok"},
        "health_status": 200,
        "shutdown_status": 200,
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        # Headers and body go out as two segments; without TCP_NODELAY the second one waits
        # for Windows' 200 ms delayed ACK, which would add ~0.25 s to every probe.
        disable_nagle_algorithm = True

        def log_message(self, *args):  # keep pytest output clean
            pass

        def _send(self, status: int, body) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state["requests"].append(("GET", self.path))
            if self.path == "/api/health":
                self._send(state["health_status"], state["health"])
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            state["requests"].append(("POST", self.path))
            if self.path == "/api/shutdown":
                self._send(state["shutdown_status"], {"ok": True})
            else:
                self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer((HOST, 0), Handler)
    # serve_forever's default 0.5 s poll interval would make every shutdown() take 0.5 s.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    state["host"], state["port"] = server.server_address[0], server.server_address[1]
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


# =========================================================================== pid file
def test_write_pid_defaults_to_own_pid(state_dir):
    assert not state_dir.exists()
    process.write_pid()
    assert state_dir.is_dir()  # ensure_dirs() created it
    assert config.PID_FILE.read_text(encoding="utf-8") == str(os.getpid())
    assert process.read_pid() == os.getpid()


def test_write_pid_explicit_value(state_dir):
    process.write_pid(4242)
    assert process.read_pid() == 4242
    process.write_pid(17)  # overwrites
    assert process.read_pid() == 17


def test_read_pid_missing_file_returns_none(state_dir):
    assert not config.PID_FILE.exists()
    assert process.read_pid() is None


@pytest.mark.parametrize("content", ["", "not-a-pid", "12.5", "\n"])
def test_read_pid_garbage_returns_none(state_dir, content):
    state_dir.mkdir(parents=True)
    config.PID_FILE.write_text(content, encoding="utf-8")
    assert process.read_pid() is None


def test_read_pid_tolerates_surrounding_whitespace(state_dir):
    state_dir.mkdir(parents=True)
    config.PID_FILE.write_text("  77\r\n", encoding="utf-8")
    assert process.read_pid() == 77


def test_clear_pid_removes_file_and_is_idempotent(state_dir):
    process.write_pid(99)
    assert config.PID_FILE.exists()
    process.clear_pid()
    assert not config.PID_FILE.exists()
    assert process.read_pid() is None
    process.clear_pid()  # nothing to remove -> no error


# =========================================================================== ports
def test_port_in_use_true_for_listening_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert process.port_in_use(HOST, port) is True


def test_port_in_use_false_for_free_port(free_port):
    assert process.port_in_use(HOST, free_port) is False


# =========================================================================== health / shutdown
def test_server_health_none_when_nothing_listens(free_port):
    started = time.perf_counter()
    assert process.server_health(HOST, free_port, timeout=0.2) is None
    assert time.perf_counter() - started < 4.0  # bounded by the timeout, no hang


def test_server_health_returns_json_from_hfserve(fake_server):
    host, port = fake_server["host"], fake_server["port"]
    assert process.server_health(host, port) == {"app": "hfserve", "status": "ok"}
    assert process.is_server_running(host, port) is True
    assert process.port_in_use(host, port) is True
    assert ("GET", "/api/health") in fake_server["requests"]


def test_server_health_rejects_other_app(fake_server):
    fake_server["health"] = {"app": "ollama", "status": "ok"}
    assert process.server_health(fake_server["host"], fake_server["port"]) is None
    assert process.is_server_running(fake_server["host"], fake_server["port"]) is False


def test_server_health_rejects_non_200(fake_server):
    fake_server["health_status"] = 503
    assert process.server_health(fake_server["host"], fake_server["port"]) is None


def test_server_health_rejects_non_dict_body(fake_server):
    fake_server["health"] = ["hfserve"]
    assert process.server_health(fake_server["host"], fake_server["port"]) is None


def test_request_shutdown(fake_server, free_port):
    host, port = fake_server["host"], fake_server["port"]
    assert process.request_shutdown(host, port) is True
    assert ("POST", "/api/shutdown") in fake_server["requests"]
    fake_server["shutdown_status"] = 500
    assert process.request_shutdown(host, port) is False
    assert process.request_shutdown(HOST, free_port, timeout=0.2) is False  # nothing listening


# =========================================================================== wait_for_server
def test_wait_for_server_false_when_process_died(free_port, spawn):
    p = spawn(code="pass")
    p.wait(timeout=10)
    started = time.perf_counter()
    assert process.wait_for_server(HOST, free_port, timeout=5.0, proc=p) is False
    assert time.perf_counter() - started < 2.0  # returns as soon as the process is seen dead


def test_wait_for_server_times_out(free_port, no_http_probe):
    started = time.perf_counter()
    assert process.wait_for_server(HOST, free_port, timeout=0.1) is False
    assert 0.1 <= time.perf_counter() - started < 3.0


def test_wait_for_server_true_when_healthy(fake_server):
    assert process.wait_for_server(fake_server["host"], fake_server["port"], timeout=5.0) is True


# =========================================================================== kill_tree
def test_kill_tree_kills_child_process(spawn):
    p = spawn()
    assert psutil.pid_exists(p.pid)
    killed = process.kill_tree(p.pid)
    assert killed == [p.pid]
    assert p.wait(timeout=5) is not None
    assert not psutil.pid_exists(p.pid)


def test_kill_tree_kills_grandchildren(spawn):
    p = spawn(code=SPAWN_CHILD, stdout=subprocess.PIPE, text=True)
    child_pid = int(p.stdout.readline().strip())
    assert psutil.pid_exists(child_pid)
    killed = process.kill_tree(p.pid)
    assert set(killed) == {p.pid, child_pid}
    assert p.wait(timeout=5) is not None
    assert _wait_until(lambda: not psutil.pid_exists(child_pid))
    assert not psutil.pid_exists(p.pid)


def test_kill_tree_can_spare_the_parent(spawn):
    p = spawn(code=SPAWN_CHILD, stdout=subprocess.PIPE, text=True)
    child_pid = int(p.stdout.readline().strip())
    killed = process.kill_tree(p.pid, include_parent=False)
    assert killed == [child_pid]
    assert _wait_until(lambda: not psutil.pid_exists(child_pid))
    assert p.poll() is None  # parent still alive


def test_kill_tree_unknown_pid_returns_empty_list():
    assert process.kill_tree(_unused_pid()) == []


def test_kill_children_of_self(spawn):
    p = spawn()
    process.kill_children_of_self()
    assert p.wait(timeout=5) is not None
    assert not psutil.pid_exists(p.pid)


# =========================================================================== find_server_processes
def test_find_server_processes_excludes_self_and_ancestors(monkeypatch):
    monkeypatch.setattr(process, "_is_hfserve_server", lambda proc, port=None: True)
    pids = {p.pid for p in process.find_server_processes()}
    assert pids  # every other process matched
    assert os.getpid() not in pids
    parent = psutil.Process().parent()
    if parent is not None:
        assert parent.pid not in pids


def test_find_server_processes_finds_serve_command_line_not_self(spawn):
    p = spawn("hfserve", "serve")  # python -c "..." hfserve serve
    assert _wait_until(lambda: _cmdline_visible(p))
    found = {q.pid for q in process.find_server_processes()}
    assert p.pid in found
    assert os.getpid() not in found


def test_is_hfserve_server_requires_serve_argument(spawn):
    serve = spawn("hfserve", "serve")
    other = spawn("hfserve", "list")
    assert process._is_hfserve_server(psutil.Process(serve.pid)) is True
    assert process._is_hfserve_server(psutil.Process(other.pid)) is False


# =========================================================================== stop_server
def test_stop_server_nothing_running_returns_zero_and_clears_pid(no_sweep, state_dir, free_port):
    # Real HTTP probe of a free port (module default timeout) and a stale pid file; the ~1 s
    # process sweep is exercised for real by test_stop_server_sweeps_leftover_serve_processes.
    process.write_pid(_unused_pid(), port=free_port)
    logs: list[str] = []
    started = time.perf_counter()
    assert process.stop_server(HOST, free_port, log=logs.append) == 0
    assert time.perf_counter() - started < 8.0
    assert logs == []  # no "Stopping server..." because nothing answered
    assert not process.pid_file(free_port).exists()
    assert process.read_pid() is None


def test_stop_server_without_pid_file(no_http_probe, no_sweep, state_dir, free_port):
    assert not config.PID_FILE.exists()
    assert process.stop_server(HOST, free_port, log=lambda *a: None) == 0
    assert not config.PID_FILE.exists()


def test_stop_server_kills_pid_from_pid_file(no_http_probe, no_sweep, state_dir, free_port, spawn):
    # The child's command line ends with "hfserve serve", so it looks like a real server.
    p = spawn("hfserve", "serve")
    process.write_pid(p.pid, port=free_port)
    assert process.stop_server(HOST, free_port, log=lambda *a: None) == 1
    assert p.wait(timeout=5) is not None
    assert not psutil.pid_exists(p.pid)
    assert not process.pid_file(free_port).exists()


def test_stop_server_leaves_foreign_pid_from_stale_pid_file_alone(no_http_probe, no_sweep, state_dir,
                                                                 free_port, spawn):
    # A stale pid file whose pid was reused by an unrelated program must never get tree-killed.
    p = spawn()
    process.write_pid(p.pid, port=free_port)
    assert process.stop_server(HOST, free_port, log=lambda *a: None) == 0
    assert p.poll() is None and psutil.pid_exists(p.pid)
    assert not process.pid_file(free_port).exists()  # the stale file is still cleaned up


def test_stop_server_kills_spawned_proc_argument(no_http_probe, no_sweep, state_dir, free_port, spawn):
    p = spawn()
    assert process.stop_server(HOST, free_port, proc=p, log=lambda *a: None) == 1
    assert p.poll() is not None
    assert not config.PID_FILE.exists()


def test_stop_server_sweeps_leftover_serve_processes(no_real_server, no_http_probe, state_dir,
                                                     free_port, spawn):
    p = spawn("hfserve", "serve", "--port", str(free_port))
    other = spawn("hfserve", "serve", "--port", str(free_port + 1))  # a server on another port
    assert _wait_until(lambda: _cmdline_visible(p) and _cmdline_visible(other))
    assert process.stop_server(HOST, free_port, log=lambda *a: None) >= 1
    assert p.wait(timeout=5) is not None
    assert not psutil.pid_exists(p.pid)
    assert other.poll() is None  # the sweep is port-specific
    assert process.stop_server(HOST, free_port, log=lambda *a: None, all_servers=True) >= 1
    assert other.wait(timeout=5) is not None


def test_pid_file_is_per_port(state_dir):
    assert process.pid_file(None) == config.PID_FILE
    assert process.pid_file(config.DEFAULT_PORT) == config.PID_FILE
    assert process.pid_file(11499).name == "server-11499.pid"
    process.write_pid(123, port=11499)
    assert process.read_pid(11499) == 123
    assert process.read_pid() is None
    process.clear_pid(11499)
    assert process.read_pid(11499) is None


def test_is_hfserve_program_and_argv_port():
    assert process._is_hfserve_program(["python.exe", "-m", "hfserve", "serve"])
    assert process._is_hfserve_program([r"C:\x\.vemv\Scripts\hfserve.exe", "serve"])
    assert not process._is_hfserve_program(["python", "-m", "http.server", "--directory", "hfserve-docs", "serve"])
    assert not process._is_hfserve_program(["python", r"E:\HuggingFaceServerModels\tools.py", "serve"])
    assert process._argv_port(["x", "--port", "11499"]) == 11499
    assert process._argv_port(["x", "--port=11499"]) == 11499
    assert process._argv_port(["x", "serve"]) is None


def test_stop_server_asks_running_server_to_shut_down(no_sweep, state_dir, fake_server):
    host, port = fake_server["host"], fake_server["port"]
    logs: list[str] = []
    started = time.perf_counter()
    n = process.stop_server(host, port, grace=0.3, log=logs.append)
    elapsed = time.perf_counter() - started
    assert n == 0  # the fake server lives in this process; nothing to kill
    assert logs == ["Stopping server..."]
    assert ("POST", "/api/shutdown") in fake_server["requests"]
    assert 0.3 <= elapsed < 5.0  # waited out the grace period since the port stayed in use
    assert not config.PID_FILE.exists()


# =========================================================================== log_tail
def test_log_tail(state_dir):
    assert process.log_tail() == ""  # no log file yet
    state_dir.mkdir(parents=True)
    config.LOG_FILE.write_text("\n".join(f"line{i}" for i in range(50)) + "\n", encoding="utf-8")
    assert process.log_tail(3) == "line47\nline48\nline49"
    assert process.log_tail(100).splitlines() == [f"line{i}" for i in range(50)]
