"""Process management: spawn / find / stop the API server and kill everything it started.

Design notes
------------
* The server process writes its PID to ``config.PID_FILE``; ``hfserve stop`` and the
  chat client's ``\\bye`` use it to find the server even from another terminal.
* On Windows the spawned server is placed in a Job Object with
  ``KILL_ON_JOB_CLOSE`` so that if the parent (the chat client) dies for any
  reason, the OS kills the server and all of its children too.
* ``stop_server`` first asks the server to shut down over HTTP, then kills the
  whole process tree (psutil), then sweeps for any leftover ``hfserve serve``
  processes by command line.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from . import config

IS_WINDOWS = sys.platform.startswith("win")

# Job-object handles must stay alive for as long as the child should be bound.
_JOB_HANDLES: list = []


# --------------------------------------------------------------------------- pid file
def pid_file(port: int | None = None) -> Path:
    """Pid file for a server on *port*: ``server.pid`` for the default port, ``server-<port>.pid`` otherwise."""
    if port is None or port == config.DEFAULT_PORT:
        return config.PID_FILE
    return config.PID_FILE.with_name(f"server-{port}.pid")


def write_pid(pid: int | None = None, port: int | None = None) -> None:
    config.ensure_dirs()
    pid_file(port).write_text(str(pid or os.getpid()), encoding="utf-8")


def read_pid(port: int | None = None) -> int | None:
    try:
        return int(pid_file(port).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def clear_pid(port: int | None = None) -> None:
    try:
        pid_file(port).unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- health
def connect_host(host: str) -> str:
    """Address to connect to for a server bound to *host* (wildcards map to loopback)."""
    return "127.0.0.1" if host in ("", "0.0.0.0", "::", "*") else host


def server_health(host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT, timeout: float = 1.5) -> dict | None:
    """Return the ``/api/health`` JSON if an hfserve server answers, else None."""
    import httpx

    try:
        # verify=False: plain http to localhost; skips building a TLS context (~0.25 s on Windows).
        r = httpx.get(f"{config.server_url(connect_host(host), port)}/api/health",
                      timeout=timeout, trust_env=False, verify=False)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and data.get("app") == config.APP_NAME:
                return data
    except Exception:
        pass
    return None


def is_server_running(host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT) -> bool:
    return server_health(host, port) is not None


def port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((connect_host(host), port)) == 0


def wait_for_server(host: str, port: int, timeout: float = config.SERVER_START_TIMEOUT,
                    proc: subprocess.Popen | None = None) -> bool:
    """Poll ``/api/health`` until it answers, the process dies, or *timeout* elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if is_server_running(host, port):
            return True
        time.sleep(0.25)
    return False


# --------------------------------------------------------------------------- spawn
def _bind_to_job(proc: subprocess.Popen) -> None:
    """Windows: kill *proc* (and its children) automatically when this process exits."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
        if ok and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            _JOB_HANDLES.append(job)
    except Exception:
        pass  # best effort; psutil tree kill is the primary mechanism


def spawn_server(host: str, port: int, model: str | None = None,
                 extra_args: list[str] | None = None) -> subprocess.Popen:
    """Start ``python -m hfserve serve`` in the background, logging to config.LOG_FILE."""
    config.ensure_dirs()
    # --host/--port are global options of the CLI and must precede the subcommand.
    cmd = [sys.executable, "-m", "hfserve", "--host", host, "--port", str(port), "serve"]
    if model:
        cmd += ["--model", model]
    if extra_args:
        cmd += list(extra_args)

    log = open(config.LOG_FILE, "a", encoding="utf-8", errors="replace")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} spawn: {' '.join(cmd)}\n")
    log.flush()

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "cwd": str(config.PROJECT_ROOT),
        "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if IS_WINDOWS:
        # New process group: Ctrl+C in the chat console is not delivered to the server;
        # shutdown is always explicit (HTTP /api/shutdown, then kill).
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
        if sys.platform.startswith("linux"):
            kwargs["preexec_fn"] = _linux_die_with_parent

    proc = subprocess.Popen(cmd, **kwargs)
    _bind_to_job(proc)
    return proc


def _linux_die_with_parent() -> None:
    """Child-side hook (Linux): SIGTERM the server if the chat client dies (PR_SET_PDEATHSIG)."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


# --------------------------------------------------------------------------- kill
def kill_tree(pid: int, include_parent: bool = True, timeout: float = 5.0) -> list[int]:
    """Terminate *pid* and all descendants. Returns the PIDs that were killed."""
    killed: list[int] = []
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return killed
    procs = parent.children(recursive=True)
    if include_parent:
        procs.append(parent)
    for p in procs:
        try:
            p.terminate()
            killed.append(p.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(alive, timeout=timeout)
    if IS_WINDOWS and include_parent and psutil.pid_exists(pid):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return killed


def _cmdline(proc: psutil.Process) -> list[str]:
    try:
        return proc.cmdline() or []
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []


def _is_hfserve_program(cmd: list[str]) -> bool:
    """True if *cmd* runs hfserve itself: ``python -m hfserve ...`` or the ``hfserve`` console script."""
    head = cmd[:4]
    for i, arg in enumerate(head):
        if arg == "-m" and i + 1 < len(head) and head[i + 1].lower() == "hfserve":
            return True
        if Path(arg).stem.lower() == "hfserve":
            return True
    return False


def _argv_port(cmd: list[str]) -> int | None:
    """The ``--port`` value in *cmd* (also ``--port=N``), or None."""
    for i, arg in enumerate(cmd):
        if arg == "--port" and i + 1 < len(cmd) and cmd[i + 1].isdigit():
            return int(cmd[i + 1])
        if arg.startswith("--port=") and arg[7:].isdigit():
            return int(arg[7:])
    return None


def _listening_ports(proc: psutil.Process) -> set[int]:
    try:
        return {c.laddr.port for c in proc.net_connections(kind="tcp")
                if c.status == psutil.CONN_LISTEN and c.laddr}
    except (psutil.Error, OSError):
        return set()


def _is_hfserve_server(proc: psutil.Process, port: int | None = None) -> bool:
    """An hfserve process that serves: ``serve`` in argv or listening on a TCP port.

    With *port*, only the server on that port matches (an actual listening socket on
    *port*, or ``serve`` in argv with ``--port`` equal to it / absent for the default port).
    """
    cmd = _cmdline(proc)
    if not cmd or not _is_hfserve_program(cmd):
        return False
    listening = _listening_ports(proc)
    if port is None:
        return "serve" in cmd or bool(listening)
    if port in listening:
        return True
    if "serve" in cmd:
        return (_argv_port(cmd) or config.DEFAULT_PORT) == port
    return False


def find_server_processes(port: int | None = None) -> list[psutil.Process]:
    """Running hfserve server processes (excluding this process and its ancestors).

    *port* restricts the result to the server bound to that port; None means all of them.
    """
    me = os.getpid()
    ancestors: set[int] = set()
    try:
        p = psutil.Process(me)
        while p is not None:
            ancestors.add(p.pid)
            p = p.parent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    out = []
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.pid in ancestors:
            continue
        if _is_hfserve_server(proc, port):
            out.append(proc)
    return out


def request_shutdown(host: str, port: int, timeout: float = 3.0) -> bool:
    import httpx

    try:
        r = httpx.post(f"{config.server_url(connect_host(host), port)}/api/shutdown",
                       timeout=timeout, trust_env=False, verify=False)
        return r.status_code == 200
    except Exception:
        return False


def stop_server(host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT,
                proc: subprocess.Popen | None = None, grace: float = config.SHUTDOWN_GRACE,
                log=print, all_servers: bool = False) -> int:
    """Stop the server on *port* and everything it spawned. Returns number of processes killed.

    Order: graceful HTTP shutdown -> wait -> kill tree of known PIDs (spawned proc,
    pid file) -> sweep for leftover hfserve server processes on *port*
    (``all_servers=True`` sweeps every hfserve server on the machine).
    """
    killed: set[int] = set()
    pids: list[int] = []
    if proc is not None and proc.poll() is None:
        pids.append(proc.pid)
    file_pid = read_pid(port)
    if file_pid and file_pid not in pids and psutil.pid_exists(file_pid):
        try:
            if _is_hfserve_program(_cmdline(psutil.Process(file_pid))):
                pids.append(file_pid)
        except psutil.Error:
            pass

    if is_server_running(host, port):
        log("Stopping server...")
        request_shutdown(host, port)
        deadline = time.time() + grace
        while time.time() < deadline:
            alive = any(psutil.pid_exists(p) for p in pids) if pids else port_in_use(host, port)
            if not alive:
                break
            time.sleep(0.2)

    for pid in pids:
        if psutil.pid_exists(pid):
            for k in kill_tree(pid):
                killed.add(k)

    for p in find_server_processes(None if all_servers else port):
        for k in kill_tree(p.pid):
            killed.add(k)

    if proc is not None:
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    clear_pid(port)
    if all_servers:
        for f in config.STATE_DIR.glob("server*.pid"):
            try:
                f.unlink()
            except OSError:
                pass
    return len(killed)


def kill_children_of_self() -> None:
    """Called by the server at exit: make sure nothing it spawned outlives it."""
    try:
        me = psutil.Process(os.getpid())
        kids = me.children(recursive=True)
    except psutil.Error:
        return
    for k in kids:
        try:
            k.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(kids, timeout=3)
    for k in alive:
        try:
            k.kill()
        except psutil.Error:
            pass


def log_tail(n: int = 40) -> str:
    try:
        lines = Path(config.LOG_FILE).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""
