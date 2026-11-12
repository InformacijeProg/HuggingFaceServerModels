"""Command-line front end: argparse subcommands plus an interactive command shell.

``python -m hfserve`` (or the ``hfserve`` script) with no arguments opens the shell;
every shell line is parsed by the very same argparse parser as the command line.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import traceback
from datetime import datetime

import httpx

from . import __version__, chat, config, process, registry
from .chat import error_text
from .registry import ModelError, ModelNotFound, human_size
from .ui import bold, dim, red, setup_console

PROG = "hfserve"
EXAMPLE_URL = "https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct"
# ``run`` flags forwarded to the server as generation options (only when given).
GEN_OPTIONS = ("temperature", "top_p", "top_k", "num_predict", "repeat_penalty", "seed", "reasoning_effort")
SHELL_COMMANDS = "list · pull <url> · run <model|#> · serve · stop · ps · rm <model> · show <model> · help · quit"
API_LINE = "  API: POST /api/chat  ·  POST /v1/chat/completions  ·  GET /api/tags"


# --------------------------------------------------------------------------- helpers
def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso)[:16]


def _json(r: httpx.Response) -> dict:
    try:
        data = r.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _print_log_hint(lines: int = 15) -> None:
    print(dim(f"server log: {config.LOG_FILE}"))
    tail = process.log_tail(lines)
    if tail:
        print(dim(tail))


def _quiet(*_args, **_kwargs) -> None:
    """Silences process.stop_server()'s own progress line."""


# --------------------------------------------------------------------------- commands
def cmd_list(args: argparse.Namespace) -> int:
    models = registry.list_models()
    index_path = registry.write_index(models)  # keep models/models.json current
    if getattr(args, "json", False):
        import json

        print(json.dumps(registry.build_index(models), indent=2, ensure_ascii=False))
        return 0
    if not models:
        print(f"No models installed. Download one with: {PROG} pull {EXAMPLE_URL}")
    else:
        header = ("#", "NAME", "SIZE", "TYPE", "DOWNLOADED")
        rows = [
            (str(i), m.repo_id + (" (incomplete)" if m.incomplete else ""), human_size(m.size_bytes),
             m.model_type or "-", _fmt_date(m.downloaded_at))
            for i, m in enumerate(models, 1)
        ]
        widths = [max(len(r[i]) for r in (header, *rows)) for i in range(5)]

        def fmt(r: tuple) -> str:
            return (f"{r[0]:>{widths[0]}}  {r[1]:<{widths[1]}}  {r[2]:>{widths[2]}}  "
                    f"{r[3]:<{widths[3]}}  {r[4]}")

        print(bold(fmt(header)))
        for r in rows:
            print(fmt(r))
    print(f"\nModels directory: {config.MODELS_DIR}   (index: {index_path.name})")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    try:
        registry.pull_model(args.ref, revision=args.revision, force=args.force)
    except ModelError as e:
        print(red(f"Error: {e}"))
        return 1
    except OSError as e:  # network (requests exceptions are OSErrors) or disk trouble
        print(red(f"Download failed: {e}"))
        return 1
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    model = registry.find_model(args.model)
    if not args.yes:
        try:
            answer = input(f"Delete {model.repo_id} ({human_size(model.size_bytes)}) at {model.path}? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print()
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Not deleted.")
            return 1
    # Unload it first if the running server has it in memory (file handles on Windows).
    health = process.server_health(args.host, args.port)
    if health and str(health.get("model") or "").lower() == model.repo_id.lower():
        try:
            httpx.post(f"{config.server_url(args.host, args.port)}/api/unload", timeout=60, trust_env=False)
        except Exception:
            pass
    try:
        registry.remove_model(model)
    except OSError as e:
        print(red(f"Could not delete {model.path}: {e}"))
        return 1
    print(f"Deleted {model.repo_id}.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    m = registry.find_model(args.model)
    rows = [
        ("Model", m.repo_id),
        ("Path", str(m.path)),
        ("Size", human_size(m.size_bytes)),
        ("Type", m.model_type or "-"),
        ("Architectures", ", ".join(m.architectures) or "-"),
        ("Revision", m.revision or "-"),
        ("Downloaded", f"{_fmt_date(m.downloaded_at)}  ({m.downloaded_at})" if m.downloaded_at else "-"),
        ("Remote code", "yes (trust_remote_code)" if m.needs_remote_code else "no"),
        ("Chat template", "yes" if m.has_chat_template else "no"),
    ]
    for k, v in rows:
        print(f"{k:<14}: {v}")
    print(f"Files ({len(m.files)}):")
    for name in m.files:
        try:
            size = human_size((m.path / name).stat().st_size)
        except OSError:
            size = "?"
        print(f"  {size:>10}  {name}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    url = config.server_url(args.host, args.port)
    if process.is_server_running(args.host, args.port):
        print(f"A server is already running at {url}")
        return 1
    if process.port_in_use(args.host, args.port):
        print(red(f"Port {args.port} on {args.host} is in use by another program (not an hfserve server). "
                  f"Use --port to choose a different one."))
        return 1
    from .server import run_server

    run_server(args.host, args.port, args.model)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    host, port = args.host, args.port
    url = config.server_url(host, port)

    # 1. resolve (pulling first when the user gave a URL / org/name that is not installed)
    ref = args.model
    try:
        model = registry.find_model(ref)
    except ModelNotFound as e:
        if not registry.looks_like_ref(ref):
            print(red(f"Error: {e}"))
            return 1
        print("Model not installed, pulling it first...")
        registry.pull_model(ref)
        model = registry.find_model(ref)

    if model.incomplete:
        print("Model download is incomplete, resuming it first...")
        registry.pull_model(model.repo_id)
        model = registry.find_model(model.repo_id)

    options = {name: value for name in GEN_OPTIONS if (value := getattr(args, name, None)) is not None}
    proc = None
    result = None
    rc = 0
    try:
        # 2. server (inside the try so Ctrl+C while it boots still tears it down)
        if process.server_health(host, port):
            print(f"Connected to running server at {url}")
        elif process.port_in_use(host, port):
            print(red(f"Error: port {port} on {host} is in use by another program (not an hfserve server). "
                      f"Use --port to choose a different one."))
            return 1
        else:
            print("Starting server ...")
            proc = process.spawn_server(host, port)
            if not process.wait_for_server(host, port, proc=proc):
                print(red("Server failed to start"))
                tail = process.log_tail(30)
                if tail:
                    print(dim(tail))
                print(dim(f"server log: {config.LOG_FILE}"))
                return 1

        # 3. banner
        print(f"Serving {bold(model.repo_id)} at {url}")
        print(dim(API_LINE))

        # 4. load
        print("Loading model ... ", end="", flush=True)
        try:
            r = httpx.post(f"{url}/api/load", json={"model": model.repo_id}, timeout=900, trust_env=False)
        except httpx.HTTPError as e:
            print()
            print(red(f"Error: cannot reach the server: {e}"))
            _print_log_hint()
            return 1
        if r.status_code != 200:
            print()
            print(red(f"Error: {error_text(r)}"))
            _print_log_hint()
            return 1
        info = _json(r)
        vram = info.get("vram_bytes") or 0
        print(f"done: {info.get('device')} / {info.get('dtype')} in {float(info.get('load_seconds') or 0):.1f}s"
              + (f" (VRAM {human_size(vram)})" if vram else ""))

        # 5. chat
        result = chat.run_chat(host, port, info.get("model") or model.repo_id, system=args.system,
                               options=options, think=not args.no_think)
        if result == "error":
            rc = 1
    finally:
        # 6. always stop the server, whatever happened above
        if result != "bye":
            print("Stopping server ...")
        n = process.stop_server(host, port, proc=proc, log=_quiet)
        _report_stopped(n, port)
    return rc


def cmd_gui(args: argparse.Namespace) -> int:
    from . import gui  # tkinter only when the window is actually wanted

    return gui.main(args.host, args.port, args.model)


def cmd_stop(args: argparse.Namespace) -> int:
    was_running = process.is_server_running(args.host, args.port)
    n = process.stop_server(args.host, args.port, log=_quiet, all_servers=args.all)
    if not was_running and n == 0:
        print("No hfserve server is running" + ("." if args.all else f" on port {args.port}."))
        return 0
    _report_stopped(n, None if args.all else args.port)
    return 0


def _report_stopped(killed: int, port: int | None) -> None:
    """Tell the user the server is gone; warn loudly if any hfserve process survived."""
    left = process.find_server_processes(port)
    if left:
        pids = ", ".join(str(p.pid) for p in left)
        print(red(f"Warning: {len(left)} hfserve server process(es) could not be terminated (pid {pids})."))
        return
    how = f"killed {killed} process(es)" if killed else "graceful shutdown"
    print(f"Server stopped ({how}); no hfserve processes left running. Bye.")


def cmd_ps(args: argparse.Namespace) -> int:
    url = config.server_url(args.host, args.port)
    health = process.server_health(args.host, args.port)
    if not health:
        print(f"Server not running ({url})")
        return 1
    print(f"Server   : {url}  (pid {health.get('pid')}, hfserve {health.get('version')})")
    try:
        r = httpx.get(f"{url}/api/ps", timeout=5, trust_env=False)
    except httpx.HTTPError as e:
        print(red(f"Error: {e}"))
        return 1
    if r.status_code != 200:
        print(red(f"Error: {error_text(r)}"))
        return 1
    models = _json(r).get("models") or []
    if not models:
        print("Model    : (none loaded)")
        return 0
    for m in models:
        vram = m.get("vram_bytes") or 0
        print(f"Model    : {m.get('name') or m.get('model')}")
        print(f"Device   : {m.get('device')}")
        print(f"Dtype    : {m.get('dtype')}")
        print(f"VRAM     : {human_size(vram) if vram else '-'}")
        print(f"Loaded   : {_fmt_date(m.get('loaded_at'))}  (took {float(m.get('load_seconds') or 0):.1f}s)")
    return 0


# --------------------------------------------------------------------------- shell
def _split_line(line: str) -> list[str]:
    """shlex with quotes but without backslash escapes, so Windows paths survive."""
    lex = shlex.shlex(line, posix=True)
    lex.whitespace_split = True
    lex.escape = ""
    lex.commenters = ""
    return list(lex)


def _server_status(host: str, port: int) -> str:
    health = process.server_health(host, port)
    if not health:
        return "not running"
    status = f"running, pid {health.get('pid')}"
    if health.get("model"):
        status += f", model {health['model']}"
    return status


def print_shell_banner(host: str, port: int) -> None:
    print(f"{bold(PROG)} {__version__} — local HuggingFace model server")
    print(f"Models directory : {config.MODELS_DIR}")
    print(f"Server           : {config.server_url(host, port)} ({_server_status(host, port)})")
    print()
    models = registry.list_models()
    if models:
        print("Installed models:")
        width = max(len(m.repo_id) for m in models)
        for i, m in enumerate(models, 1):
            print(f"  {i:>2}  {m.repo_id:<{width}}  {human_size(m.size_bytes):>8}   {m.model_type or ''}")
    else:
        print(f"No models installed. Download one with: pull {EXAMPLE_URL}")
    print()
    print(f"Commands: {SHELL_COMMANDS}")


def run_shell(parser: argparse.ArgumentParser, host: str, port: int) -> int:
    print_shell_banner(host, port)
    parser.set_defaults(host=host, port=port)  # shell commands inherit the shell's --host/--port
    while True:
        try:
            line = input(f"{PROG}> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            tokens = _split_line(line)
        except ValueError as e:
            print(red(f"Error: {e}"))
            continue
        if not tokens:
            continue
        word = tokens[0].lower()
        if word in ("quit", "exit", "q", "bye"):
            return 0
        if word in ("help", "?", "h"):
            try:
                if len(tokens) > 1:
                    parser.parse_args([tokens[1], "--help"])
                else:
                    parser.print_help()
                    print(f"\nShell: {SHELL_COMMANDS}")
            except SystemExit:
                pass
            continue
        if word == "shell":
            print("You are already in the shell.")
            continue
        try:
            args = parser.parse_args(tokens)
        except SystemExit:  # argparse printed its usage/error (or --version)
            continue
        if args.command is None:
            parser.print_help()
            continue
        try:
            args.func(args)
        except KeyboardInterrupt:
            print()
        except ModelError as e:
            print(red(f"Error: {e}"))
        except httpx.HTTPError as e:
            print(red(f"Error: {e}"))
        except Exception as e:  # keep the shell alive; HFSERVE_DEBUG=1 shows the traceback
            print(red(f"Error: {type(e).__name__}: {e}"))
            if os.environ.get("HFSERVE_DEBUG"):
                traceback.print_exc()


def cmd_shell(args: argparse.Namespace) -> int:
    return run_shell(args.parser, args.host, args.port)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Serve HuggingFace transformers models locally, Ollama style.",
        epilog=(
            "examples:\n"
            f"  {PROG} pull {EXAMPLE_URL}\n"
            f"  {PROG} run EuroLLM-1.7B-Instruct\n"
            f"  {PROG} run 1 --no-think --temperature 0.2\n"
            f"  {PROG}                    (no command: interactive shell)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default=config.DEFAULT_HOST, help=f"server address (default: {config.DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=config.DEFAULT_PORT, help=f"server port (default: {config.DEFAULT_PORT})")
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>", title="commands")

    sp = sub.add_parser("list", aliases=["ls"], help="list installed models")
    sp.add_argument("--json", action="store_true", help="print the models.json index instead of a table")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("pull", help="download a model from huggingface.co")
    sp.add_argument("ref", metavar="URL_OR_REPO_ID", help="huggingface.co URL or org/name[@revision]")
    sp.add_argument("--revision", help="branch, tag or commit to download")
    sp.add_argument("--force", action="store_true", help="re-download files even if present")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("rm", aliases=["remove"], help="delete an installed model")
    sp.add_argument("model", help="org/name, name, URL or list index")
    sp.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    sp.set_defaults(func=cmd_rm)

    sp = sub.add_parser("show", help="show details of an installed model")
    sp.add_argument("model", help="org/name, name, URL or list index")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("serve", help="run the API server in the foreground")
    sp.add_argument("--model", help="model to load at start-up")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("run", help="start the server (if needed), load a model and chat with it")
    sp.add_argument("model", help="org/name, name, URL or list index (pulled first when not installed)")
    sp.add_argument("--system", metavar="TEXT", help="system prompt")
    sp.add_argument("--temperature", type=float, metavar="F")
    sp.add_argument("--top-p", type=float, metavar="F")
    sp.add_argument("--top-k", type=int, metavar="N")
    sp.add_argument("--num-predict", type=int, metavar="N", help="maximum tokens per answer")
    sp.add_argument("--repeat-penalty", type=float, metavar="F")
    sp.add_argument("--seed", type=int, metavar="N")
    sp.add_argument("--reasoning-effort", choices=chat.REASONING_EFFORTS,
                    help="how long the model thinks (high = longest, low = fastest answers)")
    sp.add_argument("--no-think", action="store_true",
                    help="hide reasoning and ask the model not to think (when it supports that)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("gui", help="open a chat window (starts the server if needed)")
    sp.add_argument("--model", help="model to load at start-up (default: the only/loaded one)")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("stop", help="stop the running server (and everything it spawned)")
    sp.add_argument("--all", action="store_true", help="stop every hfserve server on this machine, on any port")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("ps", help="show the running server and its loaded model")
    sp.set_defaults(func=cmd_ps)

    sp = sub.add_parser("shell", help="interactive command shell (the default)")
    sp.set_defaults(func=cmd_shell)

    p.set_defaults(parser=p)
    return p


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            return run_shell(parser, args.host, args.port)
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print()
        return 130
    except ModelError as e:
        print(red(f"Error: {e}"))
        return 1
    except httpx.HTTPError as e:
        print(red(f"Error: {e}"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
