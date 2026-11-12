"""hffiles command line: look at a file, run a plan, or let the model write one.

Works on Excel (.xlsx), Word (.docx), PowerPoint (.pptx) and PDF; the format follows from
the extension.

    python -m hffiles describe <file> [--json]        what is in it (the model's view)
    python -m hffiles ops [--kind excel|word|slides|pdf] [--all]
    python -m hffiles copy <src> [dst]                byte-for-byte copy
    python -m hffiles upload <file>                   copy into files/in
    python -m hffiles new <name.xlsx> [--csv data.csv --sheet Podaci]
    python -m hffiles pdf <file> [target.pdf] [--prefer office|libreoffice]
    python -m hffiles apply <plan.json|-> [--source f] [--target t] [--loose] [--pdf]
    python -m hffiles ask "dodaj stupac PDV" [--file f] [--dry-run] [--pdf]
    python -m hffiles where                           the workspace folders and renderers

``ask`` is the whole loop headlessly: the running hfserve server turns the request
into a plan, this process validates it and does the work. ``--dry-run`` stops after
the plan, which is the honest way to see what the model actually produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, files, ops, plan
from .errors import FileError

ASK_OPTIONS = {"temperature": 0.1, "num_predict": 900, "repeat_penalty": 1.05}


def _ask_model(request: str, describe: dict | None, *, host: str, port: int,
               model: str | None, all_ops: bool, timeout: float,
               kind: str = "excel") -> tuple[str, list[dict]]:
    """Send the plan prompt to a running hfserve server and return (answer, messages)."""
    import httpx

    from hfserve import config as serve_config
    from hfserve import process

    messages = plan.build_messages(request, describe, kind=kind, all_ops=all_ops)
    base = serve_config.server_url(process.connect_host(host), port)
    with httpx.Client(base_url=base, trust_env=False, timeout=timeout) as client:
        if not model:
            health = process.server_health(host, port)
            if health is None:
                raise FileError(f"no hfserve server at {base} - start one with: hfserve serve")
            model = str(health.get("model") or "")
            if not model:
                raise FileError("the server has no model loaded - run: hfserve load <model>")
        r = client.post("/api/chat", json={"model": model, "messages": messages,
                                          "stream": False, "think": False,
                                          "options": ASK_OPTIONS})
        if r.status_code != 200:
            raise FileError(f"the server answered {r.status_code}: {r.text[:300]}")
        body = r.json() or {}
    return str((body.get("message") or {}).get("content") or ""), messages


def _print_result(result: ops.PlanResult) -> int:
    print(result.summary())
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hffiles", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sp = sub.add_parser("describe", help="what is in a workbook, document, deck or PDF")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true", help="full description instead of the digest")

    sp = sub.add_parser("ops", help="the operations a plan may use")
    sp.add_argument("--kind", choices=config.KINDS, default="excel")
    sp.add_argument("--all", action="store_true", help="every operation, not just the core set")

    sp = sub.add_parser("copy", help="byte-for-byte copy (keeps charts, macros, pivot tables)")
    sp.add_argument("source")
    sp.add_argument("target", nargs="?")

    sp = sub.add_parser("upload", help="copy a file into files/in")
    sp.add_argument("file")

    sp = sub.add_parser("new", help="create a workbook, optionally from a csv")
    sp.add_argument("name")
    sp.add_argument("--csv", help="csv file to import into the first sheet")
    sp.add_argument("--sheet", default="Podaci", help="name of the first sheet")

    sp = sub.add_parser("pdf", help="export a file to PDF (Office or LibreOffice)")
    sp.add_argument("file")
    sp.add_argument("target", nargs="?")
    sp.add_argument("--prefer", choices=("office", "libreoffice"))

    sp = sub.add_parser("apply", help="run a plan from a json file (or - for stdin)")
    sp.add_argument("plan")
    sp.add_argument("--source", help="workbook to edit (omit to create a new one)")
    sp.add_argument("--target", help="where to write (default: files/out)")
    sp.add_argument("--loose", action="store_true",
                    help="keep going after a failed operation instead of saving nothing")
    sp.add_argument("--pdf", action="store_true", help="also export the result to PDF")

    sp = sub.add_parser("ask", help="let the model plan the work, then do it")
    sp.add_argument("request")
    sp.add_argument("--file", help="workbook to work on (omit to create a new one)")
    sp.add_argument("--model", help="model name (default: whatever the server has loaded)")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--all-ops", action="store_true", help="show the model every operation")
    sp.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    sp.add_argument("--show-prompt", action="store_true", help="print the prompt that was sent")
    sp.add_argument("--kind", choices=config.KINDS, default=None,
                    help="format to create when no --file is given (default: excel)")
    sp.add_argument("--pdf", action="store_true", help="also export the result to PDF")
    sp.add_argument("--timeout", type=float, default=300.0)

    sub.add_parser("where", help="print the workspace folders")

    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0

    try:
        if args.command == "describe":
            desc = files.describe(args.file)
            print(json.dumps(desc, ensure_ascii=False, indent=2) if args.json
                  else files.digest_text(desc))
            return 0

        if args.command == "ops":
            print(ops.reference(None if args.all else plan.CORE_OPS[args.kind],
                                kind=args.kind))
            return 0

        if args.command == "copy":
            print(files.copy_file(args.source, args.target))
            return 0

        if args.command == "upload":
            print(files.upload(args.file))
            return 0

        if args.command == "pdf":
            print(files.to_pdf(args.file, args.target, prefer=args.prefer))
            return 0

        if args.command == "new":
            steps: list[dict] = [{"op": "rename_sheet", "to": args.sheet}]
            if args.csv:
                steps += [{"op": "import_csv", "path": args.csv},
                          {"op": "format_range", "range": "1", "bold": True},
                          {"op": "autofit"}]
            return _print_result(ops.apply_plan({"save_as": args.name, "ops": steps}))

        if args.command == "apply":
            text = sys.stdin.read() if args.plan == "-" else Path(args.plan).read_text(encoding="utf-8")
            try:
                raw = json.loads(text)
            except ValueError as e:
                raise FileError(f"{args.plan} is not valid JSON: {e}") from e
            if args.pdf and isinstance(raw, dict):
                raw = dict(raw, export_pdf=True)
            return _print_result(ops.apply_plan(raw, source=args.source, target=args.target,
                                                strict=not args.loose))

        if args.command == "ask":
            from hfserve import config as serve_config

            host = args.host or serve_config.DEFAULT_HOST
            port = args.port or serve_config.DEFAULT_PORT
            desc = files.describe(args.file) if args.file else None
            kind = args.kind or (desc["kind"] if desc else "excel")
            answer, messages = _ask_model(args.request, desc, host=host, port=port,
                                          model=args.model, all_ops=args.all_ops,
                                          timeout=args.timeout, kind=kind)
            if args.show_prompt:
                for m in messages:
                    print(f"--- {m['role']} ---\n{m['content']}")
            raw = plan.extract_plan(answer)
            if raw is None:
                print(f"the model did not answer with a plan:\n{answer.strip()[:800]}")
                return 1
            if args.pdf and isinstance(raw, dict):
                raw = dict(raw, export_pdf=True)
            print("plan: " + json.dumps(raw, ensure_ascii=False))
            if args.dry_run:
                ops.parse_plan(raw)  # validate, write nothing
                print("plan is valid (dry run, nothing written)")
                return 0
            return _print_result(ops.apply_plan(raw, source=args.file))

        if args.command == "where":
            config.ensure_dirs()
            print(f"workspace: {config.FILES_DIR}")
            print(f"uploads:   {config.IN_DIR}")
            print(f"results:   {config.OUT_DIR}")
            print(f"formats:   {', '.join(config.ALL_SUFFIXES)}")
            have = files.pdf_backends()
            renderers = [name for name, ok in (("Microsoft Office", have["office"]),
                                               ("LibreOffice", have["libreoffice"])) if ok]
            print("PDF export: " + (", ".join(renderers) if renderers
                                    else "no renderer found (install LibreOffice)"))
            return 0
    except FileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
