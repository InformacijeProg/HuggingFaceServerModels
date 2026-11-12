"""hfrag command line: ingest data into the index and test retrieval.

    python -m hfrag ingest-qa [--dir training/QA]     question-answer jsonl -> index
    python -m hfrag ingest-chunks <file.jsonl>        {"text", "source"} records -> index
    python -m hfrag query "Pitanje?"                  show what retrieval returns
    python -m hfrag stats                             index size
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, retrieve, store


def iter_qa_dir(folder: Path):
    for f in sorted(folder.glob("*.jsonl")):
        if f.stem.lower() in ("readme", "sources"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            q, a = rec.get("question"), rec.get("answer")
            if q and a:
                yield f"Pitanje: {q}\nOdgovor: {a}", {"source": f"qa:{f.stem}"}


def iter_chunks_file(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = (rec.get("text") or "").strip()
        if len(text) >= 80:
            yield text, {"source": str(rec.get("source") or path.stem)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hfrag", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sp = sub.add_parser("ingest-qa", help="index question-answer jsonl files")
    sp.add_argument("--dir", default=None, help="folder with *.jsonl (default: training/QA)")
    sp = sub.add_parser("ingest-chunks", help='index {"text","source"} jsonl records')
    sp.add_argument("file", nargs="+")
    sp = sub.add_parser("query", help="show retrieval for a question")
    sp.add_argument("text")
    sp.add_argument("--top-k", type=int, default=config.TOP_K)
    sub.add_parser("stats", help="index size")
    args = p.parse_args(argv)

    if args.command == "ingest-qa":
        from hftrain import config as tconfig

        folder = Path(args.dir) if args.dir else tconfig.QA_DIR
        n = store.add_documents(iter_qa_dir(folder), progress=print)
        print(f"{n:,} QA chunks indexed from {folder} -> {config.INDEX_DIR}")
    elif args.command == "ingest-chunks":
        total = 0
        for name in args.file:
            n = store.add_documents(iter_chunks_file(Path(name)), progress=print)
            print(f"{name}: {n:,} chunks")
            total += n
        print(f"{total:,} chunks indexed -> {config.INDEX_DIR}")
    elif args.command == "query":
        hits = retrieve.search(args.text, top_k=args.top_k)
        for h in hits:
            print(f"[{h.similarity:.3f}] ({h.source}) {h.text[:140]!r}")
        messages, used = retrieve.build_messages(args.text, hits)
        print(f"\ncontext chunks used: {len(used)}; system: {messages[0]['content'][:80]}...")
    elif args.command == "stats":
        print(f"{store.collection().count():,} chunks in {config.INDEX_DIR}")
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
