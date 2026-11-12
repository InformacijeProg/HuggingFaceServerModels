"""hfzakon command line: scrape Croatian legislation and index it for RAG.

    python -m hfzakon pull-zakon [--limit N]      download all consolidated laws (zakon.hr)
    python -m hfzakon pull-nn [--years 2025 2026] download the official gazette (narodne-novine)
    python -m hfzakon chunk [zakon|nn|all]        parse cached pages -> chunk jsonl files
    python -m hfzakon ingest [zakon|nn|all]       chunk files -> Chroma collection hr-zakoni
    python -m hfzakon query "Pitanje?"            show what legislation retrieval returns
    python -m hfzakon stats                       collection size

Scrapes are resumable: every page is cached under rag/zakoni/raw/, re-runs fetch
only what is missing (e.g. new gazette editions).
"""
from __future__ import annotations

import argparse
import sys

from . import config


def _chunk_files(what: str):
    files = []
    if what in ("zakon", "all") and config.ZAKON_CHUNKS.exists():
        files.append(config.ZAKON_CHUNKS)
    if what in ("nn", "all"):
        for year in config.NN_YEARS:
            p = config.CHUNKS_DIR / config.NN_CHUNKS_TPL.format(year=year)
            if p.exists():
                files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hfzakon", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sp = sub.add_parser("pull-zakon", help="download all law pages from zakon.hr")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--refetch-empty", action="store_true",
                    help="re-download cached pages that parsed to no articles")
    sub.add_parser("discover-zakon", help="probe law ids the stale sitemap misses")
    sub.add_parser("refresh", help="incremental update: new NN editions + new laws -> index")
    sp = sub.add_parser("pull-nn", help="download gazette search+document pages")
    sp.add_argument("--years", type=int, nargs="+", default=list(config.NN_YEARS))
    sp = sub.add_parser("chunk", help="parse cached pages into chunk jsonl")
    sp.add_argument("what", nargs="?", default="all", choices=["zakon", "nn", "all"])
    sp = sub.add_parser("ingest", help="index chunk files into Chroma (hr-zakoni)")
    sp.add_argument("what", nargs="?", default="all", choices=["zakon", "nn", "all"])
    sp = sub.add_parser("query", help="show retrieval for a question")
    sp.add_argument("text")
    sp.add_argument("--top-k", type=int, default=5)
    sub.add_parser("stats", help="collection size")
    args = p.parse_args(argv)

    if args.command == "pull-zakon":
        from . import zakon_hr

        fetched, total = zakon_hr.pull(limit=args.limit, refetch_empty=args.refetch_empty)
        print(f"zakon.hr: {total} laws cached ({fetched} newly fetched)")
    elif args.command == "discover-zakon":
        from . import zakon_hr

        found = zakon_hr.discover_ids()
        print(f"zakon.hr id discovery: {found} new law pages (rerun chunk+ingest to index them)")
    elif args.command == "refresh":
        from . import refresh

        def show(ev):
            eta = f" (~{int(ev['eta_s'])}s)" if ev.get("eta_s") else ""
            bar = f" {ev['done']}/{ev['total']}" if ev["total"] else ""
            print(f"  [{ev['stage']}]{bar} {ev['detail']}{eta}")

        summary = refresh.run(progress=show)
        print(summary)
    elif args.command == "pull-nn":
        from . import nn

        fetched, total = nn.pull(years=args.years)
        print(f"narodne-novine: {total} documents cached ({fetched} newly fetched)")
    elif args.command == "chunk":
        total = 0
        if args.what in ("zakon", "all"):
            from . import zakon_hr

            total += zakon_hr.chunk_all()
        if args.what in ("nn", "all"):
            from . import nn

            total += nn.chunk_all()
        print(f"{total:,} chunks written to {config.CHUNKS_DIR}")
    elif args.command == "ingest":
        from . import store

        files = _chunk_files(args.what)
        if not files:
            print("no chunk files found; run `python -m hfzakon chunk` first")
            return 1
        total = 0
        for f in files:
            total += store.ingest_file(f)
        print(f"{total:,} chunks indexed into '{config.COLLECTION}' "
              f"({store.collection().count():,} total)")
    elif args.command == "query":
        from . import store

        for h in store.search(args.text, top_k=args.top_k):
            src = h.meta.get("law") or h.meta.get("title") or "?"
            loc = f" cl.{h.meta['clanak']}" if h.meta.get("clanak") else ""
            print(f"[{h.similarity:.3f}] {src}{loc}: {h.text[:140]!r}")
    elif args.command == "stats":
        from . import store

        print(f"{store.collection().count():,} chunks in '{config.COLLECTION}' "
              f"at {config.ZAKONI_DIR.parent / 'index'}")
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
