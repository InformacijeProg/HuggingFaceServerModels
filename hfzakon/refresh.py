"""Incremental refresh of the hr-zakoni index: only what is new gets fetched and embedded.

Narodne novine is append-only (past editions never change), so a refresh pulls just the
editions beyond the last recorded one, chunks those documents, and upserts that delta —
minutes after months of staleness, versus ~an hour for a full re-embed. New consolidated
laws on zakon.hr get fresh sequential ids, so a cheap tail probe of the id space finds
them. Progress flows through a callback so the GUI can show a live status row.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

from . import config, fetch, nn, zakon_hr

STATE_FILE = config.STATE_DIR / "last_refresh.json"
LOCK_FILE = config.STATE_DIR / "refresh.lock"
LOCK_STALE_S = 2 * 3600


def nn_years() -> list[int]:
    return list(range(config.NN_YEARS[0], _dt.date.today().year + 1))


def read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"ts": "", "nn": {}}


def write_state(state: dict) -> None:
    config.ensure_dirs()
    state["ts"] = _dt.datetime.now().isoformat(timespec="seconds")
    STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def plan() -> dict[int, tuple[int, int]]:
    """{year: (recorded_max, live_max)} for every gazette year (live from the site)."""
    state = read_state()
    out = {}
    for year in nn_years():
        recorded = int(state.get("nn", {}).get(str(year), 0))
        if not recorded:  # never recorded: trust the cached search pages
            cached = sorted(config.RAW_NN_SEARCH_DIR.glob(f"{year}_*.html"))
            recorded = int(cached[-1].stem.split("_")[1]) if cached else 0
        out[year] = (recorded, nn.editions(year))
    return out


def _locked() -> bool:
    if not LOCK_FILE.exists():
        return False
    try:
        return time.time() - LOCK_FILE.stat().st_mtime < LOCK_STALE_S
    except OSError:
        return False


def maybe_run(progress=None, min_age_h: int = 24) -> dict | None:
    """run() unless the last refresh is younger than min_age_h (the GUI startup path)."""
    st = read_state()
    if st.get("ts"):
        try:
            age = _dt.datetime.now() - _dt.datetime.fromisoformat(st["ts"])
            if age < _dt.timedelta(hours=min_age_h):
                return None
        except ValueError:
            pass
    return run(progress)


def run(progress=None) -> dict:
    """Check and refresh. Returns a summary dict; progress gets dict events:

    {"stage": "check"|"laws"|"pull"|"index"|"done"|"current"|"error",
     "done": int, "total": int, "detail": str, "eta_s": float|None}
    """
    def emit(stage, done=0, total=0, detail="", eta_s=None):
        if progress:
            progress({"stage": stage, "done": done, "total": total,
                      "detail": detail, "eta_s": eta_s})

    config.ensure_dirs()
    if _locked():
        emit("current", detail="osvježavanje već u tijeku")
        return {"skipped": "locked"}
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _run(emit)
    except Exception as exc:  # noqa: BLE001 - surfaced to the GUI, old index stays usable
        emit("error", detail=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass


def _run(emit) -> dict:
    emit("check", detail="provjera novih izdanja")
    levels = plan()
    todo = {y: (rec, live) for y, (rec, live) in levels.items() if live > rec}
    new_laws = zakon_hr.discover_new()

    delta: list[dict] = []  # chunk records to embed

    # New consolidated laws from the id tail probe.
    if new_laws:
        emit("laws", done=0, total=len(new_laws), detail=f"{len(new_laws)} novih zakona")
        with config.ZAKON_CHUNKS.open("a", encoding="utf-8") as out:
            for k, lid in enumerate(new_laws, 1):
                path = config.RAW_ZAKON_DIR / "laws" / f"{lid}.html"
                law = zakon_hr.parse_law(path.read_bytes(), f"{config.ZAKON_BASE}/z/{lid}/x")
                if not law or not law["articles"]:
                    continue
                law["id"] = str(lid)
                for cid, text, meta in zakon_hr.chunk_law(law):
                    if len(text) < config.MIN_CHUNK_CHARS:
                        continue
                    rec = {"id": cid, "text": text, "meta": meta}
                    delta.append(rec)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                emit("laws", done=k, total=len(new_laws), detail=law["title"][:60])

    # New gazette editions, pulled and chunked per edition.
    total_eds = sum(live - rec for rec, live in todo.values())
    done_eds = new_docs = 0
    for year, (recorded, live) in sorted(todo.items()):
        chunk_path = config.CHUNKS_DIR / config.NN_CHUNKS_TPL.format(year=year)
        with chunk_path.open("a", encoding="utf-8") as out:
            for broj in range(recorded + 1, live + 1):
                emit("pull", done=done_eds, total=total_eds, detail=f"NN {broj}/{year}")
                page = fetch.get_cached(config.NN_SEARCH.format(year=year, broj=broj),
                                        nn.search_cache(year, broj))
                for link in nn.doc_links(page) if page else []:
                    url = config.NN_BASE + link
                    if fetch.nn_disallowed(url):
                        continue
                    cache = nn.doc_cache(link)
                    fresh = not cache.exists()
                    html = fetch.get_cached(url, cache)
                    if not html or not fresh:
                        continue
                    doc = nn.parse_doc(html, url)
                    if not doc:
                        continue
                    new_docs += 1
                    for cid, text, meta in nn.chunk_doc(doc, cache.stem):
                        if len(text) < config.MIN_CHUNK_CHARS:
                            continue
                        rec = {"id": cid, "text": text, "meta": meta}
                        delta.append(rec)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done_eds += 1

    # Embed and upsert the delta, with a measured ETA.
    if delta:
        from hfrag import store as rag_store
        from . import store

        col = store.collection()
        t0 = time.monotonic()
        for i in range(0, len(delta), 128):
            batch = delta[i:i + 128]
            col.upsert(ids=[r["id"] for r in batch],
                       documents=[r["text"] for r in batch],
                       metadatas=[store._clean_meta(r["meta"]) for r in batch],
                       embeddings=rag_store.embed_passages([r["text"] for r in batch]))
            done = i + len(batch)
            rate = done / max(time.monotonic() - t0, 0.1)
            emit("index", done=done, total=len(delta),
                 detail=f"{done:,}/{len(delta):,} odlomaka", eta_s=(len(delta) - done) / max(rate, 1))

    state = read_state()
    state.setdefault("nn", {})
    for year, (_rec, live) in levels.items():
        if live:
            state["nn"][str(year)] = live
    write_state(state)

    latest_year = max((y for y in levels if levels[y][1]), default=None)
    tag = f"NN {levels[latest_year][1]}/{latest_year}" if latest_year else ""
    if delta:
        emit("done", done=len(delta), total=len(delta),
             detail=f"{len(delta):,} novih odlomaka ({tag})")
    else:
        emit("current", detail=f"baza ažurna ({tag})")
    return {"new_laws": len(new_laws), "new_editions": total_eds,
            "new_docs": new_docs, "new_chunks": len(delta)}
