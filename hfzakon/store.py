"""The hr-zakoni Chroma collection: ingest chunk files with stable ids, query for checks.

Embeddings come from hfrag.store (same multilingual-e5-small, same query/passage
prefixes), so legislation and quiz retrieval stay directly comparable. Ids are stable
(zakon:<law>:cl<N>, nn:<doc>:cl<N>) so a re-scrape upserts changed text in place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hfrag import config as rag_config
from hfrag import store as rag_store

from . import config

_client = None


def collection():
    global _client
    import chromadb

    if _client is None:
        rag_config.ensure_dirs()
        _client = chromadb.PersistentClient(path=str(rag_config.INDEX_DIR))
    return _client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def _clean_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool)) and v != ""}


def ingest_file(path: Path, progress=print) -> int:
    col = collection()
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    n = 0

    def flush():
        nonlocal n
        if not ids:
            return
        col.upsert(ids=list(ids), documents=list(texts), metadatas=list(metas),
                   embeddings=rag_store.embed_passages(texts))
        n += len(ids)
        ids.clear(), texts.clear(), metas.clear()
        if progress:
            progress(f"  {n:,} chunks indexed from {path.name}")

    seen: set[str] = set()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["id"] in seen:          # duplicate ids collide inside one upsert batch
            continue
        seen.add(rec["id"])
        ids.append(rec["id"])
        texts.append(rec["text"])
        metas.append(_clean_meta(rec.get("meta") or {}))
        if len(ids) >= 256:
            flush()
    flush()
    return n


@dataclass
class Hit:
    text: str
    meta: dict
    similarity: float


def search(question: str, top_k: int = 5) -> list[Hit]:
    col = collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[rag_store.embed_query(question)], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    return [Hit(text=t, meta=m or {}, similarity=1.0 - d)
            for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
