"""Embedding model + Chroma collection: the shared machinery of ingest and retrieval."""
from __future__ import annotations

import hashlib
import logging
from typing import Iterable

from . import config

log = logging.getLogger("hfrag.store")

_embedder = None
_client = None


def embedder():
    """The sentence-transformers model, loaded once (GPU when free, else CPU)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        log.info("loading embedder %s", config.EMBED_MODEL)
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def collection():
    global _client
    import chromadb

    if _client is None:
        config.ensure_dirs()
        _client = chromadb.PersistentClient(path=str(config.INDEX_DIR))
    return _client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def doc_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_passages(texts: list[str]) -> list[list[float]]:
    vecs = embedder().encode([config.PASSAGE_PREFIX + t for t in texts],
                             batch_size=config.EMBED_BATCH, normalize_embeddings=True,
                             show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    v = embedder().encode([config.QUERY_PREFIX + text], normalize_embeddings=True,
                          show_progress_bar=False)[0]
    return v.tolist()


def add_documents(docs: Iterable[tuple[str, dict]], progress=None) -> int:
    """Upsert (text, metadata) pairs in batches. Returns how many went in."""
    col = collection()
    batch_docs: list[str] = []
    batch_meta: list[dict] = []
    seen: set[str] = set()  # identical texts (same id) would collide inside one upsert batch
    n = 0
    for text, meta in docs:
        i = doc_id(text)
        if i in seen:
            continue
        seen.add(i)
        batch_docs.append(text)
        batch_meta.append(meta)
        if len(batch_docs) >= 512:
            _flush(col, batch_docs, batch_meta)
            n += len(batch_docs)
            if progress:
                progress(f"  {n:,} chunks indexed")
            batch_docs, batch_meta = [], []
    if batch_docs:
        _flush(col, batch_docs, batch_meta)
        n += len(batch_docs)
        if progress:
            progress(f"  {n:,} chunks indexed")
    return n


def _flush(col, texts: list[str], metas: list[dict]) -> None:
    col.upsert(ids=[doc_id(t) for t in texts], documents=texts,
               metadatas=metas, embeddings=embed_passages(texts))
