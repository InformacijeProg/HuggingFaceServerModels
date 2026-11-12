"""Retrieve context for a question and build the grounded chat messages."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config, store

_MC_OPTIONS = re.compile(r"\s+A\)\s.+$", re.DOTALL)


@dataclass
class Hit:
    text: str
    source: str
    similarity: float


def search(question: str, top_k: int = config.TOP_K) -> list[Hit]:
    col = store.collection()
    if col.count() == 0:
        return []
    # A)-D) options dilute the embedding; retrieve with the bare question.
    query = _MC_OPTIONS.sub("", question).strip()
    res = col.query(query_embeddings=[store.embed_query(query)], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(Hit(text=text, source=(meta or {}).get("source", "?"), similarity=1.0 - dist))
    return hits


def build_messages(question: str, hits: list[Hit] | None = None) -> tuple[list[dict], list[Hit]]:
    """Few-shot chat messages, routed by retrieval confidence.

    A strong match (>= MIN_SIMILARITY) grounds the answer in those chunks only; weaker hits
    are left out entirely, because similar-but-different quiz questions act as distractors
    that a small model copies wrong answers from. With no strong match the model answers
    from its own knowledge, with a demonstration that obscure questions get "Ne znam."."""
    if hits is None:
        hits = search(question)
    used = [h for h in hits if h.similarity >= config.MIN_SIMILARITY]
    if not used:
        messages = ([{"role": "system", "content": config.SYSTEM_PROMPT}]
                    + list(config.FEW_SHOT_PLAIN)
                    + [{"role": "user", "content": question}])
        return messages, []
    parts: list[str] = []
    total = 0
    for h in used:
        if total + len(h.text) > config.MAX_CONTEXT_CHARS and parts:
            break
        parts.append("- " + h.text)
        total += len(h.text)
    user = f"Kontekst:\n{chr(10).join(parts)}\n\nPitanje: {question}"
    messages = ([{"role": "system", "content": config.SYSTEM_PROMPT}]
                + list(config.FEW_SHOT_GROUNDED)
                + [{"role": "user", "content": user}])
    return messages, used[:len(parts)]
