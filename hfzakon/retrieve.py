"""Strict grounded answering over hr-zakoni: context-only answers, otherwise "Ne znam.".

Unlike hfrag (quiz mode falls back to the model's own knowledge), legislation answers
must never be improvised by a 1.7B model: every answer is grounded in retrieved law
text and cites its source, and without a confident hit the model is shown to say
"Ne znam.".
"""
from __future__ import annotations

import re

from . import store

CLANAK_QUERY_RE = re.compile(r"\bčlan(?:ak|ka|ku)?\s+(\d+)\.?\s*([a-zčćđšž])?\b", re.IGNORECASE)

TOP_K = 6
# Law questions are paraphrases of article text, not near-duplicates like quiz QA, so
# the bar sits lower than hfrag's 0.88. Measured on the live index (2026-10-16):
# on-topic article hits score 0.85-0.90, out-of-domain questions still reach 0.81-0.83
# (e5-small compresses the range), so 0.84 is the separation point.
MIN_SIMILARITY = 0.84
MAX_CONTEXT_CHARS = 3000

SYSTEM_PROMPT = (
    "Ti si pravni asistent za hrvatske zakone i Narodne novine. Odgovaraj kratko i "
    "točno, na hrvatskom jeziku, isključivo na temelju navedenog konteksta. Uz odgovor "
    "navedi izvor (zakon i članak, ili broj Narodnih novina). Ako odgovora nema u "
    "kontekstu, reci: Ne znam.")

FEW_SHOT = [
    {"role": "user", "content":
        "Kontekst:\n- Zakon o cestama — Članak 3. (Javne ceste)\n"
        "(1) Javne ceste dijele se na autoceste, državne ceste, županijske ceste i lokalne ceste.\n\n"
        "Pitanje: Kako se dijele javne ceste?"},
    {"role": "assistant", "content":
        "Javne ceste dijele se na autoceste, državne ceste, županijske ceste i lokalne "
        "ceste (Zakon o cestama, čl. 3.)."},
    {"role": "user", "content":
        "Kontekst:\n- Zakon o cestama — Članak 3. (Javne ceste)\n"
        "(1) Javne ceste dijele se na autoceste, državne ceste, županijske ceste i lokalne ceste.\n\n"
        "Pitanje: Koliko iznosi porez na dobit?"},
    {"role": "assistant", "content": "Ne znam."},
]


def build_messages(question: str, hits: list[store.Hit] | None = None) -> tuple[list[dict], list[store.Hit]]:
    """Few-shot chat messages grounded in hr-zakoni. Weak retrieval -> empty context -> "Ne znam."."""
    if hits is None:
        hits = store.search(question, top_k=TOP_K)
    used = [h for h in hits if h.similarity >= MIN_SIMILARITY]
    # "članak N" questions name an exact article: embeddings are weak on numbers, so
    # fetch that article of the best-matching law directly and put it first.
    m = CLANAK_QUERY_RE.search(question)
    law_id = next((h.meta["law_id"] for h in used if h.meta.get("law_id")), None)
    if m and law_id:
        num = m.group(1) + (m.group(2) or "")
        got = store.collection().get(
            where={"$and": [{"law_id": law_id}, {"clanak": num}]}, include=["documents", "metadatas"])
        exact = [store.Hit(text=t, meta=mt or {}, similarity=1.0)
                 for t, mt in zip(got["documents"], got["metadatas"])]
        if exact:
            have = {h.text for h in exact}
            used = exact + [h for h in used if h.text not in have]
    parts: list[str] = []
    total = 0
    for h in used:
        if total + len(h.text) > MAX_CONTEXT_CHARS and parts:
            break
        parts.append("- " + h.text)
        total += len(h.text)
    context = "\n".join(parts) if parts else "(ništa pronađeno)"
    user = f"Kontekst:\n{context}\n\nPitanje: {question}"
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + list(FEW_SHOT)
                + [{"role": "user", "content": user}])
    return messages, used[:len(parts)]
