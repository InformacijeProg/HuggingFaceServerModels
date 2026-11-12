"""Paths and defaults for the RAG index (sibling of hfserve/hftrain, same HOME)."""
from __future__ import annotations

import os
from pathlib import Path

from hfserve import config as serve_config

RAG_DIR = Path(os.environ.get("HFRAG_HOME", serve_config.HOME_DIR / "rag")).expanduser().resolve()
INDEX_DIR = RAG_DIR / "index"            # Chroma persistent store
COLLECTION = "hr-znanje"

# Multilingual embedder; small enough to share the GPU with the LLM (falls back to CPU).
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_BATCH = 64
# e5 models expect these prefixes on both sides of the search.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

TOP_K = 4
# Router threshold. e5 scores on this index: the exact/near-exact pair scores ~0.887-0.93,
# while similar-but-different questions (dangerous distractors a small model copies from)
# score ~0.84-0.87 (measured 2026-10-16). At or above the bar the answer is grounded in the
# matching chunks; below it the model answers from its own knowledge instead, because weak
# context actively misleads a 1.7B generator.
MIN_SIMILARITY = 0.88
MAX_CONTEXT_CHARS = 2600                 # keep the prompt well inside EuroLLM's context window

SYSTEM_PROMPT = ("Ti si kviz-majstor. Odgovaraj kratko i točno, na hrvatskom jeziku. "
                 "Ako odgovora nema u kontekstu, reci: Ne znam.")
# Worked examples: small models follow demonstrations far better than instructions.
FEW_SHOT_GROUNDED = [
    {"role": "user", "content": "Kontekst:\n- Pitanje: Koji je glavni grad Francuske?\nOdgovor: Pariz.\n\n"
                                "Pitanje: Koji je glavni grad Francuske?"},
    {"role": "assistant", "content": "Pariz."},
    {"role": "user", "content": "Kontekst:\n- Pitanje: Koliko nogu ima pas?\nOdgovor: Četiri.\n\n"
                                "Pitanje: Tko je izumio telefon?"},
    {"role": "assistant", "content": "Ne znam."},
]
# Plain mode (no reliable context): answer from own knowledge, refuse the obscure.
FEW_SHOT_PLAIN = [
    {"role": "user", "content": "Koji je glavni grad Francuske?"},
    {"role": "assistant", "content": "Pariz."},
    {"role": "user", "content": "Koje je godine osnovan šahovski klub 'Plavi konj' iz Senja?"},
    {"role": "assistant", "content": "Ne znam."},
]


def ensure_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
