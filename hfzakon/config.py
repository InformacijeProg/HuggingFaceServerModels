"""Paths, collection name and crawl politeness for the legislation index."""
from __future__ import annotations

from pathlib import Path

from hfrag import config as rag_config

# Everything lives next to the quiz index, under rag/zakoni/.
ZAKONI_DIR = rag_config.RAG_DIR / "zakoni"
RAW_ZAKON_DIR = ZAKONI_DIR / "raw" / "zakon_hr"
RAW_NN_SEARCH_DIR = ZAKONI_DIR / "raw" / "nn" / "search"
RAW_NN_DOC_DIR = ZAKONI_DIR / "raw" / "nn" / "docs"
CHUNKS_DIR = ZAKONI_DIR / "chunks"
STATE_DIR = ZAKONI_DIR / "state"

ZAKON_CHUNKS = CHUNKS_DIR / "zakon_hr.chunks.jsonl"
NN_CHUNKS_TPL = "nn_{year}.chunks.jsonl"          # one file per gazette year

# Separate collection in the same Chroma store: legislation must not mix with the
# quiz distractor routing of hr-znanje.
COLLECTION = "hr-zakoni"

ZAKON_BASE = "https://www.zakon.hr"
ZAKON_SITEMAP = ZAKON_BASE + "/s/tmp/sitemap/sitemap.xml"
NN_BASE = "https://narodne-novine.nn.hr"
NN_SEARCH = (NN_BASE + "/search.aspx?sortiraj=4&kategorija=1"
             "&godina={year}&broj={broj}&rpp=200&qtype=1&pretraga=da")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) hfzakon/0.1 (local RAG, polite crawler)"
DELAY_S = 0.4                 # pause between uncached requests
TIMEOUT_S = 40
RETRIES = 3

# multilingual-e5-small embeds ~512 tokens (~1600 chars of Croatian); anything past
# that is invisible to retrieval, so chunks are split before they get there.
MAX_CHUNK_CHARS = 1600
TARGET_PART_CHARS = 1300
MIN_CHUNK_CHARS = 60

NN_YEARS = (2025, 2026)       # gazette years to mirror, NN 1/2025 .. latest 2026


def ensure_dirs() -> None:
    for d in (RAW_ZAKON_DIR, RAW_NN_SEARCH_DIR, RAW_NN_DOC_DIR, CHUNKS_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
