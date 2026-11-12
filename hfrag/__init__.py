"""hfrag - local retrieval (RAG) over Croatian knowledge for hfserve models.

Chroma vector index + multilingual embeddings. Ingest question-answer JSONL or
text chunks, retrieve context for a chat question, and build a grounded prompt
with the "ne znam" rule. See docs/TRAINING.md (RAG section) and `python -m hfrag --help`.
"""

__version__ = "0.1.0"
