"""Data sources -> documents -> tokenized, packed datasets ready for the Trainer.

Two document shapes flow through this module::

    {"text": "...", "source": "file:note.txt"}                          plain text
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "source": "dir:QA"}

`prepare_corpus` samples documents from the requested sources into ``train.jsonl`` and
``val.jsonl`` (so a training run never depends on the network), and `tokenize_corpus`
turns them into fixed-length sequences:

* text documents are concatenated (``<doc><eos><doc><eos>...``) and cut into
  ``seq_len`` pieces ("packing"); every token is a training target;
* chat documents are rendered with the model's chat template and only the assistant
  tokens are targets (labels ``-100`` elsewhere); they are padded per batch, not packed.
"""
from __future__ import annotations

import json
import logging
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from . import config

log = logging.getLogger("hftrain.data")

ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "bot": "assistant", "system": "system"}
TEXT_BATCH = 64
MIN_TAIL_TOKENS = 64


class DataError(Exception):
    """Message is safe to print to the user."""


# --------------------------------------------------------------------------- source specs
@dataclass
class SourceSpec:
    """A parsed ``--source`` argument. Forms::

        file:C:\\path\\qa.jsonl       one file: jsonl/json -> records, txt/md -> passages
        file:C:\\path\\qa.jsonl@max=500,sample=0.1
        dir:C:\\path\\folder          every such file below a folder

    Options: ``max`` (documents to take), ``sample`` (keep this fraction of the documents, e.g.
    0.1, spread evenly over all files), ``min`` (min chars of a text document), ``chunk``
    (max chars of a plain-text passage), ``fmt`` and ``field`` (record layout, see
    ``to_document``).

    A .txt/.md file with lines holding only ``config.DOC_SEPARATOR`` gives one document per
    separated part instead of passages.
    """

    spec: str
    name: str
    kind: str
    path: Path | None = None
    options: dict[str, str] = field(default_factory=dict)
    max_docs: int | None = None

    def describe(self) -> str:
        cap = f", max {self.max_docs} docs" if self.max_docs else ""
        return f"{self.name}: {self.path} ({self.kind}{cap})"


def _parse_options(text: str) -> dict[str, str]:
    opts: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        opts[key.strip().lower()] = value.strip()
    return opts


def parse_source(spec: str) -> SourceSpec:
    """Local files and folders only; nothing is truncated unless the spec says ``@max=N``."""
    body, _, opts = spec.strip().partition("@")
    options = _parse_options(opts)
    body = body.strip()
    if "max" in options:
        try:
            max_docs: int | None = int(options["max"]) or None
        except ValueError:
            raise DataError(f"Bad max in source {spec!r}: {options['max']!r} is not a number") from None
    else:
        max_docs = None
    if not body:
        raise DataError("Empty source.")
    if body.startswith(("file:", "dir:")):
        prefix, _, rest = body.partition(":")
        if not rest:
            raise DataError(f"Source {spec!r} needs a path after '{prefix}:'.")
        path = Path(rest).expanduser().resolve()
        return SourceSpec(spec, f"{prefix}:{path.name}", options.get("kind", "auto"), path=path,
                          options=options, max_docs=max_docs)
    raise DataError(f"Unknown source {spec!r}. Use file:<path> (jsonl/json records, txt/md passages) "
                    f"or dir:<folder>.")


# --------------------------------------------------------------------------- records -> documents
def detect_format(rec: dict, text_field: str = "text") -> str | None:
    keys = set(rec)
    if "messages" in keys or "conversations" in keys:
        return "messages"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "inputs" in keys and "targets" in keys:
        return "aya"
    if "prompt" in keys and ("response" in keys or "completion" in keys or "chosen" in keys):
        return "prompt"
    if "question" in keys and "answer" in keys:
        return "prompt"
    if text_field in keys:
        return "text"
    return None


def _chat(user: str, assistant: str, system: str | None = None) -> dict | None:
    user, assistant = (user or "").strip(), (assistant or "").strip()
    if not user or not assistant:
        return None
    msgs = [{"role": "system", "content": system.strip()}] if system and system.strip() else []
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def _normalise_messages(raw) -> dict | None:
    if not isinstance(raw, list):
        return None
    out: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            return None
        role = ROLE_ALIASES.get(str(m.get("role") or m.get("from") or "").lower())
        content = m.get("content", m.get("value"))
        if role is None or not isinstance(content, str):
            return None
        out.append({"role": role, "content": content.strip()})
    if not any(m["role"] == "user" for m in out) or not any(m["role"] == "assistant" for m in out):
        return None
    if not out[-1]["content"]:
        return None
    return {"messages": out}


def to_document(rec: dict, fmt: str = "auto", text_field: str = "text", index: int = 0) -> dict | None:
    """Turn one dataset record into a text or chat document, or None to skip it."""
    if not isinstance(rec, dict):
        return None
    if fmt == "auto":
        fmt = detect_format(rec, text_field)
        if fmt is None:
            return None
    if fmt == "text":
        t = rec.get(text_field)
        return {"text": t} if isinstance(t, str) else None
    if fmt == "messages":
        return _normalise_messages(rec.get("messages", rec.get("conversations")))
    if fmt == "alpaca":
        instr, inp = str(rec.get("instruction") or ""), str(rec.get("input") or "")
        user = instr + ("\n\n" + inp if inp.strip() else "")
        return _chat(user, str(rec.get("output") or ""), rec.get("system"))
    if fmt == "aya":
        return _chat(str(rec.get("inputs") or ""), str(rec.get("targets") or ""))
    if fmt == "prompt":
        user = rec.get("prompt", rec.get("question"))
        answer = rec.get("response", rec.get("completion", rec.get("chosen", rec.get("answer"))))
        if isinstance(answer, list):  # chosen as a message list
            answer = " ".join(m.get("content", "") for m in answer if isinstance(m, dict) and m.get("role") == "assistant")
        return _chat(str(user or ""), str(answer or ""), rec.get("system"))
    raise DataError(f"Unknown record format {fmt!r}; use text, messages, alpaca, aya or prompt.")


def _finalize(doc: dict | None, min_chars: int, max_chars: int) -> dict | None:
    if not doc:
        return None
    if "text" in doc:
        text = str(doc["text"]).strip()
        if len(text) < min_chars:
            return None
        if len(text) > max_chars:
            text = text[:max_chars]
        doc["text"] = text
        return doc
    msgs = doc.get("messages")
    if not msgs or not msgs[-1].get("content"):
        return None
    return doc


def doc_chars(doc: dict) -> int:
    if "text" in doc:
        return len(doc["text"])
    return sum(len(m.get("content", "")) for m in doc.get("messages", []))


# --------------------------------------------------------------------------- iteration
def _iter_records_file(path: Path) -> Iterator[dict]:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        log.warning("%s: skipping a line that is not JSON", path.name)
    else:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "rows", "examples", "train"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            for rec in data:
                yield rec


_DOC_SEPARATOR_LINE = re.compile(r"^[ \t]*" + re.escape(config.DOC_SEPARATOR) + r"[ \t]*\r?$", re.MULTILINE)
TEXT_SUFFIXES = (".txt", ".md")


def split_documents(text: str) -> list[str] | None:
    """The parts of a text marked with ``config.DOC_SEPARATOR`` lines, or None when the text
    has no such line (and is cut into passages instead)."""
    if not _DOC_SEPARATOR_LINE.search(text):
        return None
    return [part for part in _DOC_SEPARATOR_LINE.split(text) if part.strip()]


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def split_paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks; wrapped lines inside a block are joined. Files that
    hold one paragraph per line (no blank lines at all) are split on single newlines."""
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\xa0', " ")
    lines = text.split("\n")
    blocks = re.split(r"\n\s*\n", text)
    paras = [_clean_block(b) for b in blocks]
    paras = [p for p in paras if p]
    if len(lines) > 20 and len(paras) <= max(3, len(lines) // 50):
        paras = [_clean_block(line) for line in lines]
        paras = [p for p in paras if p]
    return paras


def _clean_block(block: str) -> str:
    parts = [ln.strip() for ln in block.split("\n")]
    joined = " ".join(p for p in parts if p)
    return re.sub(r"[ \t]+", " ", joined).strip()


def chunk_paragraphs(paras: list[str], max_chars: int) -> list[str]:
    """Consecutive paragraphs joined with blank lines, each chunk at most ``max_chars``
    (a single longer paragraph becomes its own chunk)."""
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if buf and size + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _iter_local(path: Path, options: dict[str, str]) -> Iterator[dict]:
    if not path.exists():
        raise DataError(f"{path} does not exist.")
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    fmt = options.get("fmt", "auto")
    field_name = options.get("field", "text")
    chunk = int(options.get("chunk", config.TEXT_CHUNK_CHARS))
    seen = False
    for f in files:
        if f.stem.lower() in ("readme", "sources"):  # folder documentation, not data (training/QA)
            continue
        suffix = f.suffix.lower()
        if suffix in (".jsonl", ".ndjson", ".json"):
            seen = True
            for i, rec in enumerate(_iter_records_file(f)):
                d = to_document(rec, fmt, field_name, i)
                if d:
                    d.setdefault("source", f"file:{f.name}")
                    yield d
        elif suffix in TEXT_SUFFIXES:
            seen = True
            raw = read_text_file(f)
            documents = split_documents(raw)
            if documents is not None:
                for part in documents:
                    paras = split_paragraphs(part)
                    if paras:
                        yield {"text": "\n\n".join(paras), "source": f"file:{f.name}"}
                continue
            for text in chunk_paragraphs(split_paragraphs(raw), chunk):
                yield {"text": text, "source": f"file:{f.name}"}
        elif path.is_file():
            raise DataError(f"{f.name}: unsupported file type; use jsonl, json, txt or md.")
    if not seen:
        raise DataError(f"No usable files under {path} (jsonl, json, txt, md).")


def iter_documents(spec: SourceSpec) -> Iterator[dict]:
    """Documents of one source, filtered and capped at ``spec.max_docs``."""
    opts = spec.options
    min_chars = int(opts.get("min", config.MIN_TEXT_CHARS))
    max_chars = int(opts.get("maxchars", config.MAX_TEXT_CHARS))
    try:
        sample = float(opts.get("sample", 1))
    except ValueError:
        raise DataError(f"Bad sample in source {spec.spec!r}: {opts['sample']!r} is not a number") from None
    if not 0 < sample <= 1:
        raise DataError(f"Bad sample in source {spec.spec!r}: use a fraction between 0 and 1, e.g. sample=0.1")
    rng = random.Random(spec.spec)
    assert spec.path is not None
    it: Iterable[dict] = _iter_local(spec.path, opts)
    n = 0
    for doc in it:
        doc = _finalize(doc, min_chars, max_chars)
        if doc is None or (sample < 1 and rng.random() >= sample):
            continue
        doc.setdefault("source", spec.name)
        yield doc
        n += 1
        if spec.max_docs and n >= spec.max_docs:
            break


# --------------------------------------------------------------------------- prepare
def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def prepare_corpus(specs: list[SourceSpec], name: str, *, val_fraction: float = config.DEFAULT_VAL_FRACTION,
                   seed: int = 42, overwrite: bool = False,
                   progress: Callable[[str], None] | None = None) -> dict:
    """Sample every source into DATA_DIR/<name>/{train,val}.jsonl and write sources.json."""
    if not name or "/" in name or "\\" in name:
        raise DataError("Corpus name must be a simple folder name, e.g. hr-v1.")
    out = config.data_dir(name)
    if (out / "train.jsonl").exists() and not overwrite:
        raise DataError(f"Corpus {name!r} already exists at {out}; use --overwrite or another --name.")
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats: dict = {
        "name": name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
        "val_fraction": val_fraction, "specs": [s.spec for s in specs], "sources": {},
        "train_docs": 0, "val_docs": 0, "train_chars": 0, "val_chars": 0,
    }
    with open(out / "train.jsonl", "w", encoding="utf-8", newline="\n") as ftr, \
            open(out / "val.jsonl", "w", encoding="utf-8", newline="\n") as fva:
        for spec in specs:
            if progress:
                progress(f"{spec.describe()} ...")
            n = 0
            for doc in iter_documents(spec):
                key = doc.get("source", spec.name)
                entry = stats["sources"].setdefault(key, {"docs": 0, "val_docs": 0, "chars": 0, "text": 0, "chat": 0})
                chars = doc_chars(doc)
                is_val = rng.random() < val_fraction
                (fva if is_val else ftr).write(json.dumps(doc, ensure_ascii=False) + "\n")
                entry["docs"] += 1
                entry["chars"] += chars
                entry["text" if "text" in doc else "chat"] += 1
                if is_val:
                    entry["val_docs"] += 1
                    stats["val_docs"] += 1
                    stats["val_chars"] += chars
                else:
                    stats["train_docs"] += 1
                    stats["train_chars"] += chars
                n += 1
                if progress and n % 1000 == 0:
                    progress(f"  {spec.name}: {n} documents")
            if progress:
                progress(f"  {spec.name}: {n} documents")
    with open(out / "sources.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


# --------------------------------------------------------------------------- tokenize
def encode_chat(messages: list[dict], tokenizer, seq_len: int, counts: dict, warned: list) -> dict | None:
    try:
        enc = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            return_assistant_tokens_mask=True)
        ids = list(enc["input_ids"])
        mask = list(enc.get("assistant_masks") or [])
    except Exception as e:  # noqa: BLE001 - template rejects the conversation
        if not warned:
            log.warning("chat template failed on an example (%s); such examples are dropped", e)
            warned.append(True)
        counts["chat_dropped"] += 1
        return None
    if len(mask) != len(ids) or not any(mask):
        if len(warned) < 2:
            log.warning("chat template has no {%% generation %%} markers: every token of a chat example "
                        "(prompt included) becomes a training target")
            warned.extend([True, True])
        mask = [1] * len(ids)
    if len(ids) > seq_len:
        ids, mask = ids[:seq_len], mask[:seq_len]
        counts["chat_truncated"] += 1
    labels = [t if m else -100 for t, m in zip(ids, mask)]
    if all(v == -100 for v in labels):
        counts["chat_dropped"] += 1
        return None
    counts["chat_examples"] += 1
    counts["sequences"] += 1
    counts["tokens"] += len(ids)
    counts["target_tokens"] += sum(1 for v in labels if v != -100)
    return {"input_ids": ids, "labels": labels}


def encode_documents(docs: Iterable[dict], tokenizer, seq_len: int, prefix: list[int], eos: int | None,
                     counts: dict) -> Iterator[dict]:
    """Packed text sequences and masked chat examples, in input order."""
    buf: list[int] = []
    texts: list[str] = []
    warned: list = []

    def flush() -> None:
        if not texts:
            return
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            buf.extend(prefix)
            buf.extend(ids)
            if eos is not None:
                buf.append(eos)
        texts.clear()

    def drain() -> Iterator[dict]:
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            del buf[:seq_len]
            counts["sequences"] += 1
            counts["tokens"] += seq_len
            counts["target_tokens"] += seq_len
            yield {"input_ids": chunk, "labels": list(chunk)}

    for doc in docs:
        if "messages" in doc:
            ex = encode_chat(doc["messages"], tokenizer, seq_len, counts, warned)
            if ex:
                yield ex
            continue
        texts.append(doc["text"])
        counts["text_docs"] += 1
        if len(texts) >= TEXT_BATCH:
            flush()
            yield from drain()
    flush()
    yield from drain()
    if len(buf) >= MIN_TAIL_TOKENS:
        counts["sequences"] += 1
        counts["tokens"] += len(buf)
        counts["target_tokens"] += len(buf)
        yield {"input_ids": list(buf), "labels": list(buf)}


def new_counts() -> dict:
    return {"sequences": 0, "tokens": 0, "target_tokens": 0, "text_docs": 0,
            "chat_examples": 0, "chat_truncated": 0, "chat_dropped": 0}


def tokenized_dir(name: str, seq_len: int) -> Path:
    return config.data_dir(name) / f"tok-{seq_len}"


def tokenize_corpus(name: str, tokenizer, seq_len: int, *, tokenizer_id: str = "", overwrite: bool = False,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Write DATA_DIR/<name>/tok-<seq_len>/{train,val} (Arrow) and meta.json; returns meta."""
    try:
        from datasets import Dataset, Features, Sequence, Value
    except ImportError as e:
        raise DataError("The 'datasets' package is missing: pip install -r requirements-train.txt") from e

    ddir = config.data_dir(name)
    if not (ddir / "train.jsonl").exists():
        raise DataError(f"No corpus named {name!r} under {config.DATA_DIR}; run: hftrain prepare --name {name} ...")
    tdir = tokenized_dir(name, seq_len)
    if (tdir / "meta.json").exists() and not overwrite:
        with open(tdir / "meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    from .model import document_prefix_ids

    prefix = document_prefix_ids(tokenizer)
    eos = tokenizer.eos_token_id
    features = Features({"input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int32"))})
    meta: dict = {"name": name, "seq_len": seq_len, "tokenizer": tokenizer_id, "prefix_ids": prefix, "eos_id": eos,
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    cache = tdir / "cache"
    for split in ("train", "val"):
        src = ddir / f"{split}.jsonl"
        if not src.exists() or src.stat().st_size == 0:
            meta[split] = None
            continue
        if progress:
            progress(f"tokenizing {split} ...")
        counts = new_counts()

        def gen(_src=src, _counts=counts):
            yield from encode_documents(read_jsonl(_src), tokenizer, seq_len, prefix, eos, _counts)

        ds = Dataset.from_generator(gen, features=features, cache_dir=str(cache))
        if len(ds) == 0:
            meta[split] = None
            continue
        ds.save_to_disk(str(tdir / split))
        meta[split] = counts
        if progress:
            progress(f"  {split}: {counts['sequences']} sequences, {counts['tokens']} tokens "
                     f"({counts['target_tokens']} targets)")
    del ds  # release memory-mapped cache files before deleting them (Windows)
    shutil.rmtree(cache, ignore_errors=True)
    with open(tdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_tokenized(name: str, seq_len: int):
    """(train, val or None, meta) as ``datasets.Dataset`` objects."""
    from datasets import load_from_disk

    tdir = tokenized_dir(name, seq_len)
    if not (tdir / "meta.json").exists():
        raise DataError(f"Corpus {name!r} is not tokenized for seq_len {seq_len}; run: "
                        f"hftrain prepare --name {name} --seq-len {seq_len} ... (or hftrain tokenize)")
    with open(tdir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    train = load_from_disk(str(tdir / "train")) if meta.get("train") else None
    if train is None:
        raise DataError(f"Corpus {name!r} has no training sequences.")
    val = load_from_disk(str(tdir / "val")) if meta.get("val") else None
    return train, val, meta


def list_corpora() -> list[dict]:
    out: list[dict] = []
    if not config.DATA_DIR.exists():
        return out
    for d in sorted(config.DATA_DIR.iterdir()):
        if not d.is_dir() or not (d / "sources.json").exists():
            continue
        try:
            with open(d / "sources.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, ValueError):
            continue
        toks = []
        for t in sorted(d.glob("tok-*")):
            try:
                with open(t / "meta.json", "r", encoding="utf-8") as f:
                    toks.append(json.load(f))
            except (OSError, ValueError):
                continue
        out.append({"name": d.name, "path": str(d), "stats": stats, "tokenized": toks})
    return out
