"""hftrain.data: source specs, record conversion, packing and corpus preparation.

No network and no real model: a whitespace ``FakeTokenizer`` stands in for the real one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hftrain import config as tconfig
from hftrain import data as hd


class FakeTokenizer:
    """Whitespace tokenizer; ids are assigned from 10 upwards as words are seen."""

    eos_token_id = 1
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]

    def __call__(self, text, add_special_tokens=True):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        return {"input_ids": self._ids(text)}

    def apply_chat_template(self, messages, tokenize=True, return_dict=True,
                            return_assistant_tokens_mask=True, **kwargs):
        ids: list[int] = []
        mask: list[int] = []
        for m in messages:
            toks = self._ids(m["content"]) + [self.eos_token_id]
            ids += toks
            mask += [1 if m["role"] == "assistant" else 0] * len(toks)
        return {"input_ids": ids, "assistant_masks": mask}


# --------------------------------------------------------------------------- parse_source
def test_parse_options_after_at(tmp_path):
    s = hd.parse_source(f"dir:{tmp_path}@max=10, min=5")
    assert s.max_docs == 10 and s.options == {"max": "10", "min": "5"}
    assert hd.parse_source(f"dir:{tmp_path}@max=0").max_docs is None  # 0 = unlimited
    assert hd.parse_source(f"dir:{tmp_path}").max_docs is None        # own data is never capped by default


def test_parse_file_and_dir_keep_windows_drive_letters():
    s = hd.parse_source(r"file:E:\data\knjiga.hr.txt@max=3")
    assert s.path == Path(r"E:\data\knjiga.hr.txt").resolve() and s.max_docs == 3
    assert s.name == "file:knjiga.hr.txt" and s.kind == "auto"
    d = hd.parse_source(r"dir:E:\data@kind=chat")
    assert d.kind == "chat" and d.path == Path(r"E:\data").resolve() and d.max_docs is None


@pytest.mark.parametrize("spec", ["", "   ", "nope", "file:", "dir:", "hf:org/repo", "wikipedia-hr",
                                  "file:x@max=abc"])
def test_parse_errors(spec):
    with pytest.raises(hd.DataError):
        hd.parse_source(spec)


# --------------------------------------------------------------------------- records
def test_detect_format():
    assert hd.detect_format({"text": "x"}) == "text"
    assert hd.detect_format({"content": "x"}, text_field="content") == "text"
    assert hd.detect_format({"messages": []}) == "messages"
    assert hd.detect_format({"conversations": []}) == "messages"
    assert hd.detect_format({"instruction": "a", "output": "b"}) == "alpaca"
    assert hd.detect_format({"inputs": "a", "targets": "b"}) == "aya"
    assert hd.detect_format({"prompt": "a", "response": "b"}) == "prompt"
    assert hd.detect_format({"question": "a", "answer": "b"}) == "prompt"
    assert hd.detect_format({"foo": 1}) is None


def test_to_document_text_and_alpaca():
    assert hd.to_document({"text": "hello"}) == {"text": "hello"}
    assert hd.to_document({"text": 5}) is None
    d = hd.to_document({"instruction": "Zbroji", "input": "2 i 3", "output": "5"})
    assert d["messages"] == [{"role": "user", "content": "Zbroji\n\n2 i 3"}, {"role": "assistant", "content": "5"}]
    d = hd.to_document({"instruction": "Reci bok", "input": "", "output": "Bok!", "system": "Ti si pomoćnik."})
    assert d["messages"][0] == {"role": "system", "content": "Ti si pomoćnik."}
    assert hd.to_document({"instruction": "x", "input": "", "output": ""}) is None


def test_to_document_messages_normalises_sharegpt_roles():
    d = hd.to_document({"conversations": [{"from": "human", "value": "Bok"}, {"from": "gpt", "value": "Bok!"}]})
    assert [m["role"] for m in d["messages"]] == ["user", "assistant"]
    assert hd.to_document({"messages": [{"role": "user", "content": "only user"}]}) is None
    assert hd.to_document({"messages": [{"role": "weird", "content": "x"}]}) is None


def test_to_document_aya_and_prompt():
    assert hd.to_document({"inputs": "Q", "targets": "A"})["messages"][-1]["content"] == "A"
    assert hd.to_document({"prompt": "Q", "completion": "A"})["messages"][0]["content"] == "Q"
    d = hd.to_document({"prompt": "Q", "chosen": [{"role": "assistant", "content": "A"}]})
    assert d["messages"][-1]["content"] == "A"


def test_add_generation_markers_on_eurollm_template():
    from hftrain import model as tmodel

    tpl = ("{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
           "<|im_start|>system\n"
           "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% endif %}<|im_end|>\n"
           "{% for message in messages %}{% if message['role'] != 'system' %}<|im_start|>{{ message['role'] }}\n"
           "{{ message['content'] }}<|im_end|>\n"
           "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    patched = tmodel.add_generation_markers(tpl)
    assert patched and "{% generation %}" in patched and "{% endgeneration %}" in patched
    assert tmodel.add_generation_markers(patched) is None   # already marked: no double patch
    assert tmodel.add_generation_markers("{{ bos }}{% for m in messages %}x{% endfor %}") is None
    assert tmodel.add_generation_markers(None) is None


def test_to_document_unknown_format_raises():
    with pytest.raises(hd.DataError):
        hd.to_document({"text": "x"}, fmt="bogus")


def test_finalize_filters():
    assert hd._finalize({"text": "short"}, 10, 100) is None
    long = hd._finalize({"text": "x" * 500}, 10, 100)
    assert len(long["text"]) == 100
    assert hd._finalize({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}]}, 10, 100) is None
    assert hd._finalize(None, 10, 100) is None


# --------------------------------------------------------------------------- packing
def test_encode_documents_packs_text_and_drops_short_tail():
    tok = FakeTokenizer()
    docs = [{"text": "a b c d e"}, {"text": "f g h i j"}, {"text": "k l m n o"}]  # 3 x (5 + eos) = 18 tokens
    counts = hd.new_counts()
    seqs = list(hd.encode_documents(docs, tok, 8, [], 1, counts))
    assert len(seqs) == 2 and all(len(s["input_ids"]) == 8 for s in seqs)
    assert seqs[0]["labels"] == seqs[0]["input_ids"]
    assert seqs[0]["input_ids"][5] == 1  # eos after the first document
    assert counts["sequences"] == 2 and counts["tokens"] == 16 and counts["text_docs"] == 3


def test_encode_documents_keeps_long_tail_and_prefix(monkeypatch):
    monkeypatch.setattr(hd, "MIN_TAIL_TOKENS", 2)
    tok = FakeTokenizer()
    counts = hd.new_counts()
    seqs = list(hd.encode_documents([{"text": "a b c"}], tok, 8, [7], 1, counts))
    assert seqs == [{"input_ids": [7, 10, 11, 12, 1], "labels": [7, 10, 11, 12, 1]}]


def test_encode_documents_masks_chat_prompt_tokens():
    tok = FakeTokenizer()
    counts = hd.new_counts()
    docs = [{"messages": [{"role": "user", "content": "kako si"}, {"role": "assistant", "content": "dobro"}]}]
    (ex,) = hd.encode_documents(docs, tok, 16, [], 1, counts)
    assert len(ex["input_ids"]) == 5  # 2 + eos + 1 + eos
    assert ex["labels"][:3] == [-100, -100, -100] and ex["labels"][3:] == ex["input_ids"][3:]
    assert counts["chat_examples"] == 1 and counts["target_tokens"] == 2


def test_encode_chat_truncates_and_drops_examples_without_targets():
    tok = FakeTokenizer()
    counts = hd.new_counts()
    long_user = " ".join(f"w{i}" for i in range(20))
    docs = [{"messages": [{"role": "user", "content": long_user}, {"role": "assistant", "content": "x"}]}]
    assert list(hd.encode_documents(docs, tok, 8, [], 1, counts)) == []  # cut before any assistant token
    assert counts["chat_dropped"] == 1 and counts["chat_truncated"] == 1
    docs = [{"messages": [{"role": "user", "content": "a b"}, {"role": "assistant", "content": long_user}]}]
    (ex,) = hd.encode_documents(docs, tok, 8, [], 1, counts)
    assert len(ex["input_ids"]) == 8 and counts["chat_truncated"] == 2


def test_encode_chat_without_generation_markers_trains_on_everything():
    class NoMaskTok(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [10, 11, 12], "assistant_masks": [0, 0, 0]}

    counts = hd.new_counts()
    docs = [{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}]
    (ex,) = hd.encode_documents(docs, NoMaskTok(), 8, [], 1, counts)
    assert ex["labels"] == [10, 11, 12]


# --------------------------------------------------------------------------- local files + prepare
def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_iter_documents_local_text_and_jsonl(tmp_path):
    book = _write(tmp_path / "knjiga.txt", "Prvi odlomak. " * 20 + "\n\n" + "Drugi odlomak. " * 20)
    docs = list(hd.iter_documents(hd.parse_source(f"file:{book}@min=10,chunk=100")))
    assert len(docs) == 2 and all(d["source"] == "file:knjiga.txt" for d in docs)
    recs = _write(tmp_path / "recs.jsonl", "\n".join([
        json.dumps({"instruction": "Reci bok", "input": "", "output": "Bok!"}),
        json.dumps({"text": "kratko"}),
        "not json",
        json.dumps({"text": "x" * 300}),
    ]))
    docs = list(hd.iter_documents(hd.parse_source(f"file:{recs}")))
    assert [("messages" in d) for d in docs] == [True, False]  # short text dropped, bad line skipped


def test_iter_documents_dir_and_max(tmp_path):
    _write(tmp_path / "d" / "a.txt", "a " * 200)
    _write(tmp_path / "d" / "b.md", "b " * 200)
    _write(tmp_path / "d" / "ignored.csv", "x,y")
    _write(tmp_path / "d" / "README.txt", "folder instructions " * 50)
    _write(tmp_path / "d" / "SOURCES.txt", "where the data came from " * 50)
    docs = list(hd.iter_documents(hd.parse_source(f"dir:{tmp_path / 'd'}@max=1")))
    assert len(docs) == 1
    docs = list(hd.iter_documents(hd.parse_source(f"dir:{tmp_path / 'd'}")))
    assert not any("README" in d["source"] or "SOURCES" in d["source"] for d in docs)  # docs never become data
    with pytest.raises(hd.DataError):
        list(hd.iter_documents(hd.parse_source(f"dir:{tmp_path / 'empty'}")))
    with pytest.raises(hd.DataError):
        list(hd.iter_documents(hd.parse_source(f"file:{tmp_path / 'd' / 'ignored.csv'}")))


def test_prepare_and_tokenize_corpus_end_to_end(train_dir, tmp_path, monkeypatch):
    text = _write(tmp_path / "t.txt", "\n\n".join("odlomak broj %d " % i * 30 for i in range(40)))
    chat = _write(tmp_path / "c.jsonl", "\n".join(
        json.dumps({"instruction": f"pitanje {i}", "input": "", "output": f"odgovor {i} je ovdje"}) for i in range(10)))
    specs = [hd.parse_source(f"file:{text}@chunk=400"), hd.parse_source(f"file:{chat}")]
    messages: list[str] = []
    stats = hd.prepare_corpus(specs, "c1", val_fraction=0.25, seed=1, progress=messages.append)
    out = tconfig.data_dir("c1")
    assert (out / "train.jsonl").exists() and (out / "val.jsonl").exists() and (out / "sources.json").exists()
    assert stats["train_docs"] + stats["val_docs"] == sum(e["docs"] for e in stats["sources"].values())
    assert stats["sources"]["file:c.jsonl"]["chat"] == 10 and stats["val_docs"] > 0
    assert messages and messages[-1].endswith("documents")
    with pytest.raises(hd.DataError):
        hd.prepare_corpus(specs, "c1")  # exists, no overwrite
    hd.prepare_corpus(specs, "c1", val_fraction=0.25, seed=1, overwrite=True)

    monkeypatch.setattr("hftrain.model.document_prefix_ids", lambda tok: [])
    meta = hd.tokenize_corpus("c1", FakeTokenizer(), 32, tokenizer_id="fake")
    assert meta["train"]["sequences"] > 0 and meta["train"]["chat_examples"] > 0
    assert not (hd.tokenized_dir("c1", 32) / "cache").exists()
    train, val, meta2 = hd.load_tokenized("c1", 32)
    assert meta2["seq_len"] == 32 and len(train) == meta["train"]["sequences"]
    row = train[0]
    assert len(row["input_ids"]) <= 32 and len(row["labels"]) == len(row["input_ids"])
    assert meta == hd.tokenize_corpus("c1", FakeTokenizer(), 32)  # cached meta is returned

    listing = hd.list_corpora()
    assert [c["name"] for c in listing] == ["c1"] and listing[0]["tokenized"][0]["seq_len"] == 32
    with pytest.raises(hd.DataError):
        hd.load_tokenized("c1", 64)
    with pytest.raises(hd.DataError):
        hd.prepare_corpus(specs, "bad/name")
