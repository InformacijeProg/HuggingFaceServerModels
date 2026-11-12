"""Paths and defaults used by hftrain.

Everything lives under ``training/`` next to ``models/`` (override with ``HFTRAIN_HOME``)::

    training/
      QA/                              your question-answer .jsonl files (see its README.txt)
      data/<name>/train.jsonl          raw documents chosen by `hftrain prepare`
      data/<name>/val.jsonl
      data/<name>/sources.json         what was sampled from where, with counts
      data/<name>/tok-<seq_len>/       tokenized + packed Arrow datasets (train, val)
      runs/<run>/run.json              config and results of one training run
      runs/<run>/checkpoints/          Trainer checkpoints (resumable)
      runs/<run>/adapter/              final LoRA adapter
      runs/<run>/log.jsonl             one JSON object per logged step
      runs/<run>/eval-*.json           evaluation reports

Environment variables:
  HFTRAIN_HOME     root of the training tree (default: <hfserve home>/training)
  HFSERVE_DEVICE / HFSERVE_DTYPE   same meaning as for the server (see hfserve/config.py)
"""
from __future__ import annotations

import os
from pathlib import Path

from hfserve import config as serve_config

TRAIN_DIR = Path(os.environ.get("HFTRAIN_HOME", serve_config.HOME_DIR / "training")).expanduser().resolve()
DATA_DIR = TRAIN_DIR / "data"
RUNS_DIR = TRAIN_DIR / "runs"
QA_DIR = TRAIN_DIR / "QA"               # question-answer .jsonl files; the default `prepare` source
DOC_SEPARATOR = "<|doc|>"               # a line holding only this starts a new document in a .txt source
TEXT_CHUNK_CHARS = 3000          # a plain-text passage = consecutive paragraphs up to this size

DEFAULT_MODEL = "utter-project/EuroLLM-1.7B-Instruct"
DEFAULT_SEQ_LEN = 1024
DEFAULT_VAL_FRACTION = 0.02
MIN_TEXT_CHARS = 200             # shorter plain-text documents are dropped
MAX_TEXT_CHARS = 40_000          # longer ones are cut (one document must not dominate)

# LoRA defaults: every linear layer of attention and MLP. Names are those used by
# Llama-style models (EuroLLM, Mistral, Qwen).
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 1           # batch 2 with PACKED seq-1024 sequences (wiki prose) reserves ~8.1 GB on the
DEFAULT_GRAD_ACCUM = 16          # 1.7B model and Windows spills VRAM into RAM (measured 2026-10-16, qa-lora-2
                                 # step ~80 s instead of ~30 s). Batch 1 x accum 16 keeps the same effective batch.
EXPECTED_TOKENS_PER_S = 1150     # rough planning number from earlier 0.9B runs; EuroLLM-1.7B is slower —
                                 # the progress bar's own estimate takes over after a few steps
DEFAULT_WARMUP_STEPS = 50

# Text and chat prompts used by `hftrain eval` when no prompt file is given. Completion
# prompts test the raw language model (plain continuation); chat prompts go through the
# chat template.
EVAL_PROMPTS: list[dict] = [
    {"kind": "completion", "text": "Zagreb je glavni grad Hrvatske. Nalazi se"},
    {"kind": "completion", "text": "Najpoznatija hrvatska jela su"},
    {"kind": "completion", "text": "Jadransko more je"},
    {"kind": "chat", "text": "Napiši tri rečenice o Dubrovniku."},
    {"kind": "chat", "text": "Koja je razlika između rijeke i jezera? Odgovori na hrvatskom."},
    {"kind": "chat", "text": "Prevedi na hrvatski: The weather is nice today, so we are going to the beach."},
    {"kind": "chat", "text": "Ispravi pravopisne pogreške: Danas sam isao u skolu i ucio hrvatski."},
]


def data_dir(name: str) -> Path:
    return DATA_DIR / name


def run_dir(name: str) -> Path:
    return RUNS_DIR / name
