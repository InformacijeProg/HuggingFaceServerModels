# Fine-tuning EuroLLM on Croatian Q&A with hftrain

`hftrain` is a separate package next to `hfserve`. It fine-tunes the installed EuroLLM
model on your question-answer data with LoRA adapters, evaluates the result, and installs
the merged model back into `models/` so `hfserve run` can serve it. Nothing in `hfserve`
depends on it.

```powershell
.vemv\Scripts\activate
pip install -r requirements-train.txt        # peft + datasets
python -m hftrain --help                     # or:  .\hftrain.cmd --help
```

## What we are working with

* **The model.** utter-project/EuroLLM-1.7B-Instruct was pretrained on about 4T tokens
  covering all official EU languages — Croatian included — plus several others, and
  instruction-tuned for chat and translation. It already speaks grammatical Croatian out of
  the box (it occasionally drifts into Slovenian on short prompts and loops without a
  repetition penalty), so fine-tuning here means teaching it answers and style, not the
  language itself. Weights are Apache 2.0.
* **The tokenizer** is multilingual and handles Croatian well: about 0.315 tokens per
  character (~3.2 characters per token), with č ć đ š ž as real vocabulary entries. Its
  chat template has no `{% generation %}` markers, so chat examples train on every token,
  question included (the tokenize step warns about this once).
* **The GPU.** RTX 2070 SUPER, 8 GB, no native bfloat16. Training runs in float16 with
  loss scaling; the frozen base takes about 3.3 GB, LoRA adapters and optimizer a few
  hundred MB, the rest is activations. Full fine-tuning of 1.7B parameters does not fit;
  LoRA does (a 2-step smoke run peaked at 3.8 GB).

## Your Q&A data

Put `.jsonl` files (one JSON object per line, UTF-8) into `training/QA/`. The record
format is auto-detected per line:

```json
{"question": "Koja je razlika između č i ć?", "answer": "..."}
{"prompt": "...", "response": "...", "system": "optional system prompt"}
{"instruction": "...", "input": "optional extra context", "output": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The `messages` form may hold several turns and an optional leading system message. Write
Croatian with proper diacritics; both question and answer must be non-empty.

## The workflow

```powershell
# 0. baseline: how the untouched model answers (perplexity needs a corpus, see step 1)
python -m hftrain eval --no-perplexity

# 1. build a corpus from training/QA and tokenize it (default source and name)
python -m hftrain prepare --name qa-v1

# 2. train a LoRA adapter (Ctrl+C is safe, --resume continues)
python -m hftrain train --data qa-v1 --run qa-lora-1 --lr 1e-4

# 3. compare base and adapter: perplexity on held-out examples and sample answers
python -m hftrain eval --run qa-lora-1 --compare-base

# 4. merge the adapter into the weights and install it as a new hfserve model
python -m hftrain export --run qa-lora-1 --name local/EuroLLM-1.7B-hr
python -m hfserve run local/EuroLLM-1.7B-hr
```

Or use the window: `python -m hftrain gui` (or `HFTrain-GUI.cmd`) has a **Prepare QA
data** button (runs step 1 with an auto-numbered corpus name), Start/Stop training with a
progress bar and console, and a Chat tab to try the base model or any run's adapter.

`prepare` writes `training/data/<name>/` (`train.jsonl`, `val.jsonl`, `sources.json`, and
the tokenized `tok-1024/`). A fraction of the examples (`--val-fraction`, default 2%) is
held out for perplexity; with only a few hundred examples raise it to 0.1-0.2 so the
validation split is not empty.

### Sources

The default source is the QA folder. `--source` names others, repeatable:

* `file:<path>` — one file: `.jsonl`/`.json` records as above, or `.txt`/`.md` cut into
  passages of ~3000 characters (lines holding only `<|doc|>` split a file into documents).
* `dir:<folder>` — every such file below a folder.

Options after `@`: `max=N` (documents to take), `sample=0.1` (keep a random tenth),
`min=N` (minimum characters of a text document), `chunk=N` (passage size). Plain-text
passages train every token and can be mixed in so a small Q&A set does not overfit:

```powershell
python -m hftrain prepare --name qa-v2 --source dir:training/QA --source file:D:\tekstovi\clanci.txt@sample=0.5
```

### Training knobs (defaults are tuned for the 8 GB card)

| Flag | Default | Notes |
|---|---|---|
| `--seq-len` | 1024 | tokens per sequence; QA examples longer than this are cut |
| `--batch-size` / `--grad-accum` | 2 / 8 | 16 sequences per optimizer step; batch 4 overflows 8 GB VRAM |
| `--lr` | 2e-4 | use 1e-4 for instruction tuning an already-instructed model |
| `--epochs` / `--max-steps` | 1 / off | small Q&A sets tolerate 2-3 epochs; watch eval loss |
| `--lora-r` / `--lora-alpha` | 32 / 64 | adapter capacity; r=64 for a large corpus |
| `--target-modules` | all attention + MLP projections | |
| `--eval-steps` / `--save-steps` | 200 / 200 | perplexity on validation sequences; checkpoints for `--resume` |

### Throughput and how long a run takes

No long EuroLLM training run has been benchmarked on this machine yet. Before planning
one, benchmark the exact configuration for a few hundred steps and plan with about 80% of
the measured idle speed; `tokens_per_s` in `log.jsonl` shows the real speed after the
first ten steps.

On Windows a CUDA program that needs more than the free VRAM does not fail: the driver
moves the overflow into shared system memory and every step waits on PCIe, easily costing
half the throughput or more. The defaults are chosen to stay inside 8 GB; if a run is
slower than the benchmark promised, look at Task Manager > Performance > GPU > Shared GPU
memory.

Croatian costs about 2.1 tokens per word (0.315 tokens per character) with EuroLLM's
tokenizer, so a thousand Q&A pairs of a few sentences each are well under 1M tokens —
minutes per epoch, not hours.

## Plan

**Phase 1: instruction tuning on your own Q&A.** EuroLLM already speaks Croatian, so
training starts with answers, not the language. Fill `training/QA/`, then `prepare` and
`train` with `--lr 1e-4`. Mix in some plain Croatian text (`file:`/`dir:` sources) so the
model does not overfit to the Q&A phrasing. Note: EuroLLM's chat template has no
`{% generation %}` markers, so chat examples train on all tokens, question included; add
markers to a template copy first if assistant-only loss turns out to matter.

**Phase 2 (only if needed): continued pretraining on Croatian text** to lift style,
orthography, or domain knowledge — any plain `.txt`/`.md` files via `file:`/`dir:`
sources.

**Phase 3: evaluate and iterate.** `hftrain eval --compare-base` with your own prompts
(`--prompts-file`), then serve the model and talk to it (`hfserve gui`). Adjust data mix,
learning rate and steps from the eval reports (`training/runs/<run>/eval-*.json`).
