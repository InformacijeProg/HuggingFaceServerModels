"""Evaluation: perplexity on the held-out split and sample generations, for the base model
and for a run's LoRA adapter (optionally side by side)."""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from . import config, data as data_mod, model as model_mod
from .train import TrainError, adapter_dir, read_run

log = logging.getLogger("hftrain.eval")


def load_for_eval(run: str | None, model_ref: str | None, device: str | None, dtype: str | None, progress=print):
    """Returns (model, tokenizer, local, run_info). With ``run`` the adapter is attached."""
    run_info = read_run(run) if run else None
    ref = model_ref or (run_info["model"] if run_info else None)
    local = model_mod.find_local(ref)
    dev, dt = model_mod.resolve_device_dtype(device, dtype)
    tok = model_mod.load_tokenizer(local)
    progress(f"loading {local.repo_id} on {dev} ({model_mod.engine.dtype_name(dt)}) ...")
    model = model_mod.load_model(local, dev, dt, training=False)
    if run:
        adir = adapter_dir(run)
        progress(f"attaching adapter {adir}")
        model = model_mod.load_adapter(model, adir)
    model.eval()
    return model, tok, local, run_info


@torch.no_grad()
def perplexity(model, tok, dataset, *, batch_size: int = 4, max_sequences: int | None = 200) -> dict:
    device = next(model.parameters()).device
    if max_sequences is not None and len(dataset) > max_sequences:
        dataset = dataset.select(range(max_sequences))
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    total_nll, total_targets, n_seq = 0.0, 0, 0
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        targets = int((batch["labels"][:, 1:] != -100).sum().item())
        total_nll += float(out.loss.item()) * targets
        total_targets += targets
        n_seq += int(batch["input_ids"].shape[0])
    mean = total_nll / max(1, total_targets)
    return {"loss": round(mean, 4), "perplexity": round(math.exp(min(mean, 50.0)), 2),
            "target_tokens": total_targets, "sequences": n_seq, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def generate(model, tok, prompts: list[dict], *, max_new_tokens: int = 160, reasoning_effort: str = "low",
             think: bool = False, temperature: float = 0.6, top_p: float = 0.95, seed: int = 1) -> list[dict]:
    """One generation per prompt. Sampling is re-seeded per prompt, so base and adapter see
    the same random draws and their outputs are comparable; temperature 0 is greedy."""
    device = next(model.parameters()).device
    end_ids = model_mod.end_of_turn_ids(tok)
    prefix = model_mod.document_prefix_ids(tok)
    results = []
    for p in prompts:
        kind = p.get("kind", "chat")
        text = p["text"]
        if kind == "completion":
            ids = prefix + tok(text, add_special_tokens=False)["input_ids"]
        else:
            rendered = model_mod.render_chat_prompt(tok, text, reasoning_effort=reasoning_effort, think=think,
                                                   system=p.get("system"))
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], device=device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True)
        if end_ids:
            kwargs["eos_token_id"] = end_ids
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs["do_sample"] = False
        t0 = time.time()
        torch.manual_seed(seed)
        out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
        new = out[0, input_ids.shape[1]:]
        finished = bool(end_ids) and int(new[-1]) in end_ids
        results.append({"kind": kind, "prompt": text, "output": tok.decode(new, skip_special_tokens=True).strip(),
                        "tokens": int(new.shape[0]), "seconds": round(time.time() - t0, 1),
                        "finish": "stop" if finished else "length"})
    return results


def load_prompts(path: Path | None, extra: list[str]) -> list[dict]:
    prompts: list[dict] = []
    if path is not None:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if line.startswith("{"):
                    prompts.append(json.loads(line))
                elif line.startswith("completion:"):
                    prompts.append({"kind": "completion", "text": line[len("completion:"):].strip()})
                else:
                    prompts.append({"kind": "chat", "text": line.strip()})
    for text in extra:
        prompts.append({"kind": "chat", "text": text})
    return prompts or list(config.EVAL_PROMPTS)


def evaluate(*, run: str | None, model_ref: str | None = None, data: str | None = None,
             seq_len: int | None = None, compare_base: bool = False, prompts: list[dict] | None = None,
             max_new_tokens: int = 160, no_generate: bool = False, no_perplexity: bool = False,
             max_sequences: int = 200, device: str | None = None, dtype: str | None = None,
             reasoning_effort: str = "low", think: bool = False, temperature: float = 0.6, seed: int = 1,
             progress=print) -> dict:
    model, tok, local, run_info = load_for_eval(run, model_ref, device, dtype, progress)
    report: dict = {"run": run, "model": local.repo_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "compare_base": compare_base and run is not None}
    data_name = data or (run_info["data"] if run_info else None)
    seq = seq_len or (run_info["config"]["seq_len"] if run_info else config.DEFAULT_SEQ_LEN)
    val_ds = None
    if not no_perplexity and data_name:
        try:
            _train, val_ds, _meta = data_mod.load_tokenized(data_name, seq)
        except data_mod.DataError as e:
            progress(f"perplexity skipped: {e}")
        if val_ds is None and _train is not None:
            progress("perplexity skipped: the corpus has no validation split")
    report["data"] = data_name if val_ds is not None else None

    def run_all(label: str) -> dict:
        part: dict = {}
        if val_ds is not None:
            progress(f"[{label}] perplexity on {min(len(val_ds), max_sequences)} validation sequences ...")
            part["perplexity"] = perplexity(model, tok, val_ds, max_sequences=max_sequences)
            progress(f"[{label}] loss {part['perplexity']['loss']}  perplexity {part['perplexity']['perplexity']}")
        if not no_generate:
            progress(f"[{label}] generating {len(prompts or config.EVAL_PROMPTS)} samples ...")
            part["samples"] = generate(model, tok, prompts or config.EVAL_PROMPTS, max_new_tokens=max_new_tokens,
                                       reasoning_effort=reasoning_effort, think=think, temperature=temperature,
                                       seed=seed)
        return part

    if run and compare_base:
        with model.disable_adapter():
            report["base"] = run_all("base")
        report["adapter"] = run_all("adapter")
    elif run:
        report["adapter"] = run_all("adapter")
    else:
        report["base"] = run_all("base")

    out_dir = config.run_dir(run) if run else config.TRAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / (f"eval-{stamp}.json" if run else f"eval-base-{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["path"] = str(out)
    return report


def format_report(report: dict) -> str:
    lines: list[str] = []
    parts = [(k, report[k]) for k in ("base", "adapter") if k in report]
    for label, part in parts:
        if "perplexity" in part:
            p = part["perplexity"]
            lines.append(f"{label:<8} loss {p['loss']:.4f}   perplexity {p['perplexity']:.2f}   "
                         f"({p['target_tokens']} target tokens, {p['sequences']} sequences)")
    if len(parts) == 2 and all("perplexity" in p for _, p in parts):
        b, a = parts[0][1]["perplexity"]["perplexity"], parts[1][1]["perplexity"]["perplexity"]
        if b > 0:
            lines.append(f"perplexity change: {b:.2f} -> {a:.2f} ({(a - b) / b * 100:+.1f}%)")
    samples = [(label, part.get("samples")) for label, part in parts if part.get("samples")]
    if samples:
        n = len(samples[0][1])
        for i in range(n):
            prompt = samples[0][1][i]
            lines.append("")
            lines.append(f"[{prompt['kind']}] {prompt['prompt']}")
            for label, s in samples:
                if i < len(s):
                    lines.append(f"  {label:<8}| {s[i]['output'].replace(chr(10), chr(10) + '          ')}")
    lines.append("")
    lines.append(f"report: {report.get('path')}")
    return "\n".join(lines)
