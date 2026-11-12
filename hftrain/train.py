"""LoRA fine-tuning of an installed model with the transformers Trainer.

Mixed precision follows the GPU: float16 (with loss scaling) on Turing such as the RTX
2070 SUPER, bfloat16 on Ampere and newer. LoRA weights are float32 either way. The base
weights are frozen, so an 8 GB card fits a 1B model at seq_len 1024 with batch 2 (x8
accumulation) and gradient checkpointing. Batch 4 fills the card and Windows moves VRAM into
shared system memory, which makes training 2-3x slower.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_callback import ProgressCallback

from . import config, data as data_mod, model as model_mod

log = logging.getLogger("hftrain.train")


class TrainError(Exception):
    """Message is safe to print to the user."""


@dataclass
class TrainConfig:
    data: str
    run: str
    model: str = config.DEFAULT_MODEL
    seq_len: int = config.DEFAULT_SEQ_LEN
    batch_size: int = config.DEFAULT_BATCH_SIZE
    grad_accum: int = config.DEFAULT_GRAD_ACCUM
    lr: float = config.DEFAULT_LR
    epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = config.DEFAULT_WARMUP_STEPS
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    lora_r: int = config.DEFAULT_LORA_R
    lora_alpha: int = config.DEFAULT_LORA_ALPHA
    lora_dropout: float = config.DEFAULT_LORA_DROPOUT
    target_modules: tuple[str, ...] = config.DEFAULT_TARGET_MODULES
    train_embeddings: bool = False
    gradient_checkpointing: bool = True
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 2
    eval_sequences: int = 200
    seed: int = 42
    device: str | None = None
    dtype: str | None = None
    resume: bool = False


class JsonlLogger(TrainerCallback):
    """Appends every Trainer log line to <run>/log.jsonl with wall time and tokens/s."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.t0 = time.time()
        self._last = (time.time(), 0)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        now = time.time()
        rec = {"step": state.global_step, "epoch": round(float(state.epoch or 0), 4),
               "elapsed": round(now - self.t0, 1), **logs}
        seen = getattr(state, "num_input_tokens_seen", 0) or 0
        t_prev, seen_prev = self._last
        if seen > seen_prev and now > t_prev:
            rec["tokens_per_s"] = round((seen - seen_prev) / (now - t_prev), 1)
            rec["tokens_seen"] = seen
        self._last = (now, seen)
        if torch.cuda.is_available():
            rec["vram_max_mb"] = int(torch.cuda.max_memory_allocated() // 2**20)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class ConsoleProgress(TrainerCallback):
    """One progress bar for the whole run: steps, elapsed/remaining time, loss, tokens/s, VRAM.

    Replaces the Trainer's own bar, which prints a raw dict every logging step. The bar starts
    with the first training step, so its time estimate leaves out model loading and the initial
    evaluation; later evaluations count, they are part of the run. Evaluations, checkpoints and
    a nearly full GPU are reported on their own lines above the bar.

    With ``progress_file`` the same numbers go to a small JSON file (at most every 2 s), which
    the GUI reads for its own progress bar.
    """

    VRAM_WARN = 0.95  # share of the card reserved before Windows starts spilling into system RAM
    FIELDS = ("loss", "tok/s", "VRAM", "eval")

    def __init__(self, progress_file: Path | None = None) -> None:
        self.bar = None
        self.eval_bar = None
        self.progress_file = progress_file
        self._t_start = 0.0
        self._step_start = 0
        self._last = (0.0, 0)
        self._post: dict[str, str] = {}
        self._stats: dict = {}
        self._best_eval: float | None = None
        self._warned_vram = False
        self._dumped = 0.0

    def _write(self, msg: str) -> None:
        (self.bar.write if self.bar is not None else tqdm.write)(msg)

    def _dump(self, state, phase: str = "training", force: bool = False) -> None:
        if self.progress_file is None:
            return
        now = time.time()
        if not force and now - self._dumped < 2.0:
            return
        self._dumped = now
        done = state.global_step - self._step_start
        elapsed = now - self._t_start if self._t_start else 0.0
        remaining = (elapsed / done * (state.max_steps - state.global_step)) if done > 0 and self._t_start else None
        rec = {"phase": phase, "step": state.global_step, "max_steps": state.max_steps,
               "elapsed_s": round(elapsed, 1), "remaining_s": None if remaining is None else round(remaining, 1),
               "tokens_seen": getattr(state, "num_input_tokens_seen", 0) or 0, "updated": now, **self._stats}
        try:
            tmp = self.progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec), encoding="utf-8")
            tmp.replace(self.progress_file)
        except OSError:  # the GUI may be reading it right now; next time
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        self._dump(state, phase="starting", force=True)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.bar is None and state.is_world_process_zero:
            self.bar = tqdm(total=state.max_steps, initial=state.global_step, desc="training", unit="step",
                            dynamic_ncols=True, smoothing=0,
                            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                       "[{elapsed} elapsed, {remaining} left]{postfix}")
            self._t_start = time.time()
            self._step_start = state.global_step
            self._last = (self._t_start, getattr(state, "num_input_tokens_seen", 0) or 0)

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.update(state.global_step - self.bar.n)
            self._dump(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.bar is None or not logs or "loss" not in logs:
            return
        now = time.time()
        seen = getattr(state, "num_input_tokens_seen", 0) or 0
        t_prev, seen_prev = self._last
        if seen > seen_prev and now > t_prev:
            self._stats["tokens_per_s"] = round((seen - seen_prev) / (now - t_prev), 1)
            self._post["tok/s"] = f"{self._stats['tokens_per_s']:,.0f}"
        self._last = (now, seen)
        self._stats["loss"] = round(float(logs["loss"]), 4)
        self._post["loss"] = f"{float(logs['loss']):.3f}"
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            self._stats.update(vram_reserved_mb=int(reserved // 2**20), vram_total_mb=int(total // 2**20))
            self._post["VRAM"] = f"{reserved / 2**30:.1f}/{total / 2**30:.0f} GB"
            if reserved > self.VRAM_WARN * total and not self._warned_vram:
                self._warned_vram = True
                self._write("warning: the GPU is almost full; Windows may move memory into system RAM and slow "
                            "training down several times. Close other GPU programs (hfserve, Ollama, browser).")
        self.bar.set_postfix_str("  ".join(f"{k} {self._post[k]}" for k in self.FIELDS if k in self._post),
                                 refresh=True)
        self._dump(state, force=True)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not state.is_world_process_zero or eval_dataloader is None:
            return
        if self.eval_bar is None:
            self.eval_bar = tqdm(total=len(eval_dataloader), desc="evaluating", unit="batch", leave=False,
                                 dynamic_ncols=True, position=1 if self.bar is not None else 0)
        self.eval_bar.update(1)
        self._dump(state, phase="evaluating")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.eval_bar is not None:
            self.eval_bar.close()
            self.eval_bar = None
        loss = (metrics or {}).get("eval_loss")
        if loss is None:
            return
        loss = float(loss)
        best = "" if self._best_eval is None else (" (new best)" if loss < self._best_eval
                                                     else f" (best {self._best_eval:.4f})")
        self._best_eval = loss if self._best_eval is None else min(loss, self._best_eval)
        self._stats["eval_loss"] = round(loss, 4)
        self._post["eval"] = f"{loss:.3f}"
        self._write(f"step {state.global_step}: eval loss {loss:.4f}, perplexity {math.exp(min(loss, 50)):.2f}{best}")
        self._dump(state, force=True)

    def on_save(self, args, state, control, **kwargs):
        self._write(f"step {state.global_step}: checkpoint saved (Ctrl+C now loses nothing; --resume continues)")

    def on_train_end(self, args, state, control, **kwargs):
        self._dump(state, phase="finished", force=True)
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class StopFile(TrainerCallback):
    """Stops the run cleanly when the file ``<run>/STOP`` appears (the GUI's Stop button).

    After the current step a checkpoint is saved, then KeyboardInterrupt ends the run exactly
    like Ctrl+C would, so it is marked "interrupted" and ``--resume`` continues from there.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stopping = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stopping and self.path.exists():
            self.stopping = True
            control.should_save = True
            tqdm.write(f"step {state.global_step}: stop requested, saving a checkpoint ...")
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.stopping:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise KeyboardInterrupt("stopped from the GUI")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_run(run: str) -> dict:
    p = config.run_dir(run) / "run.json"
    if not p.exists():
        raise TrainError(f"No run named {run!r} under {config.RUNS_DIR}.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_run(run: str, info: dict) -> None:
    rd = config.run_dir(run)
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / "run.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    tmp.replace(rd / "run.json")


def latest_checkpoint(run: str) -> Path | None:
    cdir = config.run_dir(run) / "checkpoints"
    cps = [p for p in cdir.glob("checkpoint-*") if p.is_dir()] if cdir.exists() else []
    if not cps:
        return None
    return max(cps, key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1)


def adapter_dir(run: str) -> Path:
    """The final adapter, or the newest checkpoint while a run is still in progress."""
    final = config.run_dir(run) / "adapter"
    if (final / "adapter_config.json").exists():
        return final
    cp = latest_checkpoint(run)
    if cp is not None and (cp / "adapter_config.json").exists():
        return cp
    raise TrainError(f"Run {run!r} has no adapter yet (no adapter/ folder and no checkpoint).")


def list_runs() -> list[dict]:
    out = []
    if not config.RUNS_DIR.exists():
        return out
    for d in sorted(config.RUNS_DIR.iterdir()):
        if (d / "run.json").exists():
            try:
                out.append(read_run(d.name))
            except (TrainError, ValueError):
                continue
    return out


def _eval_losses(log_history: list[dict]) -> tuple[float | None, float | None]:
    evals = [h["eval_loss"] for h in log_history if "eval_loss" in h]
    if not evals:
        return None, None
    return round(float(evals[0]), 4), round(float(evals[-1]), 4)


def train(cfg: TrainConfig, progress=print) -> dict:
    rd = config.run_dir(cfg.run)
    existing = (rd / "run.json").exists()
    if existing and not cfg.resume:
        info = read_run(cfg.run)
        if info.get("status") == "finished":
            raise TrainError(f"Run {cfg.run!r} already finished; pick another --run name (or --resume to continue).")
    local = model_mod.find_local(cfg.model)
    dev, dt = model_mod.resolve_device_dtype(cfg.device, cfg.dtype)
    if not dev.startswith("cuda"):
        log.warning("Training on %s: expect it to be very slow.", dev)

    train_ds, val_ds, meta = data_mod.load_tokenized(cfg.data, cfg.seq_len)
    tok = model_mod.load_tokenizer(local)
    progress(f"data {cfg.data}: {len(train_ds)} training sequences of {cfg.seq_len} tokens"
             + (f", {len(val_ds)} validation sequences" if val_ds is not None else ", no validation split"))

    progress(f"loading {local.repo_id} on {dev} ({model_mod.engine.dtype_name(dt)}) ...")
    model = model_mod.load_model(local, dev, dt, training=True)
    model = model_mod.add_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout,
                               target_modules=cfg.target_modules, train_embeddings=cfg.train_embeddings,
                               gradient_checkpointing=cfg.gradient_checkpointing)
    trainable, total = model_mod.param_summary(model)
    progress(f"LoRA r={cfg.lora_r} alpha={cfg.lora_alpha} on {', '.join(cfg.target_modules)}: "
             f"{trainable/1e6:.1f}M trainable of {total/1e6:.0f}M parameters")

    info = {
        "run": cfg.run, "status": "running", "started_at": _now(), "finished_at": None,
        "model": local.repo_id, "model_path": str(local.path), "data": cfg.data, "data_meta": meta,
        "config": asdict(cfg), "device": dev, "dtype": model_mod.engine.dtype_name(dt),
        "trainable_params": trainable, "total_params": total,
    }
    if existing and cfg.resume:
        info["resumed_from"] = read_run(cfg.run).get("started_at")
    write_run(cfg.run, info)

    eval_subset = None
    if val_ds is not None:
        eval_subset = val_ds.select(range(min(len(val_ds), cfg.eval_sequences)))
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    use_cuda = dev.startswith("cuda")
    args = TrainingArguments(
        output_dir=str(rd / "checkpoints"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.scheduler,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if eval_subset is not None else "no",
        eval_steps=cfg.eval_steps,
        eval_on_start=eval_subset is not None,
        fp16=use_cuda and dt == torch.float16,
        bf16=dt == torch.bfloat16,
        report_to=[],
        dataloader_num_workers=0,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        label_names=["labels"],
        include_num_input_tokens_seen=True,
        seed=cfg.seed,
        use_cpu=not use_cuda,
        disable_tqdm=False,
    )
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=train_ds,
                      eval_dataset=eval_subset, processing_class=tok, callbacks=[JsonlLogger(rd / "log.jsonl")])
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(ConsoleProgress(rd / "progress.json"))
    stop_file = rd / "STOP"
    stop_file.unlink(missing_ok=True)  # a leftover from an earlier stop must not end this run at once
    trainer.add_callback(StopFile(stop_file))
    resume_from = latest_checkpoint(cfg.run) if cfg.resume else None
    if resume_from is not None:
        progress(f"resuming from {resume_from.name}")
    steps = cfg.max_steps if cfg.max_steps > 0 else math.ceil(
        math.ceil(len(train_ds) / cfg.batch_size) / cfg.grad_accum * cfg.epochs)
    step_tokens = cfg.batch_size * cfg.grad_accum * cfg.seq_len
    hours = steps * step_tokens / config.EXPECTED_TOKENS_PER_S / 3600
    progress(f"{steps:,} optimizer steps of up to {step_tokens:,} tokens; about {hours:.1f} h at "
             f"{config.EXPECTED_TOKENS_PER_S:,} tokens/s (the bar's estimate takes over after a few steps). "
             f"Ctrl+C stops, --resume continues from the last checkpoint (every {cfg.save_steps} steps).")
    t0 = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except BaseException as e:  # KeyboardInterrupt included: leave a record of what happened
        info.update(status="failed" if not isinstance(e, KeyboardInterrupt) else "interrupted",
                    finished_at=_now(), error=f"{type(e).__name__}: {e}"[:500],
                    steps=trainer.state.global_step)
        write_run(cfg.run, info)
        raise
    seconds = time.time() - t0

    trainer.model.save_pretrained(str(rd / "adapter"))
    start_loss, end_loss = _eval_losses(trainer.state.log_history)
    info.update(
        status="finished", finished_at=_now(), steps=trainer.state.global_step,
        seconds=round(seconds, 1), train_loss=round(float(result.metrics.get("train_loss", 0.0)), 4),
        eval_loss_start=start_loss, eval_loss_end=end_loss,
        tokens_seen=int(getattr(trainer.state, "num_input_tokens_seen", 0) or 0),
        adapter=str(rd / "adapter"),
        vram_max_mb=int(torch.cuda.max_memory_allocated() // 2**20) if torch.cuda.is_available() else None,
    )
    write_run(cfg.run, info)
    return info
