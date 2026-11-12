"""Command-line front end: ``python -m hftrain <command>`` (or ``hftrain.cmd`` on Windows).

    prepare             build a corpus from the QA folder (or named files) and tokenize it
    tokenize            re-tokenize an existing corpus at another --seq-len
    data                list prepared corpora
    train               LoRA fine-tune an installed model on a corpus
    runs                list training runs
    gui                 window with Start/Stop training, progress bar, console and a chat
    eval                perplexity + sample generations (base, adapter, or both)
    export              merge an adapter into the base model and install it for hfserve
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from hfserve.registry import ModelError, human_size
from hfserve.ui import bold, dim, green, red, setup_console, yellow

from . import __version__, config
from .data import DataError

PROG = "hftrain"
SOURCE_HELP = ("data source; repeatable. `file:<path>` (jsonl/json records, txt/md passages) or "
               "`dir:<folder>`, optionally followed by `@max=N,sample=0.1,...`. "
               "Default: the QA folder, dir:training/QA")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass


def _fmt_int(n) -> str:
    return f"{int(n):,}" if isinstance(n, (int, float)) else "-"


# --------------------------------------------------------------------------- commands
def cmd_prepare(args: argparse.Namespace) -> int:
    from . import data as data_mod

    raw = args.source or [f"dir:{config.QA_DIR}"]
    specs = [data_mod.parse_source(s) for s in raw]
    print(bold(f"corpus {args.name}") + f" -> {config.data_dir(args.name)}")
    for s in specs:
        print(f"  {s.describe()}")
    stats = data_mod.prepare_corpus(specs, args.name, val_fraction=args.val_fraction, seed=args.seed,
                                    overwrite=args.overwrite, progress=lambda m: print(dim(m)))
    print()
    print(bold(f"{'SOURCE':<40}{'DOCS':>9}{'VAL':>7}{'TEXT':>8}{'CHAT':>8}{'CHARS':>14}"))
    for key, e in stats["sources"].items():
        print(f"{key[:39]:<40}{e['docs']:>9,}{e['val_docs']:>7,}{e['text']:>8,}{e['chat']:>8,}{e['chars']:>14,}")
    print(f"{'total':<40}{stats['train_docs'] + stats['val_docs']:>9,}{stats['val_docs']:>7,}"
          f"{'':>16}{stats['train_chars'] + stats['val_chars']:>14,}")
    if args.no_tokenize:
        print(f"\nNot tokenized (--no-tokenize). Later: {PROG} tokenize --name {args.name} --seq-len {args.seq_len}")
        return 0
    return _tokenize(args.name, args.seq_len, args.model, overwrite=True)


def _tokenize(name: str, seq_len: int, model_ref: str | None, overwrite: bool) -> int:
    from . import data as data_mod, model as model_mod

    local = model_mod.find_local(model_ref)
    tok = model_mod.load_tokenizer(local)
    print(f"\ntokenizing with {local.repo_id} at seq_len {seq_len} ...")
    meta = data_mod.tokenize_corpus(name, tok, seq_len, tokenizer_id=local.repo_id, overwrite=overwrite,
                                    progress=lambda m: print(dim(m)))
    tr, va = meta.get("train") or {}, meta.get("val") or {}
    print(green(f"train: {_fmt_int(tr.get('sequences'))} sequences, {_fmt_int(tr.get('tokens'))} tokens "
                f"({_fmt_int(tr.get('target_tokens'))} targets; {_fmt_int(tr.get('chat_examples'))} chat examples)"))
    if va:
        print(green(f"val:   {_fmt_int(va.get('sequences'))} sequences, {_fmt_int(va.get('tokens'))} tokens"))
    else:
        print(yellow("val:   none (raise --val-fraction to get perplexity numbers)"))
    if tr.get("chat_truncated"):
        print(yellow(f"{tr['chat_truncated']} chat examples were cut to {seq_len} tokens"))
    print(f"\nNext: {PROG} train --data {name} --run <run-name>")
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    return _tokenize(args.name, args.seq_len, args.model, overwrite=args.overwrite)


def cmd_data(args: argparse.Namespace) -> int:
    from .data import list_corpora

    corpora = list_corpora()
    if not corpora:
        print(f"No corpora under {config.DATA_DIR}. Put .jsonl files into {config.QA_DIR} "
              f"and run: {PROG} prepare --name qa-v1")
        return 0
    for c in corpora:
        st = c["stats"]
        toks = ", ".join(f"seq {t['seq_len']}: {_fmt_int((t.get('train') or {}).get('tokens'))} tokens"
                         for t in c["tokenized"]) or "not tokenized"
        print(bold(c["name"]) + f"  {_fmt_int(st['train_docs'])} train / {_fmt_int(st['val_docs'])} val documents, "
              f"{human_size(st['train_chars'] + st['val_chars'])} of text  [{toks}]")
        print(dim("  sources: " + ", ".join(st.get("specs", []))))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    cfg = TrainConfig(
        data=args.data, run=args.run, model=args.model or config.DEFAULT_MODEL, seq_len=args.seq_len,
        batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, epochs=args.epochs,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        scheduler=args.scheduler, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=tuple(m.strip() for m in args.target_modules.split(",") if m.strip()),
        train_embeddings=args.train_embeddings, gradient_checkpointing=not args.no_gradient_checkpointing,
        save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps,
        eval_sequences=args.eval_sequences, seed=args.seed, device=args.device, dtype=args.dtype,
        resume=args.resume,
    )
    print(bold(f"run {cfg.run}") + f" -> {config.run_dir(cfg.run)}")
    try:
        info = train(cfg, progress=print)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(red("CUDA out of memory."))
            print("Try a smaller --batch-size (and a larger --grad-accum to keep the effective batch), a smaller "
                  "--seq-len, or a lower --lora-r. Close other GPU programs, including a running hfserve server.")
            return 1
        raise
    print()
    print(green(f"finished: {info['steps']} steps, train loss {info['train_loss']}, "
                f"eval loss {info.get('eval_loss_start')} -> {info.get('eval_loss_end')}, "
                f"{_fmt_int(info.get('tokens_seen'))} tokens in {info['seconds']:.0f}s"
                + (f", peak VRAM {info['vram_max_mb']} MB" if info.get("vram_max_mb") else "")))
    print(f"adapter: {info['adapter']}")
    print(f"Next: {PROG} eval --run {cfg.run} --compare-base   then   {PROG} export --run {cfg.run} --name local/<new-name>")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    form = {"data": args.data, "run": args.run, "model": args.model, "lora_r": args.lora_r, "lr": args.lr,
            "epochs": args.epochs}
    return gui_main(form={k: v for k, v in form.items() if v is not None}, start=args.start)


def cmd_runs(args: argparse.Namespace) -> int:
    from .train import list_runs

    runs = list_runs()
    if not runs:
        print(f"No runs under {config.RUNS_DIR}.")
        return 0
    print(bold(f"{'RUN':<20}{'STATUS':<12}{'DATA':<14}{'STEPS':>7}{'LOSS':>8}{'EVAL':>16}  MODEL"))
    for r in runs:
        ev = "-"
        if r.get("eval_loss_start") is not None:
            ev = f"{r['eval_loss_start']:.3f}->{r['eval_loss_end']:.3f}" if r.get("eval_loss_end") is not None else f"{r['eval_loss_start']:.3f}"
        loss = f"{r['train_loss']:.3f}" if isinstance(r.get("train_loss"), (int, float)) else "-"
        print(f"{r['run'][:19]:<20}{r.get('status', '-'):<12}{str(r.get('data', '-'))[:13]:<14}"
              f"{_fmt_int(r.get('steps')):>7}{loss:>8}{ev:>16}  {r.get('model', '-')}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, format_report, load_prompts

    if not args.run and not args.model and not args.data:
        print(dim(f"evaluating the base model {config.DEFAULT_MODEL}; give --run <run> to evaluate an adapter"))
    prompts = load_prompts(Path(args.prompts_file) if args.prompts_file else None, args.prompt or [])
    report = evaluate(
        run=args.run, model_ref=args.model, data=args.data, seq_len=args.seq_len, compare_base=args.compare_base,
        prompts=prompts, max_new_tokens=args.max_new_tokens, no_generate=args.no_generate,
        no_perplexity=args.no_perplexity, max_sequences=args.sequences, device=args.device, dtype=args.dtype,
        reasoning_effort=args.reasoning_effort, think=args.think, temperature=args.temperature, seed=args.seed,
        progress=print,
    )
    print()
    print(format_report(report))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_merged

    local = export_merged(args.run, args.name, dtype=args.dtype, force=args.force, model_ref=args.model,
                          progress=print)
    print(green(f"installed {local.repo_id} ({human_size(local.size_bytes)}) at {local.path}"))
    print(f"Serve it:  hfserve run {local.repo_id}      (models/models.json updated)")
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="Fine-tune an installed hfserve model on Croatian (LoRA).")
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sp = sub.add_parser("prepare", help="build a corpus from the QA folder (or named files) and tokenize it")
    sp.add_argument("--name", default="qa-v1", help="corpus name (default qa-v1)")
    sp.add_argument("--source", action="append", help=SOURCE_HELP)
    sp.add_argument("--val-fraction", type=float, default=config.DEFAULT_VAL_FRACTION)
    sp.add_argument("--seq-len", type=int, default=config.DEFAULT_SEQ_LEN)
    sp.add_argument("--model", help="model whose tokenizer is used")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--no-tokenize", action="store_true", help="only write train/val.jsonl")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("tokenize", help="(re)tokenize an existing corpus")
    sp.add_argument("--name", required=True)
    sp.add_argument("--seq-len", type=int, default=config.DEFAULT_SEQ_LEN)
    sp.add_argument("--model")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_tokenize)

    sp = sub.add_parser("data", help="list prepared corpora")
    sp.set_defaults(func=cmd_data)

    sp = sub.add_parser("train", help="LoRA fine-tune a model on a prepared corpus")
    sp.add_argument("--data", required=True, help="corpus name from `prepare`")
    sp.add_argument("--run", required=True, help="run name, e.g. qa-lora-1")
    sp.add_argument("--model", help=f"installed model to start from (default {config.DEFAULT_MODEL})")
    sp.add_argument("--seq-len", type=int, default=config.DEFAULT_SEQ_LEN)
    sp.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    sp.add_argument("--grad-accum", type=int, default=config.DEFAULT_GRAD_ACCUM)
    sp.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    sp.add_argument("--epochs", type=float, default=1.0)
    sp.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps")
    sp.add_argument("--warmup-steps", type=int, default=config.DEFAULT_WARMUP_STEPS)
    sp.add_argument("--weight-decay", type=float, default=0.0)
    sp.add_argument("--scheduler", default="cosine", help="cosine, linear, constant, constant_with_warmup")
    sp.add_argument("--lora-r", type=int, default=config.DEFAULT_LORA_R)
    sp.add_argument("--lora-alpha", type=int, default=config.DEFAULT_LORA_ALPHA)
    sp.add_argument("--lora-dropout", type=float, default=config.DEFAULT_LORA_DROPOUT)
    sp.add_argument("--target-modules", default=",".join(config.DEFAULT_TARGET_MODULES))
    sp.add_argument("--train-embeddings", action="store_true",
                    help="also train embed_tokens and lm_head fully (needs much more VRAM)")
    sp.add_argument("--no-gradient-checkpointing", action="store_true", help="faster, needs more VRAM")
    sp.add_argument("--save-steps", type=int, default=200)
    sp.add_argument("--eval-steps", type=int, default=200)
    sp.add_argument("--logging-steps", type=int, default=10)
    sp.add_argument("--eval-sequences", type=int, default=200, help="validation sequences per evaluation")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--device", help="cuda or cpu")
    sp.add_argument("--dtype", help="float16, bfloat16 or float32")
    sp.add_argument("--resume", action="store_true", help="continue from the run's last checkpoint")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("runs", help="list training runs")
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("gui", help="window with Start/Stop training, progress bar, console and a chat")
    sp.add_argument("--data", help="fill in the corpus")
    sp.add_argument("--run", help="fill in the run name")
    sp.add_argument("--model", help="fill in the base model")
    sp.add_argument("--lora-r", type=int, help="fill in the LoRA rank")
    sp.add_argument("--lr", type=float, help="fill in the learning rate")
    sp.add_argument("--epochs", type=float, help="fill in the epochs")
    sp.add_argument("--start", action="store_true", help="press Start right away")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("eval", help="perplexity and sample generations")
    sp.add_argument("--run", help="evaluate this run's adapter (omit for the base model)")
    sp.add_argument("--model", help="model to evaluate / base for the adapter")
    sp.add_argument("--data", help="corpus for perplexity (default: the run's corpus)")
    sp.add_argument("--seq-len", type=int)
    sp.add_argument("--compare-base", action="store_true", help="also evaluate the base model, side by side")
    sp.add_argument("--prompt", action="append", help="extra chat prompt; repeatable")
    sp.add_argument("--prompts-file", help="one prompt per line; prefix 'completion:' for plain continuation")
    sp.add_argument("--max-new-tokens", type=int, default=160)
    sp.add_argument("--sequences", type=int, default=200, help="validation sequences for perplexity")
    sp.add_argument("--no-generate", action="store_true")
    sp.add_argument("--no-perplexity", action="store_true")
    sp.add_argument("--reasoning-effort", default="low", help="reasoning models only: high, medium, low")
    sp.add_argument("--think", action="store_true", help="let the model reason before answering")
    sp.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (model card: 0.6); 0 = greedy")
    sp.add_argument("--seed", type=int, default=1, help="sampling seed, reused for every prompt and for base vs adapter")
    sp.add_argument("--device")
    sp.add_argument("--dtype")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("export", help="merge a run's adapter into the base model and install it for hfserve")
    sp.add_argument("--run", required=True)
    sp.add_argument("--name", required=True, help="new model name, e.g. local/EuroLLM-1.7B-hr")
    sp.add_argument("--model", help="base model (default: the run's model)")
    sp.add_argument("--dtype", default="bfloat16", help="dtype of the saved weights")
    sp.add_argument("--force", action="store_true", help="replace an existing model of that name")
    sp.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    _setup_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except (DataError, ModelError) as e:
        print(red(f"Error: {e}"))
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:  # noqa: BLE001
        from .train import TrainError

        if isinstance(e, TrainError):
            print(red(f"Error: {e}"))
            return 1
        traceback.print_exc()
        print(red(f"Error: {e}"))
        return 1
