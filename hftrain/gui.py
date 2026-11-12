"""Desktop window for hftrain: prepare the QA corpus, start and stop training with a progress
bar and a console, and chat with a model (or a run's LoRA adapter) to check its Croatian.

    python -m hftrain gui            (or double-click HFTrain-GUI.cmd)

"Prepare QA data" runs ``python -m hftrain prepare`` on ``training/QA`` as a child process.
Training runs as a child process, ``python -m hftrain train ...``, exactly as from a console:
its output fills the console panel and ``<run>/progress.json`` drives the progress bar. Stop
writes ``<run>/STOP``; the run saves a checkpoint and ends as "interrupted", and Start with
"Resume" continues it. The chat loads the model inside this window's process, on the GPU, or on
the CPU while a training run holds the GPU (both together would not fit in 8 GB).
"""
from __future__ import annotations

import codecs
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from hfserve import registry

from . import config, data as data_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# harmless load-time warnings some models' code triggers; not shown in the console
NOISE_RE = re.compile(r"`cache_position` is part of|seems to expect `cache_position`"
                      # tqdm bars that end up on a line of their own; the window has its own progress bar
                      r"|^\s*(training|evaluating:?|Loading weights:?)\s+\d+%\|")
GPU_QUERY = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"]
GPU_POLL_S = 2.0
MAX_CONSOLE_LINES = 5000
HEARTBEAT_S = 60          # console progress line at most this often
REPETITION_PENALTY = 1.1  # small models loop without it
LORA_RANKS = ("8", "16", "32", "64", "128")


# --------------------------------------------------------------------------- helpers (no Tk)
def fmt_hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def split_output(buf: str) -> tuple[list[tuple[str, str]], str]:
    """Split decoded child output into ``("line", text)`` for text ended by a newline and
    ``("live", text)`` for carriage-return redraws (progress bars). Returns the parts and the
    unterminated rest; a trailing ``\\r`` is held back because ``\\n`` may follow it."""
    parts: list[tuple[str, str]] = []
    hold = ""
    if buf.endswith("\r"):
        buf, hold = buf[:-1], "\r"
    start = 0
    for m in re.finditer(r"\r\n|\n|\r", buf):
        parts.append(("live" if m.group() == "\r" else "line", ANSI_RE.sub("", buf[start:m.start()])))
        start = m.end()
    return parts, buf[start:] + hold


def list_runs_light() -> list[dict]:
    """run.json of every run plus whether it has an adapter or checkpoint (without importing torch)."""
    out: list[dict] = []
    if not config.RUNS_DIR.exists():
        return out
    for d in sorted(config.RUNS_DIR.iterdir()):
        p = d / "run.json"
        if not p.is_file():
            continue
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cdir = d / "checkpoints"
        cps = [c for c in cdir.glob("checkpoint-*") if c.is_dir()] if cdir.is_dir() else []
        info["run"] = d.name
        info["has_checkpoint"] = bool(cps)
        info["has_adapter"] = (d / "adapter" / "adapter_config.json").exists() or any(
            (c / "adapter_config.json").exists() for c in cps)
        out.append(info)
    return out


def next_run_name(names: list[str], prefix: str = "qa-lora-") -> str:
    nums = [int(m.group(1)) for n in names if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", n))]
    return f"{prefix}{max(nums, default=0) + 1}"


def next_corpus_name(names: list[str], prefix: str = "qa-v") -> str:
    nums = [int(m.group(1)) for n in names if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", n))]
    return f"{prefix}{max(nums, default=0) + 1}"


def corpus_tokens(corpus: dict, seq_len: int = config.DEFAULT_SEQ_LEN) -> int | None:
    for meta in corpus.get("tokenized", []):
        if meta.get("seq_len") == seq_len:
            return (meta.get("train") or {}).get("tokens")
    return None


def build_train_argv(python: str, *, data: str, run: str, model: str, lora_r: int, lr: float, epochs: float,
                     resume: bool) -> list[str]:
    argv = [python, "-u", "-m", "hftrain", "train", "--data", data, "--run", run, "--model", model,
            "--lora-r", str(lora_r), "--lora-alpha", str(2 * lora_r), "--lr", repr(lr), "--epochs", repr(epochs)]
    if resume:
        argv.append("--resume")
    return argv


def gpu_status_line(csv: str) -> str:
    """One status-bar row from ``nvidia-smi`` CSV output (first GPU)."""
    first = csv.strip().splitlines()[0] if csv.strip() else ""
    f = [x.strip() for x in first.split(",")]
    if len(f) < 4:
        return "GPU: no NVIDIA GPU found"

    def num(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    util, used, total, temp, power = (num(x) for x in (f[1:] + ["", ""])[:5])
    parts = [f"GPU: {f[0]}"]
    if util is not None:
        parts.append(f"load {util:.0f}%")
    if used is not None and total:
        parts.append(f"VRAM {used / 1024:.1f} / {total / 1024:.1f} GB ({100 * used / total:.0f}%)")
    if temp is not None:
        parts.append(f"{temp:.0f} °C")
    if power is not None:
        parts.append(f"{power:.0f} W")
    return "   |   ".join(parts)


def child_python() -> str:
    """python.exe next to pythonw.exe: the child needs a real stdout for its output."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def progress_line(p: dict) -> str:
    step, total = p.get("step", 0), p.get("max_steps") or 0
    parts = [f"step {step:,}/{total:,}" + (f" ({100 * step / total:.1f}%)" if total else "")]
    if p.get("loss") is not None:
        parts.append(f"loss {p['loss']:.3f}")
    if p.get("eval_loss") is not None:
        parts.append(f"eval {p['eval_loss']:.3f}")
    if p.get("tokens_per_s"):
        parts.append(f"{p['tokens_per_s']:,.0f} tok/s")
    if p.get("vram_total_mb"):
        parts.append(f"VRAM {p['vram_reserved_mb'] / 1024:.1f}/{p['vram_total_mb'] / 1024:.1f} GB")
    parts.append(f"{fmt_hms(p.get('elapsed_s'))} elapsed, {fmt_hms(p.get('remaining_s'))} left")
    return " | ".join(parts)


# --------------------------------------------------------------------------- training child process
class TrainProcess:
    """``python -m hftrain train ...`` with its output pushed to a queue as ("out", kind, text)
    events and a final ("exit", code)."""

    def __init__(self, argv: list[str], events: queue.Queue) -> None:
        self.events = events
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, cwd=str(PROJECT_ROOT), creationflags=flags)
        threading.Thread(target=self._read, name="hftrain-gui-reader", daemon=True).start()

    def _read(self) -> None:
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        rest = ""
        while True:
            chunk = self.proc.stdout.read1(65536)
            if not chunk:
                break
            parts, rest = split_output(rest + dec.decode(chunk))
            for kind, text in parts:
                self.events.put(("out", kind, text))
        tail = ANSI_RE.sub("", rest + dec.decode(b"", final=True)).strip("\r\n")
        if tail.strip():
            self.events.put(("out", "line", tail))
        self.events.put(("exit", self.proc.wait()))

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        if self.running:
            self.proc.kill()


class QueueLogHandler(logging.Handler):
    def __init__(self, events: queue.Queue) -> None:
        super().__init__(logging.INFO)
        self.events = events
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.events.put(("log", self.format(record)))
        except Exception:  # noqa: BLE001 - logging must never break the window
            pass


# --------------------------------------------------------------------------- window
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.train: TrainProcess | None = None
        self.prep: TrainProcess | None = None
        self.prep_name: str | None = None
        self.train_run: str | None = None
        self.train_started = 0.0
        self.stop_requested = False
        self.close_after_exit = False
        self.last_heartbeat = 0.0
        self.last_step = -1
        self.last_progress_poll = 0.0
        self.corpora: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.chat_choices: dict[str, tuple[str, str | None]] = {}
        self.engine = None              # hfserve Engine, created once torch is imported
        self.engine_ready = False
        self.chat_device: str | None = None
        self.chat_busy = False
        self.chat_cancel: threading.Event | None = None
        self.history: list[dict] = []
        self.answer_kind: str | None = None
        self.answer_text = ""

        root.title("hftrain - fine-tuning EuroLLM")
        root.geometry("1100x800")
        root.minsize(820, 560)
        self._style()
        status = ttk.Frame(root, padding=(12, 2, 12, 6))
        status.pack(side="bottom", fill="x")
        ttk.Separator(root, orient="horizontal").pack(side="bottom", fill="x", padx=10)
        self.l_gpu = ttk.Label(status, text="GPU: ...", style="Muted.TLabel")
        self.l_gpu.pack(side="left")
        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.train_tab = ttk.Frame(nb, padding=10)
        self.chat_tab = ttk.Frame(nb, padding=10)
        nb.add(self.train_tab, text="  Training  ")
        nb.add(self.chat_tab, text="  Chat  ")
        self._build_train_tab()
        self._build_chat_tab()
        self.refresh_lists()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(50, self._poll)
        threading.Thread(target=self._import_engine, name="hftrain-gui-import", daemon=True).start()
        threading.Thread(target=self._watch_gpu, name="hftrain-gui-gpu", daemon=True).start()

    def _watch_gpu(self) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        while True:
            try:
                out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=10, creationflags=flags)
                line = gpu_status_line(out.stdout) if out.returncode == 0 else "GPU: nvidia-smi failed"
            except (OSError, subprocess.SubprocessError):
                line = "GPU: nvidia-smi not found"
            self.events.put(("gpu", line))
            if line.endswith("not found"):
                return
            time.sleep(GPU_POLL_S)

    # ----------------------------------------------------------------- look
    def _style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        tkfont.nametofont("TkDefaultFont").configure(family="Segoe UI", size=10)
        tkfont.nametofont("TkTextFont").configure(family="Segoe UI", size=10)
        style.configure("Big.TButton", padding=(14, 6))
        style.configure("Muted.TLabel", foreground="#666666")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

    # ----------------------------------------------------------------- training tab
    def _build_train_tab(self) -> None:
        t = self.train_tab
        form = ttk.LabelFrame(t, text="What to learn", padding=10)
        form.pack(fill="x")
        self.v_corpus = tk.StringVar()
        self.v_model = tk.StringVar(value=config.DEFAULT_MODEL)
        self.v_run = tk.StringVar()
        self.v_rank = tk.StringVar(value=str(config.DEFAULT_LORA_R))
        self.v_lr = tk.StringVar(value=f"{config.DEFAULT_LR:g}")
        self.v_epochs = tk.StringVar(value="1")
        self.v_resume = tk.BooleanVar(value=False)

        def row(r: int, c: int, label: str, widget: tk.Widget) -> None:
            ttk.Label(form, text=label).grid(row=r, column=c, sticky="w", padx=(0 if c == 0 else 18, 6), pady=4)
            widget.grid(row=r, column=c + 1, sticky="we", pady=4)

        self.cb_corpus = ttk.Combobox(form, textvariable=self.v_corpus, state="readonly", width=28)
        self.cb_model = ttk.Combobox(form, textvariable=self.v_model, state="readonly", width=34)
        cb_rank = ttk.Combobox(form, textvariable=self.v_rank, values=LORA_RANKS, width=8)
        e_run = ttk.Entry(form, textvariable=self.v_run, width=30)
        e_lr = ttk.Entry(form, textvariable=self.v_lr, width=12)
        e_epochs = ttk.Entry(form, textvariable=self.v_epochs, width=8)
        ck_resume = ttk.Checkbutton(form, text="Resume this run from its last checkpoint", variable=self.v_resume,
                                    command=self._on_resume_toggle)
        row(0, 0, "Corpus", self.cb_corpus)
        row(0, 2, "Base model", self.cb_model)
        row(1, 0, "Run name", e_run)
        row(1, 2, "LoRA rank", cb_rank)
        row(2, 0, "Learning rate", e_lr)
        row(2, 2, "Epochs", e_epochs)
        ck_resume.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        # (widget, state when enabled); locked while a run trains, so the form shows what is running
        self.form_widgets = [(self.cb_corpus, "readonly"), (self.cb_model, "readonly"), (cb_rank, "normal"),
                             (e_run, "normal"), (e_lr, "normal"), (e_epochs, "normal"), (ck_resume, "normal")]
        for cb in (self.cb_corpus, self.cb_model, cb_rank):
            cb.bind("<MouseWheel>", lambda e: "break")  # a stray scroll must not switch corpus or model
        self.l_estimate = ttk.Label(form, text="", style="Muted.TLabel")
        self.l_estimate.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        for v in (self.v_corpus, self.v_epochs):
            v.trace_add("write", lambda *_: self._update_estimate())
        self.v_run.trace_add("write", lambda *_: self._on_run_name())

        bar = ttk.Frame(t)
        bar.pack(fill="x", pady=(12, 6))
        self.b_prepare = ttk.Button(bar, text="Prepare QA data", style="Big.TButton", command=self.start_prepare)
        self.b_prepare.pack(side="left")
        self.b_start = ttk.Button(bar, text="Start training", style="Big.TButton", command=self.start_training)
        self.b_start.pack(side="left", padx=8)
        self.b_stop = ttk.Button(bar, text="Stop", style="Big.TButton", command=self.stop_training, state="disabled")
        self.b_stop.pack(side="left")
        self.l_status = ttk.Label(bar, text="Idle", style="Status.TLabel")
        self.l_status.pack(side="left", padx=12)

        self.pb = ttk.Progressbar(t, orient="horizontal", mode="determinate")
        self.pb.pack(fill="x", pady=(4, 2), ipady=4)
        self.l_progress = ttk.Label(t, text="", style="Muted.TLabel")
        self.l_progress.pack(fill="x")

        ttk.Label(t, text="Console").pack(anchor="w", pady=(10, 2))
        self.console = ScrolledText(t, height=18, wrap="word", font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4",
                                    insertbackground="#d4d4d4", relief="flat", borderwidth=0, padx=8, pady=6)
        self.console.pack(fill="both", expand=True)
        self.console.tag_configure("cmd", foreground="#9cdcfe")
        self.console.tag_configure("note", foreground="#c586c0")
        self.console.tag_configure("good", foreground="#6a9955")
        self.console.tag_configure("bad", foreground="#f48771")
        self.console.tag_configure("beat", foreground="#dcdcaa")
        self.console.configure(state="disabled")

    def log(self, text: str, tag: str | None = None) -> None:
        c = self.console
        at_end = c.yview()[1] > 0.999
        c.configure(state="normal")
        c.insert("end", text + "\n", (tag,) if tag else ())
        lines = int(c.index("end-1c").split(".")[0])
        if lines > MAX_CONSOLE_LINES:
            c.delete("1.0", f"{lines - MAX_CONSOLE_LINES}.0")
        c.configure(state="disabled")
        if at_end:
            c.see("end")

    def refresh_lists(self) -> None:
        self.corpora = {c["name"]: c for c in data_mod.list_corpora()}
        names = list(self.corpora)
        self.cb_corpus["values"] = names
        if self.v_corpus.get() not in self.corpora and names:
            self.v_corpus.set(names[-1])
        models = [m.repo_id for m in registry.list_models() if not m.incomplete]
        self.cb_model["values"] = models
        if self.v_model.get() not in models and models:
            self.v_model.set(models[0])
        self.runs = {r["run"]: r for r in list_runs_light()}
        current = self.runs.get(self.v_run.get().strip())
        if not self.v_run.get().strip() or (current and current.get("status") == "finished"):
            self.v_run.set(next_run_name(list(self.runs)))
        else:
            self._on_run_name()  # a run that just stopped becomes resumable
        self._refresh_chat_models()
        self._update_estimate()

    def _on_run_name(self) -> None:
        info = self.runs.get(self.v_run.get().strip())
        resumable = bool(info and info.get("status") != "finished" and info.get("has_checkpoint"))
        self.v_resume.set(resumable)
        if resumable:
            self._fill_from_run(info)
        self._update_estimate()

    def _on_resume_toggle(self) -> None:
        info = self.runs.get(self.v_run.get().strip())
        if self.v_resume.get():
            if not info or not info.get("has_checkpoint"):
                messagebox.showinfo("Resume", f"Run {self.v_run.get()!r} has no checkpoint to resume from.")
                self.v_resume.set(False)
                return
            self._fill_from_run(info)

    def _fill_from_run(self, info: dict) -> None:
        cfg = info.get("config") or {}
        if cfg.get("data") in self.corpora:
            self.v_corpus.set(cfg["data"])
        if cfg.get("model"):
            self.v_model.set(cfg["model"])
        if cfg.get("lora_r") is not None:
            self.v_rank.set(str(cfg["lora_r"]))
        if cfg.get("lr") is not None:
            self.v_lr.set(f"{cfg['lr']:g}")
        if cfg.get("epochs") is not None:
            self.v_epochs.set(f"{cfg['epochs']:g}")

    def _update_estimate(self) -> None:
        if not hasattr(self, "l_estimate"):
            return
        c = self.corpora.get(self.v_corpus.get())
        if not c:
            self.l_estimate.configure(text=f"No prepared corpus yet: put .jsonl files into {config.QA_DIR} "
                                           f"and press 'Prepare QA data'.")
            return
        tokens = corpus_tokens(c)
        if tokens is None:
            self.l_estimate.configure(text=f"{c['name']} is not tokenized for seq {config.DEFAULT_SEQ_LEN} yet: "
                                           f"run  hftrain tokenize --name {c['name']}")
            return
        try:
            epochs = float(self.v_epochs.get())
        except ValueError:
            epochs = 1.0
        hours = tokens * epochs / config.EXPECTED_TOKENS_PER_S / 3600
        self.l_estimate.configure(text=f"{c['name']}: {tokens / 1e6:,.1f}M training tokens x {epochs:g} epoch(s), "
                                       f"about {hours:.1f} h at {config.EXPECTED_TOKENS_PER_S:,} tokens/s")

    def start_prepare(self) -> None:
        """Run ``hftrain prepare`` on the QA folder as a child process, into the console."""
        if (self.train is not None and self.train.running) or (self.prep is not None and self.prep.running):
            return
        qa_files = [p for p in config.QA_DIR.rglob("*")
                    if p.suffix.lower() in (".jsonl", ".json", ".txt", ".md") and p.stem.lower() != "readme"] \
            if config.QA_DIR.exists() else []
        if not qa_files:
            messagebox.showinfo("Prepare QA data",
                                f"No data files found.\n\nPut .jsonl files with your questions and answers into\n"
                                f"{config.QA_DIR}\n(see its README.txt for the formats), then press this again.")
            return
        name = next_corpus_name(list(self.corpora))
        argv = [child_python(), "-u", "-m", "hftrain", "prepare", "--name", name]
        self.log("")
        self.log("$ hftrain " + " ".join(argv[4:]), "cmd")
        try:
            self.prep = TrainProcess(argv, self.events)
        except OSError as e:
            self.log(f"could not start prepare: {e}", "bad")
            self.prep = None
            return
        self.prep_name = name
        self.b_prepare.configure(state="disabled")
        self.b_start.configure(state="disabled")
        self.l_status.configure(text=f"Preparing corpus {name} from {len(qa_files)} file(s) ...")

    def _on_prepare_exit(self, code: int) -> None:
        name = self.prep_name
        self.prep = None
        self.prep_name = None
        self.b_prepare.configure(state="normal")
        self.b_start.configure(state="normal")
        self.refresh_lists()
        if code == 0:
            if name in self.corpora:
                self.v_corpus.set(name)
            self.l_status.configure(text=f"Corpus {name} ready")
            self.log(f"corpus {name} prepared. Press 'Start training'.", "good")
        else:
            self.l_status.configure(text=f"Prepare failed (exit code {code})")
            self.log(f"prepare ended with exit code {code}; see the messages above.", "bad")

    def start_training(self) -> None:
        if (self.train is not None and self.train.running) or (self.prep is not None and self.prep.running):
            return
        run = self.v_run.get().strip()
        try:
            lora_r = int(self.v_rank.get())
            lr = float(self.v_lr.get())
            epochs = float(self.v_epochs.get())
            if lora_r <= 0 or lr <= 0 or epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Settings", "LoRA rank, learning rate and epochs must be positive numbers.")
            return
        if not run or not re.fullmatch(r"[A-Za-z0-9._-]+", run):
            messagebox.showerror("Run name", "Use letters, digits, '.', '_' and '-' for the run name.")
            return
        if not self.v_corpus.get():
            messagebox.showerror("Corpus", "There is no prepared corpus. Run  hftrain prepare  first.")
            return
        resume = self.v_resume.get()
        info = self.runs.get(run)
        if info and not resume:
            if info.get("status") == "finished":
                messagebox.showerror("Run name", f"Run {run!r} is already finished. Pick a new run name.")
                return
            if info.get("has_checkpoint") and not messagebox.askyesno(
                    "Start over?", f"Run {run!r} has checkpoints. Start it over from the beginning?\n\n"
                                   "(Tick 'Resume' to continue it instead.)"):
                return

        if self.engine is not None and self.engine.is_loaded and (self.chat_device or "").startswith("cuda"):
            self.log("unloading the chat model to free the GPU for training", "note")
            self.unload_chat()

        rd = config.run_dir(run)
        for leftover in (rd / "progress.json", rd / "STOP"):
            leftover.unlink(missing_ok=True)
        argv = build_train_argv(child_python(), data=self.v_corpus.get(), run=run, model=self.v_model.get(),
                                lora_r=lora_r, lr=lr, epochs=epochs, resume=resume)
        self.log("")
        self.log("$ hftrain " + " ".join(argv[4:]), "cmd")
        try:
            self.train = TrainProcess(argv, self.events)
        except OSError as e:
            self.log(f"could not start training: {e}", "bad")
            return
        self.train_run = run
        self.train_started = time.time()
        self.stop_requested = False
        self.last_heartbeat = 0.0
        self.last_step = -1
        self.b_start.configure(state="disabled")
        self.b_prepare.configure(state="disabled")
        self.b_stop.configure(state="normal", text="Stop")
        self._lock_form(True)
        self.l_status.configure(text=f"Training {run}: loading ...")
        self.l_progress.configure(text="Loading the data and the model, then a first evaluation (a few minutes)")
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)
        self._set_chat_controls()

    def _lock_form(self, locked: bool) -> None:
        for widget, enabled_state in self.form_widgets:
            widget.configure(state="disabled" if locked else enabled_state)

    def stop_training(self) -> None:
        if self.train is None or not self.train.running:
            return
        if not self.stop_requested:
            self.stop_requested = True
            (config.run_dir(self.train_run) / "STOP").write_text("stop requested from the GUI\n", encoding="utf-8")
            self.log("stop requested: the run saves a checkpoint after the current step, then ends", "note")
            self.l_status.configure(text=f"Stopping {self.train_run} after the current step ...")
            self.b_stop.configure(text="Force stop")
            return
        if messagebox.askyesno("Force stop", "Kill the training process now?\n\n"
                                             "Everything since the last checkpoint is lost."):
            self.train.kill()
            self.log("training process killed", "bad")

    def _mark_killed(self, run: str) -> None:
        p = config.run_dir(run) / "run.json"
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            if info.get("status") == "running":
                info.update(status="interrupted", error="killed from the GUI")
                p.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError):
            pass

    def _poll_progress(self) -> None:
        if self.train is None or not self.train.running or not self.train_run:
            return
        try:
            prog = json.loads((config.run_dir(self.train_run) / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if prog.get("updated", 0) < self.train_started - 1:
            return
        total = prog.get("max_steps") or 0
        step = prog.get("step", 0)
        if not total or prog.get("phase") == "starting":
            return
        if str(self.pb.cget("mode")) != "determinate":
            self.pb.stop()
            self.pb.configure(mode="determinate")
        self.pb.configure(maximum=total, value=step)
        line = progress_line(prog)
        self.l_progress.configure(text=line)
        if not self.stop_requested:
            phase = "evaluating" if prog.get("phase") == "evaluating" else "training"
            self.l_status.configure(text=f"{self.train_run}: {phase}, {100 * step / total:.1f}%")
        now = time.time()
        if step != self.last_step and now - self.last_heartbeat >= HEARTBEAT_S:
            self.last_heartbeat, self.last_step = now, step
            self.log(time.strftime("[%H:%M:%S] ") + line, "beat")

    def _on_train_exit(self, code: int) -> None:
        run = self.train_run
        self.pb.stop()
        self.pb.configure(mode="determinate")
        if code == 0:
            self.pb.configure(value=self.pb.cget("maximum"))
            self.l_status.configure(text=f"{run}: finished")
            self.log(f"training {run} finished. Try it in the Chat tab (model + adapter {run}).", "good")
        elif code == 130:
            self.l_status.configure(text=f"{run}: stopped")
            self.log(f"training {run} stopped. Tick 'Resume' and press Start to continue from the last checkpoint.",
                     "note")
        else:
            if run:
                self._mark_killed(run)
            self.l_status.configure(text=f"{run}: failed (exit code {code})")
            self.log(f"training {run} ended with exit code {code}; see the messages above.", "bad")
        self.train = None
        self._lock_form(False)
        self.b_start.configure(state="normal")
        self.b_prepare.configure(state="normal")
        self.b_stop.configure(state="disabled", text="Stop")
        self.refresh_lists()
        self._set_chat_controls()
        if self.close_after_exit:
            self.root.destroy()

    # ----------------------------------------------------------------- chat tab
    def _build_chat_tab(self) -> None:
        t = self.chat_tab
        top = ttk.Frame(t)
        top.pack(fill="x")
        ttk.Label(top, text="Model").pack(side="left")
        self.v_chat_model = tk.StringVar()
        self.cb_chat = ttk.Combobox(top, textvariable=self.v_chat_model, state="readonly", width=58)
        self.cb_chat.pack(side="left", padx=6)
        self.cb_chat.bind("<MouseWheel>", lambda e: "break")
        self.b_load = ttk.Button(top, text="Load", command=self.load_chat, state="disabled")
        self.b_load.pack(side="left")
        self.b_unload = ttk.Button(top, text="Unload", command=self.unload_chat, state="disabled")
        self.b_unload.pack(side="left", padx=6)
        self.l_chat = ttk.Label(top, text="Preparing PyTorch ...", style="Muted.TLabel")
        self.l_chat.pack(side="left", padx=8)

        opts = ttk.Frame(t)
        opts.pack(fill="x", pady=(8, 6))
        ttk.Label(opts, text="Temperature").pack(side="left")
        self.v_temp = tk.StringVar(value="0.7")
        ttk.Spinbox(opts, from_=0.0, to=1.5, increment=0.1, textvariable=self.v_temp, width=5).pack(side="left", padx=6)
        ttk.Label(opts, text="Max tokens").pack(side="left", padx=(12, 0))
        self.v_maxtok = tk.StringVar(value="512")
        ttk.Spinbox(opts, from_=32, to=4096, increment=64, textvariable=self.v_maxtok, width=6).pack(side="left", padx=6)
        self.v_think = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Let it think first", variable=self.v_think).pack(side="left", padx=12)
        ttk.Button(opts, text="New chat", command=self.new_chat).pack(side="right")

        self.transcript = ScrolledText(t, wrap="word", font=("Segoe UI", 11), relief="flat", borderwidth=1,
                                       padx=12, pady=10, bg="#ffffff")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("you", foreground="#0b5cad", font=("Segoe UI", 11, "bold"), spacing1=10)
        self.transcript.tag_configure("bot", foreground="#1b1b1b", spacing1=2)
        self.transcript.tag_configure("botname", foreground="#2e7d32", font=("Segoe UI", 11, "bold"), spacing1=10)
        self.transcript.tag_configure("think", foreground="#8a8a8a", font=("Segoe UI", 10, "italic"))
        self.transcript.tag_configure("info", foreground="#9a9a9a", font=("Segoe UI", 9))
        self.transcript.configure(state="disabled")

        bottom = ttk.Frame(t)
        bottom.pack(fill="x", pady=(8, 0))
        self.input = tk.Text(bottom, height=3, wrap="word", font=("Segoe UI", 11), relief="solid", borderwidth=1,
                             padx=8, pady=6)
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self._on_enter)
        btns = ttk.Frame(bottom)
        btns.pack(side="left", padx=(8, 0))
        self.b_send = ttk.Button(btns, text="Send", style="Big.TButton", command=self.send, state="disabled")
        self.b_send.pack(fill="x")
        self.b_cancel = ttk.Button(btns, text="Stop", command=self.cancel_answer, state="disabled")
        self.b_cancel.pack(fill="x", pady=(4, 0))
        ttk.Label(t, text="Enter sends, Shift+Enter starts a new line. Pick 'model + adapter <run>' to try a "
                          "training run.", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

    def _refresh_chat_models(self) -> None:
        self.chat_choices = {}
        for m in registry.list_models():
            if not m.incomplete:
                self.chat_choices[m.repo_id] = (m.repo_id, None)
        for r in self.runs.values():
            if r.get("has_adapter") and r.get("model"):
                state = "" if r.get("status") == "finished" else f"  (last checkpoint, {r.get('status')})"
                self.chat_choices[f"{r['model']}  +  adapter {r['run']}{state}"] = (r["model"], r["run"])
        labels = list(self.chat_choices)
        self.cb_chat["values"] = labels
        if self.v_chat_model.get() not in self.chat_choices and labels:
            adapters = [lab for lab in labels if self.chat_choices[lab][1]]
            self.v_chat_model.set(adapters[-1] if adapters else labels[0])

    def _import_engine(self) -> None:
        try:
            from hfserve.engine import Engine
            from . import model as _model, train as _train  # noqa: F401 - warm the imports

            self.events.put(("engine_ready", Engine()))
        except Exception as e:  # noqa: BLE001
            self.events.put(("engine_error", f"{type(e).__name__}: {e}"))

    def _set_chat_controls(self) -> None:
        busy = self.chat_busy
        loaded = self.engine is not None and self.engine.is_loaded
        self.b_load.configure(state="normal" if self.engine_ready and not busy else "disabled")
        self.b_unload.configure(state="normal" if loaded and not busy else "disabled")
        self.b_send.configure(state="normal" if loaded and not busy else "disabled")
        self.b_cancel.configure(state="normal" if busy and self.chat_cancel is not None else "disabled")

    def load_chat(self) -> None:
        label = self.v_chat_model.get()
        if label not in self.chat_choices or self.engine is None or self.chat_busy:
            return
        model_ref, run = self.chat_choices[label]
        training = self.train is not None and self.train.running
        device, dtype = ("cpu", "float32") if training else (None, None)
        self.chat_busy = True
        self.chat_cancel = None
        self._set_chat_controls()
        self.l_chat.configure(text="Loading on the CPU (training is using the GPU) ..." if training else "Loading ...")
        eng = self.engine

        def work() -> None:
            try:
                from . import model as model_mod
                from .train import adapter_dir

                eng.unload()
                eng.load(model_mod.find_local(model_ref), device=device, dtype=dtype)
                if run:
                    eng.model = model_mod.load_adapter(eng.model, adapter_dir(run))
                    eng.model.eval()
                self.events.put(("chat_loaded", label, eng.device))
            except Exception as e:  # noqa: BLE001
                eng.unload()
                self.events.put(("chat_error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hftrain-gui-load", daemon=True).start()

    def unload_chat(self) -> None:
        if self.engine is not None:
            self.engine.unload()
        self.chat_device = None
        self.l_chat.configure(text="No model loaded")
        self._set_chat_controls()

    def new_chat(self) -> None:
        if self.chat_busy and self.chat_cancel is not None:
            self.chat_cancel.set()
        self.history = []
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _say(self, text: str, tag: str) -> None:
        tr = self.transcript
        tr.configure(state="normal")
        tr.insert("end", text, (tag,))
        tr.configure(state="disabled")
        tr.see("end")

    def _on_enter(self, event) -> str | None:
        if event.state & 0x0001:  # Shift: new line
            return None
        self.send()
        return "break"

    def send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text or self.chat_busy or self.engine is None or not self.engine.is_loaded:
            return
        try:
            temp = float(self.v_temp.get())
            max_new = int(self.v_maxtok.get())
        except ValueError:
            temp, max_new = 0.7, 512
        self.input.delete("1.0", "end")
        self.history.append({"role": "user", "content": text})
        self._say("You\n", "you")
        self._say(text + "\n", "bot")
        self._say("Model\n", "botname")
        self.answer_kind, self.answer_text = None, ""
        self.chat_busy = True
        self.chat_cancel = threading.Event()
        self._set_chat_controls()
        messages = list(self.history)
        think = self.v_think.get()
        cancel = self.chat_cancel
        eng = self.engine

        def work() -> None:
            from hfserve.engine import GenerationStats

            tk_kwargs = {"reasoning_effort": "low"} if eng.template_supports("reasoning_effort") else {}
            if eng.template_supports("enable_thinking"):
                tk_kwargs["enable_thinking"] = think
            try:
                for item in eng.generate_stream(messages=messages, max_new_tokens=max_new, temperature=temp,
                                                repetition_penalty=REPETITION_PENALTY, cancel=cancel,
                                                template_kwargs=tk_kwargs, think=think):
                    if isinstance(item, GenerationStats):
                        self.events.put(("chat_done", item))
                    else:
                        self.events.put(("chat_delta", item[0], item[1]))
            except Exception as e:  # noqa: BLE001
                self.events.put(("chat_failed", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hftrain-gui-chat", daemon=True).start()

    def cancel_answer(self) -> None:
        if self.chat_cancel is not None:
            self.chat_cancel.set()

    def _on_delta(self, kind: str, delta: str) -> None:
        if kind == "thinking":
            if self.answer_kind != "thinking":
                self._say("(thinking) ", "think")
            self._say(delta, "think")
        else:
            if self.answer_kind == "thinking":
                self._say("\n", "think")
            if self.answer_kind != "content":
                delta = delta.lstrip()
            self._say(delta, "bot")
            self.answer_text += delta
        self.answer_kind = kind

    def _on_answer_done(self, stats) -> None:
        self.history.append({"role": "assistant", "content": stats.text or self.answer_text})
        rate = stats.completion_tokens / stats.gen_seconds if stats.gen_seconds > 0 else 0.0
        note = {"length": ", cut at max tokens", "cancel": ", stopped"}.get(stats.finish_reason, "")
        self._say(f"\n{stats.completion_tokens} tokens, {rate:.1f} tokens/s{note}\n", "info")
        self.chat_busy = False
        self.chat_cancel = None
        self._set_chat_controls()

    # ----------------------------------------------------------------- event loop
    def _poll(self) -> None:
        try:
            for _ in range(500):
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        now = time.time()
        if now - self.last_progress_poll >= 1.0:
            self.last_progress_poll = now
            self._poll_progress()
        self.root.after(50, self._poll)

    def _handle(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "gpu":
            self.l_gpu.configure(text=ev[1])
        elif kind == "out":
            if ev[1] == "line" and ev[2].strip() and not NOISE_RE.search(ev[2]):
                text = ev[2].rstrip()
                tag = "bad" if ("Error" in text or "Traceback" in text) else (
                    "good" if text.startswith("finished") else None)
                self.log(text, tag)
        elif kind == "exit":
            if self.prep is not None:
                self._on_prepare_exit(ev[1])
            else:
                self._on_train_exit(ev[1])
        elif kind == "log":
            self.log(ev[1], "note")
        elif kind == "engine_ready":
            self.engine = ev[1]
            self.engine_ready = True
            self.l_chat.configure(text="No model loaded")
            self._set_chat_controls()
        elif kind == "engine_error":
            self.l_chat.configure(text=f"Chat unavailable: {ev[1]}")
        elif kind == "chat_loaded":
            self.chat_device = ev[2]
            self.chat_busy = False
            self.l_chat.configure(text=f"Loaded on {ev[2]}")
            self._say(f"--- {ev[1]} loaded on {ev[2]} ---\n", "info")
            self._set_chat_controls()
            self.input.focus_set()
        elif kind == "chat_error":
            self.chat_busy = False
            self.chat_device = None
            self.l_chat.configure(text="Loading failed")
            self._set_chat_controls()
            messagebox.showerror("Chat", ev[1])
        elif kind == "chat_delta":
            self._on_delta(ev[1], ev[2])
        elif kind == "chat_done":
            self._on_answer_done(ev[1])
        elif kind == "chat_failed":
            self._say(f"\n[error: {ev[1]}]\n", "info")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            self.chat_busy = False
            self.chat_cancel = None
            self._set_chat_controls()

    def on_close(self) -> None:
        if self.train is not None and self.train.running:
            if self.close_after_exit:
                return
            if not messagebox.askyesno("Training is running",
                                       "Stop the training run and close?\n\nA checkpoint is saved first, "
                                       "so 'Resume' can continue it later. The window closes when it has stopped."):
                return
            self.close_after_exit = True
            if not self.stop_requested:
                self.stop_training()
            return
        if self.prep is not None and self.prep.running:
            self.prep.kill()  # corpus preparation is safe to interrupt; nothing partial is used
        if self.chat_cancel is not None:
            self.chat_cancel.set()
        self.root.destroy()


def prefill(app: App, *, data: str | None = None, run: str | None = None, model: str | None = None,
            lora_r: int | None = None, lr: float | None = None, epochs: float | None = None) -> None:
    """Fill the training form (``hftrain gui --data ... --run ...``). The run name goes first,
    because choosing a resumable run fills the form from that run."""
    if run:
        app.v_run.set(run)
    if data:
        app.v_corpus.set(data)
    if model:
        app.v_model.set(model)
    if lora_r:
        app.v_rank.set(str(lora_r))
    if lr:
        app.v_lr.set(f"{lr:g}")
    if epochs:
        app.v_epochs.set(f"{epochs:g}")


def main(form: dict | None = None, start: bool = False) -> int:
    if sys.stdout is None or sys.stderr is None:  # pythonw: no console
        devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = sys.stdout or devnull
        sys.stderr = sys.stderr or devnull
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on scaled displays
        except Exception:  # noqa: BLE001
            pass
    root = tk.Tk()
    app = App(root)
    handler = QueueLogHandler(app.events)
    for name in ("hfserve", "hftrain"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    if form:
        prefill(app, **form)
    if start:
        root.after(1000, app.start_training)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
