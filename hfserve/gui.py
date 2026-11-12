"""Tkinter chat window talking to the hfserve server over HTTP.

``python -m hfserve gui [--model NAME]`` (or HFServe-Chat.cmd) first asks which of
the installed models to use - the pick list comes from ``config.MODELS_DIR``, i.e.
the models folder configured in .env - then opens the chat window, starts the API
server when none is running (and stops it again on close when it started one),
loads the chosen model and chats over the same endpoints as the console client:
``GET /api/tags``, ``POST /api/load`` and streaming ``POST /api/chat``.
``--model NAME`` skips the question. Picking another model in the window's combo
box swaps it in place, and the choice is remembered for the next start.
All HTTP runs in worker threads; widgets are only touched from the Tk thread,
fed through an event queue like hftrain's GUI.
"""
from __future__ import annotations

import json
import os
import shutil
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import httpx

from . import config, process, registry
from .chat import _stats_line, error_text
from .registry import human_size

POLL_MS = 50
LOAD_TIMEOUT = 900  # seconds; first load of a big model on a cold disk is slow


class App:
    def __init__(self, root: tk.Tk, host: str, port: int, model: str | None = None) -> None:
        self.root = root
        self.host = host
        self.port = port
        self.base = config.server_url(process.connect_host(host), port)
        self.want_model = model
        self.model: str | None = None  # loaded model, None until /api/load succeeded
        self.spawned = None  # Popen when this window started the server (stopped again on close)
        self.history: list[dict] = []
        self.busy = False  # a load or an answer is in flight
        self.cancel: threading.Event | None = None
        self.answer_kind: str | None = None
        self.answer_text = ""
        # Attached file (a copy under files/in, never the user's own), what hffiles found
        # in it, and whether the answer in flight is a plan to execute.
        self.file_path = None
        self.file_desc: dict | None = None
        self.plan_pending = False
        self.events: queue.Queue = queue.Queue()
        # trust_env=False: never send localhost traffic through an HTTP(S)_PROXY.
        self.client = httpx.Client(base_url=self.base, trust_env=False)

        root.title("hfserve chat")
        root.geometry("860x640")
        root.minsize(560, 420)
        self._style()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        threading.Thread(target=self._connect, name="hfserve-gui-connect", daemon=True).start()
        self.zakoni_refreshing = False
        threading.Thread(target=self._zakoni_refresh_worker, name="hfserve-gui-zakoni",
                         daemon=True).start()
        root.after(POLL_MS, self._poll)

    # ----------------------------------------------------------------- UI
    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Muted.TLabel", foreground="#777777")
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text="Model").pack(side="left")
        self.v_model = tk.StringVar()
        self.cb_model = ttk.Combobox(top, textvariable=self.v_model, state="readonly", width=46)
        self.cb_model.pack(side="left", padx=6)
        self.cb_model.bind("<MouseWheel>", lambda e: "break")
        # Picking another model swaps it in place: /api/load unloads the current one
        # (the engine holds one at a time), so the chat continues with the new model.
        self.cb_model.bind("<<ComboboxSelected>>", self._on_model_pick)
        self.b_load = ttk.Button(top, text="Load", command=self.load_model, state="disabled")
        self.b_load.pack(side="left")
        self.l_status = ttk.Label(top, text="Connecting ...", style="Muted.TLabel")
        self.l_status.pack(side="left", padx=8)

        opts = ttk.Frame(outer)
        opts.pack(fill="x", pady=(8, 6))
        ttk.Label(opts, text="System").pack(side="left")
        self.v_system = tk.StringVar()
        ttk.Entry(opts, textvariable=self.v_system, width=40).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Label(opts, text="Temperature").pack(side="left", padx=(12, 0))
        self.v_temp = tk.StringVar(value="0.6")
        ttk.Spinbox(opts, from_=0.0, to=1.5, increment=0.1, textvariable=self.v_temp, width=5).pack(side="left", padx=6)
        ttk.Label(opts, text="Max tokens").pack(side="left", padx=(6, 0))
        self.v_maxtok = tk.StringVar(value="512")
        ttk.Spinbox(opts, from_=32, to=4096, increment=64, textvariable=self.v_maxtok, width=6).pack(side="left", padx=6)
        ttk.Label(opts, text="Repeat penalty").pack(side="left", padx=(6, 0))
        self.v_repeat = tk.StringVar(value="1.1")  # small models loop without it
        ttk.Spinbox(opts, from_=1.0, to=1.5, increment=0.05, textvariable=self.v_repeat, width=5).pack(side="left", padx=6)
        self.v_think = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Think", variable=self.v_think).pack(side="left", padx=(6, 0))
        self.v_rag = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Kviz baza (RAG)", variable=self.v_rag).pack(side="left", padx=(6, 0))
        self.v_rag_zakoni = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Zakoni (RAG)", variable=self.v_rag_zakoni).pack(side="left", padx=(6, 0))
        ttk.Button(opts, text="New chat", command=self.new_chat).pack(side="right")

        # File row. With a file attached, every message becomes a plan that hffiles executes
        # on a copy; the model only decides which operations to run. Excel, Word,
        # PowerPoint and PDF all go through the same row.
        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Datoteka").pack(side="left")
        self.b_attach = ttk.Button(row, text="Priloži ...", command=self.attach_file)
        self.b_attach.pack(side="left", padx=6)
        self.l_file = ttk.Label(row, text="nema priložene datoteke", style="Muted.TLabel")
        self.l_file.pack(side="left", padx=6)
        self.b_detach = ttk.Button(row, text="Odvoji", command=self.detach_file,
                                   state="disabled")
        self.b_detach.pack(side="right")
        ttk.Button(row, text="Otvori mapu", command=self.open_out_dir).pack(side="right", padx=6)
        self.b_pdf = ttk.Button(row, text="U PDF", command=self.save_pdf, state="disabled")
        self.b_pdf.pack(side="right", padx=6)
        self.b_save = ttk.Button(row, text="Spremi kao ...", command=self.save_copy,
                                 state="disabled")
        self.b_save.pack(side="right")

        # Legislation-index refresh status: packed (before the transcript) only while
        # hfzakon refreshes in the background, so the user sees what is happening.
        self.refresh_row = ttk.Frame(outer)
        self.l_refresh = ttk.Label(self.refresh_row, text="", style="Muted.TLabel")
        self.l_refresh.pack(side="left")
        self.p_refresh = ttk.Progressbar(self.refresh_row, length=220, mode="determinate")

        self.transcript = ScrolledText(outer, wrap="word", font=("Segoe UI", 11), relief="flat", borderwidth=1,
                                       padx=12, pady=10, bg="#ffffff")
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("you", foreground="#0b5cad", font=("Segoe UI", 11, "bold"), spacing1=10)
        self.transcript.tag_configure("bot", foreground="#1b1b1b", spacing1=2)
        self.transcript.tag_configure("botname", foreground="#2e7d32", font=("Segoe UI", 11, "bold"), spacing1=10)
        self.transcript.tag_configure("think", foreground="#8a8a8a", font=("Segoe UI", 10, "italic"))
        self.transcript.tag_configure("info", foreground="#9a9a9a", font=("Segoe UI", 9))
        self.transcript.configure(state="disabled")

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(8, 0))
        self.input = tk.Text(bottom, height=3, wrap="word", font=("Segoe UI", 11), relief="solid", borderwidth=1,
                             padx=8, pady=6)
        self.input.pack(side="left", fill="x", expand=True)
        self.input.bind("<Return>", self._on_enter)
        btns = ttk.Frame(bottom)
        btns.pack(side="left", padx=(8, 0))
        self.b_send = ttk.Button(btns, text="Send", style="Big.TButton", command=self.send, state="disabled")
        self.b_send.pack(fill="x")
        self.b_stop = ttk.Button(btns, text="Stop", command=self.cancel_answer, state="disabled")
        self.b_stop.pack(fill="x", pady=(4, 0))
        ttk.Label(outer, text="Enter sends, Shift+Enter starts a new line.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

    def _say(self, text: str, tag: str) -> None:
        tr = self.transcript
        tr.configure(state="normal")
        tr.insert("end", text, (tag,))
        tr.configure(state="disabled")
        tr.see("end")

    def _set_controls(self) -> None:
        loaded = self.model is not None
        self.b_load.configure(state="normal" if self.cb_model["values"] and not self.busy else "disabled")
        self.b_send.configure(state="normal" if loaded and not self.busy else "disabled")
        self.b_stop.configure(state="normal" if self.busy and self.cancel is not None else "disabled")
        self.b_attach.configure(state="disabled" if self.busy else "normal")
        attached = self.file_path is not None and not self.busy
        for button in (self.b_detach, self.b_save, self.b_pdf):
            button.configure(state="normal" if attached else "disabled")

    # ------------------------------------------------------------------ files
    def attach_file(self) -> None:
        """Pick a file, copy it into files/in and read what is in it (worker thread)."""
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Odaberi datoteku",
            filetypes=[("Svi podržani formati", "*.xlsx *.xlsm *.docx *.pptx *.pdf"),
                       ("Excel", "*.xlsx *.xlsm *.xltx *.xltm"), ("Word", "*.docx"),
                       ("PowerPoint", "*.pptx"), ("PDF", "*.pdf"),
                       ("Sve datoteke", "*.*")])
        if not path:
            return
        self.l_file.configure(text="čitam ...")

        def work() -> None:
            try:
                from hffiles import files as _files

                copy = _files.upload(path)  # the user's own file is never written to
                desc = _files.describe(copy)
                self.events.put(("file_attached", copy, desc, _files.digest_text(desc),
                                 _files.summary(desc)))
            except Exception as e:  # noqa: BLE001 - a bad file must not kill the chat
                self.events.put(("file_error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-gui-attach", daemon=True).start()

    def detach_file(self) -> None:
        self.file_path = None
        self.file_desc = None
        self.l_file.configure(text="nema priložene datoteke")
        self._set_controls()

    def open_out_dir(self) -> None:
        try:
            from hffiles import config as fconfig

            fconfig.ensure_dirs()
            os.startfile(fconfig.OUT_DIR)  # noqa: S606 - Windows file manager
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("hfserve chat", f"{type(e).__name__}: {e}")

    def save_copy(self) -> None:
        """The download button: copy the file being worked on wherever the user wants it."""
        if self.file_path is None or self.busy:
            return
        suffix = self.file_path.suffix
        target = filedialog.asksaveasfilename(
            title="Spremi kopiju", initialfile=self.file_path.name, defaultextension=suffix,
            filetypes=[(suffix.lstrip(".").upper(), f"*{suffix}"), ("Sve datoteke", "*.*")])
        if not target:
            return
        try:
            shutil.copy2(self.file_path, target)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("hfserve chat", f"{type(e).__name__}: {e}")
            return
        self._say(f"[spremljeno: {target}]\n", "info")

    def save_pdf(self) -> None:
        """Render the attached file to PDF (Office or LibreOffice does the rendering)."""
        if self.file_path is None or self.busy:
            return
        source = self.file_path
        target = filedialog.asksaveasfilename(
            title="Spremi kao PDF", initialfile=f"{source.stem}.pdf", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")])
        if not target:
            return
        self.busy = True
        self._set_controls()
        self._say(f"[izvoz u PDF: {source.name} ...]\n", "info")

        def work() -> None:
            try:
                from hffiles import files as _files

                self.events.put(("pdf_ready", _files.to_pdf(source, target)))
            except Exception as e:  # noqa: BLE001
                self.events.put(("file_error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-gui-pdf", daemon=True).start()

    # ----------------------------------------------------------------- server
    def _connect(self) -> None:
        """Worker: make sure a server runs, list the models, auto-load one. Events only."""
        try:
            health = process.server_health(self.host, self.port)
            if health is None:
                if process.port_in_use(self.host, self.port):
                    self.events.put(("fatal", f"Port {self.port} on {self.host} is in use by another program "
                                              f"(not an hfserve server)."))
                    return
                self.events.put(("status", "Starting server ..."))
                self.spawned = process.spawn_server(self.host, self.port)
                if not process.wait_for_server(self.host, self.port, proc=self.spawned):
                    tail = process.log_tail(10)
                    self.events.put(("fatal", "The server failed to start.\n\n"
                                              f"server log: {config.LOG_FILE}\n{tail}"))
                    return
                health = process.server_health(self.host, self.port) or {}
            r = self.client.get("/api/tags", timeout=10)
            names = [m.get("name", "") for m in (r.json() or {}).get("models") or []] if r.status_code == 200 else []
            loaded = str(health.get("model") or "") or None
            self.events.put(("connected", names, loaded))
            # Load without a click when the choice is unambiguous: the model asked for
            # (the startup picker always supplies one), the one the server already has,
            # or the only one installed. A server holding a different model is swapped.
            want = self.want_model or loaded or (names[0] if len(names) == 1 else None)
            if want and want != loaded:
                self.events.put(("autoload", want))
        except Exception as e:  # noqa: BLE001 - shown in the window
            self.events.put(("fatal", f"{type(e).__name__}: {e}"))

    def _on_model_pick(self, _event=None) -> None:
        """Combo box changed: load the new model unless it is the one already loaded."""
        name = self.v_model.get()
        if not name or name == self.model:
            return
        if self.busy:
            # A load or an answer is in flight. Put the box back on the loaded model so it
            # never shows something other than what the server actually has.
            if self.model:
                self.v_model.set(self.model)
            self.l_status.configure(text="Busy - wait for the answer, then pick the model again")
            return
        self.load_model()

    def load_model(self) -> None:
        name = self.v_model.get()
        if not name or self.busy:
            return
        self.busy = True
        self._set_controls()
        self.l_status.configure(text=f"Loading {name} ...")

        def work() -> None:
            try:
                r = self.client.post("/api/load", json={"model": name}, timeout=LOAD_TIMEOUT)
                if r.status_code != 200:
                    self.events.put(("load_error", error_text(r)))
                    return
                self.events.put(("loaded", r.json() or {}))
            except Exception as e:  # noqa: BLE001
                self.events.put(("load_error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-gui-load", daemon=True).start()

    # ----------------------------------------------------------------- chat
    def new_chat(self) -> None:
        if self.busy and self.cancel is not None:
            self.cancel.set()
        self.history = []
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _on_enter(self, event) -> str | None:
        if event.state & 0x0001:  # Shift: new line
            return None
        self.send()
        return "break"

    def send(self) -> None:
        text = self.input.get("1.0", "end").strip()
        if not text or self.busy or self.model is None:
            return
        try:
            temp = float(self.v_temp.get())
        except ValueError:
            temp = config.DEFAULT_TEMPERATURE
        try:
            max_new = int(self.v_maxtok.get())
        except ValueError:
            max_new = 512
        try:
            repeat = float(self.v_repeat.get())
        except ValueError:
            repeat = 1.1
        self.input.delete("1.0", "end")
        self.history.append({"role": "user", "content": text})
        self._say("You\n", "you")
        self._say(text + "\n", "bot")
        self._say(self.model.rsplit("/", 1)[-1] + "\n", "botname")
        self.answer_kind, self.answer_text = None, ""
        self.busy = True
        self.cancel = threading.Event()
        self._set_controls()

        system = self.v_system.get().strip()
        messages = ([{"role": "system", "content": system}] if system else []) + list(self.history)
        if self.file_path is not None:
            # File mode wins over the RAG toggles: the model is asked for a plan, and
            # hffiles - not the model - edits the file once the answer is in.
            try:
                from hffiles import plan as _plan

                messages = _plan.build_messages(text, self.file_desc)
                temp = min(temp, 0.2)        # JSON, not prose
                max_new = max(max_new, 700)  # a plan must not be cut off half way
                self.plan_pending = True
                self._say(f"[plan za {self.file_path.name}]\n", "info")
                if self.v_rag.get() or self.v_rag_zakoni.get():
                    self._say("[RAG se ne koristi dok je priložena datoteka]\n", "info")
            except Exception as e:  # noqa: BLE001 - fall back to an ordinary answer
                self._say(f"[hffiles nedostupan ({type(e).__name__}: {e}) - obično pitanje]\n",
                          "info")
        elif self.v_rag_zakoni.get():
            # Strict legislation answers (hfzakon): context-only, cites zakon/NN, else "Ne znam.".
            # Takes precedence over the quiz base when both toggles are on.
            if self.zakoni_refreshing:
                self._say("[baza zakona se upravo osvježava — najnovija izdanja NN "
                          "možda još nisu uključena]\n", "info")
            try:
                from hfzakon import retrieve as _zakoni

                messages, used = _zakoni.build_messages(text)
                self._say(f"[RAG zakoni: {len(used)} iz baze]\n", "info")
            except Exception as e:  # noqa: BLE001 - missing package/index must not kill the chat
                self._say(f"[RAG nedostupan: {type(e).__name__}: {e}]\n", "info")
        elif self.v_rag.get():
            # Grounded single-turn answer from the local index (hfrag); history is shown
            # in the transcript but the model sees only the retrieved context + question.
            try:
                from hfrag import retrieve as _rag

                messages, used = _rag.build_messages(text)
                self._say(f"[RAG: {len(used)} iz baze]\n", "info")
            except Exception as e:  # noqa: BLE001 - missing package/index must not kill the chat
                self._say(f"[RAG nedostupan: {type(e).__name__}: {e}]\n", "info")
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.v_think.get(),
            "options": {"temperature": temp, "num_predict": max_new, "repeat_penalty": repeat},
        }
        cancel = self.cancel

        def work() -> None:
            try:
                with self.client.stream("POST", "/api/chat", json=body,
                                        timeout=httpx.Timeout(None, connect=10)) as r:
                    if r.status_code != 200:
                        r.read()
                        self.events.put(("chat_failed", error_text(r)))
                        return
                    for line in r.iter_lines():
                        # Leaving the `with` block closes the response: the server cancels generation.
                        if cancel.is_set():
                            self.events.put(("chat_done", {"done_reason": "cancel"}))
                            return
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if obj.get("error") and "message" not in obj:
                            self.events.put(("chat_failed", str(obj["error"])))
                            return
                        msg = obj.get("message") or {}
                        if msg.get("thinking"):
                            self.events.put(("chat_delta", "thinking", msg["thinking"]))
                        if msg.get("content"):
                            self.events.put(("chat_delta", "content", msg["content"]))
                        if obj.get("done"):
                            self.events.put(("chat_done", obj))
                            return
                self.events.put(("chat_done", {}))
            except Exception as e:  # noqa: BLE001
                self.events.put(("chat_failed", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-gui-chat", daemon=True).start()

    def cancel_answer(self) -> None:
        if self.cancel is not None:
            self.cancel.set()

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

    def _on_done(self, done: dict) -> None:
        if self.plan_pending:
            # A plan is not conversation, so it stays out of the history.
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
        elif self.answer_text:
            self.history.append({"role": "assistant", "content": self.answer_text})
        elif self.history and self.history[-1]["role"] == "user":
            self.history.pop()  # no answer came back: forget the question too
        self._say("\n" + _stats_line(done) + "\n", "info")
        self.cancel = None
        if self.plan_pending:
            self.plan_pending = False
            if self._run_plan(self.answer_text):
                self._set_controls()  # stays busy until the plan has run
                return
        self.busy = False
        self._set_controls()
        self.input.focus_set()

    # ------------------------------------------------------- plan execution
    def _run_plan(self, answer: str) -> bool:
        """Run the plan the model just wrote. True while the work is in flight."""
        try:
            from hffiles import ops as _ops, plan as _plan
        except Exception as e:  # noqa: BLE001
            self._say(f"[hffiles nedostupan: {type(e).__name__}: {e}]\n", "info")
            return False
        raw = _plan.extract_plan(answer)
        if raw is None:
            self._say("[model nije vratio plan u JSON-u - ponovi zahtjev jasnije "
                      "(npr. \"sortiraj po stupcu Iznos\")]\n", "info")
            return False
        self._say("[plan: " + json.dumps(raw, ensure_ascii=False)[:400] + "]\n", "info")
        source = self.file_path

        def work() -> None:
            try:
                result = _ops.apply_plan(raw, source=source)
                desc = None
                if result.saved and result.target is not None:
                    from hffiles import files as _files

                    desc = _files.describe(result.target)
                self.events.put(("plan_result", result, desc))
            except Exception as e:  # noqa: BLE001
                self.events.put(("file_error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-gui-plan", daemon=True).start()
        return True

    def _on_plan_result(self, result, desc: dict | None) -> None:
        for line in result.lines():
            self._say(f"[{line}]\n", "info")
        if result.saved and result.target is not None:
            # Keep working on what was just produced, so a follow-up ("sad dodaj zbroj")
            # builds on it instead of starting from the upload again.
            self.file_path = result.target
            if desc is not None:
                self.file_desc = desc
            self.l_file.configure(text=f"{result.target.name} (rezultat)")
        self.busy = False
        self.cancel = None
        self._set_controls()
        self.input.focus_set()

    # ----------------------------------------------------------------- event loop
    def _zakoni_refresh_worker(self) -> None:
        """Startup check of the legislation index; pulls and embeds only what is new."""
        try:
            from hfzakon import refresh as _refresh

            _refresh.maybe_run(progress=lambda ev: self.events.put(("zakoni", ev)))
        except Exception as e:  # noqa: BLE001 - a failed refresh must never kill the chat
            self.events.put(("zakoni", {"stage": "error", "done": 0, "total": 0,
                                        "detail": f"{type(e).__name__}: {e}", "eta_s": None}))

    def _on_zakoni(self, ev: dict) -> None:
        stage, detail = ev["stage"], ev["detail"]
        if stage in ("check", "laws", "pull", "index"):
            self.zakoni_refreshing = True
            if not self.refresh_row.winfo_ismapped():
                self.refresh_row.pack(fill="x", pady=(0, 4), before=self.transcript)
                self.p_refresh.pack(side="left", padx=8)
            names = {"check": "Osvježavam bazu zakona", "laws": "Novi zakoni",
                     "pull": "Preuzimam", "index": "Indeksiram"}
            eta = ""
            if ev.get("eta_s"):
                eta = (f" · još ~{max(1, round(ev['eta_s'] / 60))} min" if ev["eta_s"] > 45
                       else f" · još ~{int(ev['eta_s'])} s")
            self.l_refresh.configure(text=f"{names[stage]}: {detail}{eta}")
            if ev["total"]:
                self.p_refresh.configure(maximum=ev["total"], value=ev["done"])
        else:  # done | current | error
            self.zakoni_refreshing = False
            if stage == "done":
                self._say(f"[baza zakona osvježena: {detail}]\n", "info")
            elif stage == "error":
                self._say(f"[osvježavanje baze zakona nije uspjelo: {detail} — "
                          f"koristi se postojeća baza]\n", "info")
            if self.refresh_row.winfo_ismapped():
                self.refresh_row.pack_forget()

    def _poll(self) -> None:
        try:
            for _ in range(500):
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(POLL_MS, self._poll)

    def _handle(self, ev: tuple) -> None:
        if ev[0] == "zakoni":
            self._on_zakoni(ev[1])
            return
        kind = ev[0]
        if kind == "status":
            self.l_status.configure(text=ev[1])
        elif kind == "connected":
            names, loaded = ev[1], ev[2]
            self.cb_model["values"] = names
            if loaded:
                self.model = loaded
            pick = (self.want_model or loaded or config.read_last_model()
                    or (names[0] if names else ""))
            if pick in names:
                self.v_model.set(pick)
            via = "started" if self.spawned is not None else "running"
            if not names:
                self.l_status.configure(text=f"No models installed (server {via} at {self.base})")
            elif loaded:
                self.l_status.configure(text=f"{loaded} (server {via} at {self.base})")
            else:
                self.l_status.configure(text=f"Server {via} at {self.base}")
            self._set_controls()
        elif kind == "autoload":
            if ev[1] in self.cb_model["values"]:
                self.v_model.set(ev[1])
                self.load_model()
        elif kind == "loaded":
            info = ev[1]
            self.model = info.get("model") or self.v_model.get()
            vram = info.get("vram_bytes") or 0
            self.l_status.configure(
                text=f"{self.model}: {info.get('device')} / {info.get('dtype')} "
                     f"in {float(info.get('load_seconds') or 0):.1f}s"
                     + (f" (VRAM {human_size(vram)})" if vram else ""))
            self._say(f"--- {self.model} loaded on {info.get('device')} ---\n", "info")
            config.write_last_model(self.model)  # preselected the next time the picker opens
            self.history = []
            self.busy = False
            self._set_controls()
            self.input.focus_set()
        elif kind == "load_error":
            self.busy = False
            self.l_status.configure(text="Loading failed")
            self._set_controls()
            messagebox.showerror("hfserve chat", ev[1])
        elif kind == "file_attached":
            path, desc, digest, summary = ev[1], ev[2], ev[3], ev[4]
            self.file_path, self.file_desc = path, desc
            self.l_file.configure(text=summary)
            self._say(f"--- priložen {path.name} ---\n{digest}\n", "info")
            self._say("[pitaj što treba napraviti; Python radi izmjene na kopiji u files/out]\n",
                      "info")
            self._set_controls()
        elif kind == "file_error":
            self.l_file.configure(text="nema priložene datoteke" if self.file_path is None
                                  else self.file_path.name)
            self._say(f"[datoteka: {ev[1]}]\n", "info")
            self.busy = False
            self.cancel = None
            self._set_controls()
        elif kind == "pdf_ready":
            self._say(f"[PDF: {ev[1]}]\n", "info")
            self.busy = False
            self._set_controls()
        elif kind == "plan_result":
            self._on_plan_result(ev[1], ev[2])
        elif kind == "chat_delta":
            self._on_delta(ev[1], ev[2])
        elif kind == "chat_done":
            self._on_done(ev[1])
        elif kind == "chat_failed":
            self._say(f"\n[error: {ev[1]}]\n", "info")
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            self.busy = False
            self.cancel = None
            self._set_controls()
        elif kind == "fatal":
            self.l_status.configure(text="Error")
            messagebox.showerror("hfserve chat", ev[1])

    def on_close(self) -> None:
        if self.cancel is not None:
            self.cancel.set()
        try:
            self.client.close()
        except Exception:
            pass
        if self.spawned is not None:
            # This window started the server, so it takes it down again; a server
            # that was already running stays up for its other clients.
            threading.Thread(target=process.stop_server,
                             args=(self.host, self.port),
                             kwargs={"proc": self.spawned, "log": lambda *a, **k: None},
                             daemon=True).start()
            self.root.after(300, self.root.destroy)  # give the HTTP shutdown a head start
        else:
            self.root.destroy()


def main(host: str = config.DEFAULT_HOST, port: int = config.DEFAULT_PORT, model: str | None = None) -> int:
    root = tk.Tk()
    root.withdraw()  # stays hidden while the model question is on screen
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass
    style.configure("Muted.TLabel", foreground="#777777")
    style.configure("Big.TButton", font=("Segoe UI", 10, "bold"))

    if model is None:
        # Fail loud about the models folder before showing an empty list: a path that
        # does not exist is a settings mistake, not "no models downloaded yet".
        problem = None
        try:
            config.require_models_dir()
        except config.ConfigError as e:
            problem = str(e)
        if config.SETTINGS_PROBLEMS:  # a port that is not a number, an unreadable URL
            messagebox.showwarning("hfserve chat", "\n".join(config.SETTINGS_PROBLEMS))
        if problem:
            messagebox.showerror(
                "hfserve chat",
                f"{problem}\n\nYou can still download a model in the next window - the "
                f"folder is created when the download starts.")
        else:
            try:
                installed = [m for m in registry.list_models() if not m.incomplete]
            except Exception as e:  # noqa: BLE001 - an unreadable drive must still say why
                installed = []
                messagebox.showerror("hfserve chat",
                                     f"Could not read\n  {config.MODELS_DIR}\n\n{e}")
            if not installed:
                messagebox.showwarning(
                    "hfserve chat",
                    f"No models are installed in\n  {config.MODELS_DIR}\n"
                    f"(that folder comes from {config.MODELS_DIR_SOURCE}).\n\n"
                    f"Download one in the next window, or run:  hfserve pull <org/name>")

        from . import picker  # tkinter-only, imported like gui itself: only when needed

        model = picker.choose_model(root, preselect=config.read_last_model())
        if model is None:  # cancelled: no model, so no chat
            root.destroy()
            return 1

    App(root, host, port, model)
    root.deiconify()
    root.mainloop()
    return 0
