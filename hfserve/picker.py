"""Startup model chooser for the hfserve chat GUI.

``hfserve gui`` opens this window first. It lists the models installed in
``config.MODELS_DIR`` - the models folder configured in .env - preselects the one
used last, and can download another model from huggingface.co straight into that
folder, so it is immediately usable by the server, the console chat and the
PowerShell starter (``pull_model`` rewrites ``models/models.json`` itself).

"Check fit" asks the Hub how big the download really is and says whether this
machine can run it, rather than guessing from a model's name.

Everything that touches the network or the disk runs in a worker thread and
reports back through an event queue; widgets are only touched from the Tk thread,
the same pattern as gui.py and hftrain's window.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from . import config, registry
from .registry import human_size

POLL_MS = 60
GPU_QUERY = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
SEARCH_LIMIT = 25

# A starting point, not a catalogue: instruction-tuned models small enough for a
# consumer card, smallest first. No size is hard-coded here - "Check fit" asks the
# Hub for the real figures. Any other repo id (or URL) can be typed instead.
SUGGESTED: tuple[tuple[str, str], ...] = (
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B - strong for its size"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "1.7B - very small, English-first"),
    ("utter-project/EuroLLM-1.7B-Instruct", "1.7B - 35 European languages, incl. Croatian"),
    ("BSC-LT/salamandra-2b-instruct", "2B - European languages"),
    ("Qwen/Qwen2.5-3B-Instruct", "3B - needs roughly 6 GB"),
    ("microsoft/Phi-3.5-mini-instruct", "3.8B - tight on an 8 GB card"),
    ("Qwen/Qwen2.5-7B-Instruct", "7B - needs roughly 16 GB"),
    ("utter-project/EuroLLM-9B-Instruct", "9B - needs roughly 18 GB"),
)


# --------------------------------------------------------------------------- helpers (no Tk)
def _no_window_flags() -> int:
    """Keep nvidia-smi from flashing a console window on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def gpu_vram_gb() -> tuple[str, float | None]:
    """``(description, total VRAM in GB)`` from nvidia-smi; VRAM is None without a GPU."""
    try:
        out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=10,
                             creationflags=_no_window_flags())
    except (OSError, subprocess.SubprocessError):
        return "No NVIDIA GPU found (nvidia-smi missing) - models would run on the CPU.", None
    if out.returncode != 0 or not out.stdout.strip():
        return "No NVIDIA GPU found - models would run on the CPU.", None
    fields = [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")]
    name = fields[0] if fields and fields[0] else "GPU"
    try:
        total = float(fields[1]) / 1024.0
    except (IndexError, ValueError):
        return f"GPU: {name}", None
    return f"GPU: {name} with {total:.1f} GB VRAM", total


def fit_verdict(total_bytes: int, vram_gb: float | None) -> str:
    """Plain-language answer to "can this machine run it?", in float16."""
    need = total_bytes / (1024 ** 3)
    if vram_gb is None:
        return (f"{need:.1f} GB of weights. No GPU detected, so this runs on the CPU and needs "
                f"about that much free system RAM - it works, but slowly.")
    if need <= vram_gb * 0.60:
        return f"{need:.1f} GB of weights against {vram_gb:.1f} GB VRAM - fits comfortably."
    if need <= vram_gb * 0.90:
        return (f"{need:.1f} GB of weights against {vram_gb:.1f} GB VRAM - fits, but tight: keep "
                f"the context short and nothing else on the GPU.")
    return (f"{need:.1f} GB of weights against {vram_gb:.1f} GB VRAM - too big for this card in "
            f"float16. Windows would spill it into system RAM and it would crawl; pick a smaller one.")


def search_hub(query: str, limit: int = SEARCH_LIMIT) -> list[tuple[str, int]]:
    """``[(repo_id, downloads)]`` for text-generation models matching *query*."""
    from huggingface_hub import HfApi

    import inspect

    api = HfApi()
    kwargs = {"search": query, "sort": "downloads", "direction": -1, "limit": limit}
    # Ask the signature rather than catching TypeError: a TypeError raised *inside*
    # list_models would otherwise be retried unfiltered and return unservable repos.
    if "pipeline_tag" in inspect.signature(api.list_models).parameters:
        kwargs["pipeline_tag"] = "text-generation"
    return [(m.id, int(getattr(m, "downloads", 0) or 0)) for m in api.list_models(**kwargs)]


# --------------------------------------------------------------------------- the window
class ModelPicker:
    """Modal "which model?" window. ``result`` is the chosen repo id, or None."""

    def __init__(self, root: tk.Tk, preselect: str | None = None) -> None:
        self.root = root
        self.result: str | None = None
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.installed: list[registry.LocalModel] = []
        self.vram_gb: float | None = None
        self.vram_probed = False  # None alone cannot tell "no GPU" from "not asked yet"
        self.want = preselect
        self._poll_id: str | None = None

        self.win = tk.Toplevel(root)
        self.win.title("hfserve - choose a model")
        self.win.geometry("780x600")
        self.win.minsize(660, 520)
        self.win.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self._build()
        # transient() makes a window follow its master's state, and the chat root is
        # withdrawn while this dialog is up. A transient of a withdrawn master is never
        # mapped (measured on Windows: viewable=False), which would leave wait_window
        # blocking with nothing on screen - so only claim the master when it is visible.
        if root.winfo_viewable():
            self.win.transient(root)
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()
        self.win.bind("<Escape>", lambda _e: self.on_cancel())

        self.refresh_installed()
        threading.Thread(target=self._gpu_worker, name="hfserve-picker-gpu", daemon=True).start()
        self._poll_id = self.win.after(POLL_MS, self._poll)
        try:
            self.win.wait_visibility()  # grabbing an unviewable window raises TclError
            self.win.grab_set()
        except tk.TclError:
            pass  # no grab: the dialog still works, it is just not modal
        root.wait_window(self.win)

    # ----------------------------------------------------------------- UI
    def _build(self) -> None:
        outer = ttk.Frame(self.win, padding=10)
        outer.pack(fill="both", expand=True)

        inst = ttk.LabelFrame(outer, text="Installed models", padding=8)
        inst.pack(fill="both", expand=True)
        ttk.Label(inst, text=f"{config.MODELS_DIR}", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(inst, text=f"(folder from {config.MODELS_DIR_SOURCE})",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 6))

        self.tree = ttk.Treeview(inst, columns=("name", "size", "type"), show="headings", height=8)
        for col, text, width in (("name", "Model", 420), ("size", "Size", 90), ("type", "Type", 110)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")
        bar = ttk.Scrollbar(inst, orient="vertical", command=self.tree.yview)
        bar.pack(fill="y", side="left")
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.bind("<Double-1>", lambda _e: self.on_use())
        self.tree.bind("<Return>", lambda _e: self.on_use())

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(8, 4))
        self.b_use = ttk.Button(row, text="Use this model", command=self.on_use, style="Big.TButton")
        self.b_use.pack(side="left")
        ttk.Button(row, text="Refresh", command=self.refresh_installed).pack(side="left", padx=6)
        ttk.Button(row, text="Cancel", command=self.on_cancel).pack(side="right")

        dl = ttk.LabelFrame(outer, text="Download another model from huggingface.co", padding=8)
        dl.pack(fill="x", pady=(10, 0))
        self.l_gpu = ttk.Label(dl, text="Looking for a GPU ...", style="Muted.TLabel")
        self.l_gpu.pack(anchor="w")

        pick = ttk.Frame(dl)
        pick.pack(fill="x", pady=(6, 4))
        ttk.Label(pick, text="Suggestions").pack(side="left")
        self.v_suggest = tk.StringVar()
        self.cb_suggest = ttk.Combobox(pick, textvariable=self.v_suggest, state="readonly", width=56,
                                       values=[f"{rid}   -   {note}" for rid, note in SUGGESTED])
        self.cb_suggest.pack(side="left", padx=6)
        self.cb_suggest.bind("<<ComboboxSelected>>", self._on_suggest)

        entry = ttk.Frame(dl)
        entry.pack(fill="x", pady=(4, 4))
        ttk.Label(entry, text="Model id or URL").pack(side="left")
        self.v_ref = tk.StringVar()
        e = ttk.Entry(entry, textvariable=self.v_ref, width=42)
        e.pack(side="left", padx=6, fill="x", expand=True)
        e.bind("<Return>", lambda _e: self.on_check())
        self.b_search = ttk.Button(entry, text="Search Hub", command=self.on_search)
        self.b_search.pack(side="left")
        self.b_check = ttk.Button(entry, text="Check fit", command=self.on_check)
        self.b_check.pack(side="left", padx=6)
        self.b_download = ttk.Button(entry, text="Download", command=self.on_download)
        self.b_download.pack(side="left")

        self.lb_found = tk.Listbox(dl, height=5)
        self.lb_found.pack(fill="x", pady=(4, 4))
        self.lb_found.bind("<<ListboxSelect>>", self._on_found)
        self.lb_found.bind("<Double-1>", lambda _e: self.on_check())

        self.l_dl = ttk.Label(dl, text="Pick a suggestion, or type any huggingface.co model id.",
                              wraplength=720, justify="left")
        self.l_dl.pack(anchor="w", pady=(2, 4))
        self.pb = ttk.Progressbar(dl, mode="indeterminate")

    # ----------------------------------------------------------------- installed list
    def refresh_installed(self) -> None:
        try:
            # Broad on purpose: a half-written config.json in one model folder must not
            # take down the whole chooser (_inspect_dir can raise ValueError/JSON errors).
            self.installed = [m for m in registry.list_models() if not m.incomplete]
        except Exception as e:  # noqa: BLE001 - shown in the window
            self.installed = []
            self.l_dl.configure(text=f"Could not read {config.MODELS_DIR}: {e}")
        self.tree.delete(*self.tree.get_children())
        for m in self.installed:
            self.tree.insert("", "end", iid=m.repo_id,
                             values=(m.repo_id, human_size(m.size_bytes), m.model_type or "-"))
        # Match case-insensitively: on Windows the folder casing can differ from the id
        # the user typed when downloading, and the fresh model should still be selected.
        pick = None
        if self.want:
            by_lower = {m.repo_id.lower(): m.repo_id for m in self.installed}
            pick = by_lower.get(self.want.lower())
        if pick is None and self.installed:
            pick = self.installed[0].repo_id
        if pick:
            self.tree.selection_set(pick)
            self.tree.focus(pick)
            self.tree.see(pick)
        self._set_controls()

    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _set_controls(self) -> None:
        self.b_use.configure(state="disabled" if self.busy or not self.installed else "normal")
        for b in (self.b_search, self.b_check, self.b_download):
            b.configure(state="disabled" if self.busy else "normal")

    def _start_busy(self, message: str) -> None:
        self.busy = True
        self.l_dl.configure(text=message)
        if not self.pb.winfo_ismapped():
            self.pb.pack(fill="x", pady=(2, 0))
        self.pb.start(12)
        self._set_controls()

    def _stop_busy(self, message: str) -> None:
        self.busy = False
        self.pb.stop()
        if self.pb.winfo_ismapped():
            self.pb.pack_forget()
        self.l_dl.configure(text=message)
        self._set_controls()

    # ----------------------------------------------------------------- actions
    def _on_suggest(self, _event=None) -> None:
        text = self.v_suggest.get()
        if text:
            self.v_ref.set(text.split("   -   ")[0].strip())

    def _on_found(self, _event=None) -> None:
        sel = self.lb_found.curselection()
        if sel:
            self.v_ref.set(self.lb_found.get(sel[0]).split("   ")[0].strip())

    def _close(self) -> None:
        """Cancel the pending poll before the widgets go away, then close for good."""
        if self._poll_id is not None:
            try:
                self.win.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        for step in (self.win.grab_release, self.win.destroy):
            try:
                step()
            except tk.TclError:
                pass

    def on_use(self) -> None:
        if self.busy:
            return
        name = self._selected()
        if not name:
            messagebox.showinfo("hfserve", "No model selected. Download one first, or pick a row.",
                                parent=self.win)
            return
        self.result = name
        self._close()

    def on_cancel(self) -> None:
        if self.busy and not messagebox.askyesno(
                "hfserve",
                "A download is still running. Close anyway?\n\nIt stops when the program exits, "
                "and resumes where it left off next time you download the same model.",
                parent=self.win):
            return
        self.result = None
        self._close()

    def on_search(self) -> None:
        query = self.v_ref.get().strip()
        if not query or self.busy:
            return
        self._start_busy(f"Searching huggingface.co for {query!r} ...")

        def work() -> None:
            try:
                self.events.put(("found", search_hub(query)))
            except Exception as e:  # noqa: BLE001 - shown in the window
                self.events.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-picker-search", daemon=True).start()

    def on_check(self) -> None:
        ref = self.v_ref.get().strip()
        if not ref or self.busy:
            return
        self._start_busy(f"Asking the Hub about {ref} ...")

        def work() -> None:
            try:
                self.events.put(("checked", registry.hub_download_plan(ref)))
            except Exception as e:  # noqa: BLE001
                self.events.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-picker-check", daemon=True).start()

    def on_download(self) -> None:
        ref = self.v_ref.get().strip()
        if not ref or self.busy:
            return
        self._start_busy(f"Downloading {ref} into {config.MODELS_DIR} - this can take "
                         f"several minutes; the window stays open.")

        def work() -> None:
            try:
                model = registry.pull_model(ref, quiet=True)
                self.events.put(("pulled", model.repo_id, model.size_bytes))
            except Exception as e:  # noqa: BLE001
                self.events.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, name="hfserve-picker-pull", daemon=True).start()

    def _gpu_worker(self) -> None:
        description, vram = gpu_vram_gb()
        self.events.put(("gpu", description, vram))

    # ----------------------------------------------------------------- events
    def _poll(self) -> None:
        self._poll_id = None
        if not self.win.winfo_exists():
            return  # closed while this tick was pending: the queue is nobody's business now
        try:
            for _ in range(100):
                self._handle(self.events.get_nowait())
        except queue.Empty:
            pass
        except Exception as e:  # noqa: BLE001
            # One bad event must not kill the loop: without this the dialog would sit
            # there with every button disabled, because busy is never cleared again.
            try:
                self._stop_busy(f"{type(e).__name__}: {e}")
            except tk.TclError:
                pass
        finally:
            if self.win.winfo_exists():
                self._poll_id = self.win.after(POLL_MS, self._poll)

    def _handle(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == "gpu":
            self.l_gpu.configure(text=ev[1])
            self.vram_gb = ev[2]
            self.vram_probed = True
        elif kind == "found":
            self.lb_found.delete(0, "end")
            for repo_id, downloads in ev[1]:
                self.lb_found.insert("end", f"{repo_id}   ({downloads:,} downloads)")
            self._stop_busy(f"{len(ev[1])} models found. Pick one, then Check fit."
                            if ev[1] else "Nothing found for that search.")
        elif kind == "checked":
            plan = ev[1]
            if not plan["servable"]:
                self._stop_busy(f"{plan['repo_id']} has no config.json, so transformers cannot "
                                f"serve it. It is probably a GGUF or other-runtime repo.")
                return
            total = plan["total_bytes"]
            if total <= 0:
                verdict = "The Hub reported no file sizes, so the download size is unknown."
            elif not self.vram_probed:
                verdict = "Still reading the GPU - press Check fit again in a moment."
            else:
                verdict = fit_verdict(total, self.vram_gb)
            self._stop_busy(f"{plan['repo_id']}: {plan['files']} files, {human_size(total)} "
                            f"to download. {verdict}")
        elif kind == "pulled":
            repo_id, size = ev[1], ev[2]
            self.want = repo_id
            self._stop_busy(f"Downloaded {repo_id} ({human_size(size)}). It is installed and "
                            f"selected; press 'Use this model' to chat with it.")
            self.refresh_installed()
        elif kind == "error":
            self._stop_busy(ev[1])
            messagebox.showerror("hfserve", ev[1], parent=self.win)


def choose_model(root: tk.Tk, preselect: str | None = None) -> str | None:
    """Show the picker and return the chosen model id, or None when cancelled."""
    return ModelPicker(root, preselect=preselect).result
