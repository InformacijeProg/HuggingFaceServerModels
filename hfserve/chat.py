"""Interactive console chat client (Ollama-style ``>>>`` prompt) talking to the hfserve server.

Only the HTTP API documented in docs/API.md is used:
``POST /api/chat`` (streaming NDJSON), ``GET /api/tags`` and ``POST /api/load``.

Reasoning display: the server streams reasoning in ``message.thinking`` and the
answer in ``message.content``. Thinking deltas are printed dimmed (or replaced by
one ``[thinking…]`` marker when ``\\set think off``); content deltas are printed
normally. As a fallback for servers that do not split, ``<think>...</think>``
tags inside ``content`` are detected too. Only the answer is kept in the history.
"""
from __future__ import annotations

import json
import queue
import sys
import threading

import httpx

from . import config, process
from .registry import human_size
from .ui import bold, cyan, dim, red, setup_console

# Reasoning blocks a model may wrap in tags (fallback when the server does not split them).
THINK_TAGS = ("think", "thinking", "reasoning")
_OPEN_TAGS = tuple(f"<{t}>" for t in THINK_TAGS)
_CLOSE_TAGS = tuple(f"</{t}>" for t in THINK_TAGS)
_ALL_TAGS = _OPEN_TAGS + _CLOSE_TAGS
_MAX_TAG_LEN = max(len(t) for t in _ALL_TAGS)

REASONING_EFFORTS = ("high", "medium", "low")
# "high" thinks for hundreds of tokens even on trivial prompts; medium/low stay short.
DEFAULT_REASONING_EFFORT = "medium"

PROMPT = ">>> "
CONT_PROMPT = "... "
MULTILINE_MARK = '"""'
BYE_CMD = "\\bye"
HELP_CMD = "\\help"


def _parse_effort(value: str) -> str:
    v = value.strip().lower()
    if v not in REASONING_EFFORTS:
        raise ValueError(value)
    return v


# \set option -> (parser, description of a valid value)
OPTIONS: dict[str, tuple] = {
    "temperature": (float, "a number"),
    "top_p": (float, "a number"),
    "top_k": (int, "a whole number"),
    "num_predict": (int, "a whole number"),
    "repeat_penalty": (float, "a number"),
    "seed": (int, "a whole number"),
    "reasoning_effort": (_parse_effort, "one of high, medium, low"),
}

HELP_TEXT = """Commands (prefix with \\ or /, case-insensitive):
  \\bye, \\quit, \\exit, \\q   stop the server and exit
  \\help, \\?                show this help
  \\clear                   clear the conversation (keeps the system prompt)
  \\system [text]           show or set the system prompt (resets the conversation)
  \\set [name value]        show or set a generation option: temperature, top_p,
                           top_k, num_predict, repeat_penalty, seed
  \\set reasoning_effort high|medium|low
                           how long the model thinks (high = longest, low = fastest answers)
  \\set think on|off        show reasoning dimmed (on) or hide it and ask the model
                           not to think when it supports that (off)
  \\show                    model, server, options and message count
  \\models                  list installed models
  \\load <model>            load another model and chat with it (resets the conversation)
  \\stats on|off            show the token/speed line after each answer
  \"\"\"                      start (and end) a multi-line message
Ctrl+C while the model is answering interrupts the answer; Ctrl+C at the prompt exits."""


def _write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class ThinkFilter:
    """Prints one streamed answer, separating reasoning from the answer text.

    Two inputs: ``feed_thinking()`` for the server's native ``message.thinking``
    deltas and ``feed()`` for ``message.content`` deltas. ``feed()`` additionally
    detects ``<think>...</think>`` tags (split across chunks or not) as a
    fallback. Reasoning is printed dimmed, or replaced by one ``[thinking…]``
    marker when *show_thinking* is False; the answer starts on its own line.
    The caller keeps the raw content for the history.
    """

    def __init__(self, show_thinking: bool = True, write=_write):
        self.show = show_thinking
        self._write = write
        self._buf = ""
        self.in_think = False
        self._marker_done = False
        self._skip_ws = False
        self._last = "\n"
        self._tag_opened = False  # True while inside a <think> block found in content text

    # -- public -----------------------------------------------------------
    def feed_thinking(self, text: str) -> None:
        """A native reasoning delta from the server."""
        if not text:
            return
        if self._buf:  # a pending '<' from content text can no longer be a tag
            rest, self._buf = self._buf, ""
            self._emit(rest)
        if not self.in_think:
            self._open()
        self._emit(text)

    def feed(self, text: str) -> None:
        """An answer (content) delta; closes a native thinking block and scans for tags."""
        if not text:
            return
        if self.in_think and not self._tag_opened:  # native thinking block ends here
            self._close()
        self._buf += text
        while self._buf:
            lt = self._buf.find("<")
            if lt < 0:
                self._emit(self._buf)
                self._buf = ""
                return
            if lt > 0:
                self._emit(self._buf[:lt])
                self._buf = self._buf[lt:]
            tag = self._match_tag()
            if tag is None:  # may still become a tag: wait for more data
                return
            if tag == "":  # a literal '<'
                self._emit("<")
                self._buf = self._buf[1:]
            else:
                self._buf = self._buf[len(tag):]
                self._toggle(tag)

    def flush(self) -> None:
        """End of stream: print whatever is still buffered (an unfinished tag is literal text)."""
        rest, self._buf = self._buf, ""
        if rest:
            self._emit(rest)
        self.in_think = False
        self._tag_opened = False

    # -- internals --------------------------------------------------------
    def _match_tag(self) -> str | None:
        """Tag at the start of the buffer; "" if there is none; None if undecidable yet."""
        head = self._buf[:_MAX_TAG_LEN].lower()
        for tag in _ALL_TAGS:
            if head.startswith(tag):
                return tag
        if len(head) < _MAX_TAG_LEN and any(t.startswith(head) for t in _ALL_TAGS):
            return None
        return ""

    def _emit(self, text: str) -> None:
        if self._skip_ws:
            text = text.lstrip("\r\n")
            if not text.strip():
                return
            self._skip_ws = False
        self._last = text[-1]
        if self.in_think:
            if self.show:
                self._write(dim(text))
        else:
            self._write(text)

    def _open(self) -> None:
        self.in_think = True
        self._skip_ws = True
        if self.show:
            self._write(dim("Thinking...") + "\n")
        elif not self._marker_done:
            self._write(dim("[thinking…]"))
            self._marker_done = True

    def _close(self) -> None:
        self.in_think = False
        self._tag_opened = False
        if self.show:
            if self._last != "\n":
                self._write("\n")
            self._write(dim("...done thinking.") + "\n\n")
        else:
            self._write("\n")
        self._skip_ws = True

    def _toggle(self, tag: str) -> None:
        if tag.startswith("</"):
            if self.in_think:
                self._close()
            else:
                # Closing tag without an opening one (the chat template pre-opened
                # the block): swallow it and the whitespace that follows.
                self._skip_ws = True
        elif not self.in_think:
            self._open()
            self._tag_opened = True


_EOS = object()


def _iter_lines_interruptible(r: httpx.Response):
    """Yield response lines while staying responsive to Ctrl+C on Windows.

    A blocking socket read on Windows only notices KeyboardInterrupt once data arrives,
    so the read happens in a helper thread and the main thread polls a queue.
    """
    q: queue.Queue = queue.Queue()

    def reader() -> None:
        try:
            for line in r.iter_lines():
                q.put(line)
        except Exception as e:  # noqa: BLE001 - forwarded to the consumer
            q.put(e)
        finally:
            q.put(_EOS)

    threading.Thread(target=reader, name="hfserve-chat-reader", daemon=True).start()
    while True:
        try:
            item = q.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is _EOS:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def error_text(r: httpx.Response) -> str:
    try:
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    body = (r.text or "").strip()
    return f"HTTP {r.status_code}" + (f": {body[:300]}" if body else "")


def _stats_line(done: dict) -> str:
    parts = []
    n = done.get("eval_count")
    if isinstance(n, (int, float)):
        parts.append(f"{int(n)} tokens")
        dur = done.get("eval_duration") or 0
        if isinstance(dur, (int, float)) and dur > 0:
            parts.append(f"{n / (dur / 1e9):.1f} tok/s")
    p = done.get("prompt_eval_count")
    if isinstance(p, (int, float)):
        parts.append(f"prompt {int(p)} tokens")
    total = done.get("total_duration") or 0
    if not parts and total:
        parts.append(f"{total / 1e9:.1f}s")
    line = "[" + " · ".join(parts) + "]" if parts else "[done]"
    if done.get("done_reason") == "length":
        line += " (max tokens reached)"
    return line


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in ("on", "1", "true", "yes", "y"):
        return True
    if v in ("off", "0", "false", "no", "n"):
        return False
    return None


class ChatSession:
    def __init__(self, host: str, port: int, model: str, system: str | None = None,
                 options: dict | None = None, think: bool = True):
        self.base = config.server_url(host, port)
        self.model = model
        self.system = system
        self.options: dict = dict(options or {})
        self.options.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
        self.history: list[dict] = []
        self.show_stats = True
        # One switch for both: show reasoning in the console and ask the server for it.
        self.think = think
        # trust_env=False: never send localhost traffic through an HTTP(S)_PROXY.
        self.client = httpx.Client(base_url=self.base, trust_env=False)

    # -- helpers ----------------------------------------------------------
    def messages(self) -> list[dict]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        return msgs + self.history

    def _unreachable(self, exc: Exception) -> None:
        print()
        print(red(f"Error: lost the connection to the server at {self.base} "
                  f"({type(exc).__name__}: {exc})"))
        print(dim(f"server log: {config.LOG_FILE}"))
        tail = process.log_tail(15)
        if tail:
            print(dim(tail))

    # -- commands ---------------------------------------------------------
    def cmd_help(self, _arg: str) -> None:
        print(HELP_TEXT)

    def cmd_clear(self, _arg: str) -> None:
        self.history.clear()
        print("Conversation cleared.")

    def cmd_system(self, arg: str) -> None:
        if not arg:
            print(f"System prompt: {self.system}" if self.system else "No system prompt set.")
            return
        self.system = arg
        self.history.clear()
        print("System prompt set (conversation cleared).")

    def cmd_set(self, arg: str) -> None:
        if not arg:
            self._print_options()
            return
        name, _, value = arg.partition(" ")
        name = name.strip().lower().replace("-", "_")
        value = value.strip()
        if name in ("think", "thinking"):
            flag = _parse_bool(value)
            if flag is None:
                print(red("Usage: \\set think on|off"))
                return
            self.think = flag
            print(f"Reasoning {'on: shown dimmed' if flag else 'off: hidden, and the model is asked not to think'}.")
            return
        if name == "stats":
            self.cmd_stats(value)
            return
        if name not in OPTIONS:
            print(red(f"Unknown option '{name}'. Options: {', '.join(OPTIONS)}, think, stats"))
            return
        parser, expected = OPTIONS[name]
        if not value:
            print(f"{name} = {self.options.get(name, '(server default)')}")
            return
        if value.lower() in ("default", "none", "unset"):
            if name == "reasoning_effort":  # always sent; "default" means medium
                self.options[name] = DEFAULT_REASONING_EFFORT
                print(f"{name} = {DEFAULT_REASONING_EFFORT}")
            else:
                self.options.pop(name, None)
                print(f"{name} reset to the server default.")
            return
        try:
            parsed = parser(value)
        except ValueError:
            print(red(f"{name} must be {expected}."))
            return
        self.options[name] = parsed
        print(f"{name} = {parsed}")

    def _print_options(self) -> None:
        if self.options:
            for k, v in self.options.items():
                print(f"  {k:<17} {v}")
        else:
            print("  (server defaults)")
        print(f"  {'think':<17} {'on' if self.think else 'off'}")
        print(f"  {'stats':<17} {'on' if self.show_stats else 'off'}")

    def cmd_show(self, _arg: str) -> None:
        print(f"Model    : {self.model}")
        print(f"Server   : {self.base}")
        print(f"System   : {self.system if self.system else '(none)'}")
        print("Options  :")
        self._print_options()
        print(f"Messages : {len(self.history)}")

    def cmd_stats(self, arg: str) -> None:
        flag = _parse_bool(arg)
        if flag is None:
            print(f"Stats are {'on' if self.show_stats else 'off'}. Usage: \\stats on|off")
            return
        self.show_stats = flag
        print(f"Stats {'on' if flag else 'off'}.")

    def cmd_models(self, _arg: str) -> None:
        r = self.client.get("/api/tags", timeout=10)
        if r.status_code != 200:
            print(red(f"Error: {error_text(r)}"))
            return
        models = (r.json() or {}).get("models") or []
        if not models:
            print("No models installed.")
            return
        width = max(len(m.get("name", "")) for m in models)
        for i, m in enumerate(models, 1):
            name = m.get("name", "")
            mark = "*" if name.lower() == self.model.lower() else " "
            family = (m.get("details") or {}).get("family") or ""
            print(f"{mark} {i:>2}  {name:<{width}}  {human_size(m.get('size') or 0):>9}  {family}")

    def cmd_load(self, arg: str) -> None:
        if not arg:
            print(red("Usage: \\load <model>"))
            return
        print(f"Loading {arg} ... ", end="", flush=True)
        r = self.client.post("/api/load", json={"model": arg}, timeout=900)
        if r.status_code != 200:
            print()
            print(red(f"Error: {error_text(r)}"))
            return
        data = r.json() or {}
        vram = data.get("vram_bytes") or 0
        print(f"done: {data.get('device')} / {data.get('dtype')} in {float(data.get('load_seconds') or 0):.1f}s"
              + (f" (VRAM {human_size(vram)})" if vram else ""))
        new_model = data.get("model") or arg
        if new_model != self.model:
            self.model = new_model
            self.history.clear()
            print(f"Now chatting with {bold(self.model)} (conversation cleared).")

    COMMANDS = {
        "help": cmd_help, "?": cmd_help,
        "clear": cmd_clear,
        "system": cmd_system,
        "set": cmd_set,
        "show": cmd_show,
        "stats": cmd_stats,
        "models": cmd_models,
        "load": cmd_load,
    }
    EXIT_COMMANDS = ("bye", "quit", "exit", "q")

    def run_command(self, line: str) -> str | None:
        """Execute a backslash command. Returns "bye" to leave, "error" if the server is gone."""
        body = line[1:].strip()
        name, _, arg = body.partition(" ")
        name = name.lower()
        arg = arg.strip()
        if name in self.EXIT_COMMANDS:
            print("Stopping server ...")
            return "bye"
        handler = self.COMMANDS.get(name)
        if handler is None:
            print("Unknown command; \\help for help")
            return None
        try:
            handler(self, arg)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except httpx.TransportError as e:
            self._unreachable(e)
            return "error"
        except httpx.HTTPError as e:
            print(red(f"Error: {e}"))
        return None

    # -- chat -------------------------------------------------------------
    def send(self, text: str) -> str | None:
        """Send one user message and stream the answer. Returns "error" if the server is gone."""
        self.history.append({"role": "user", "content": text})
        body = {
            "model": self.model,
            "messages": self.messages(),
            "stream": True,
            "think": self.think,
            "options": self.options,
        }
        filt = ThinkFilter(self.think)
        chunks: list[str] = []  # answer text only; reasoning is never stored
        done = False
        interrupted = False
        try:
            with self.client.stream("POST", "/api/chat", json=body,
                                    timeout=httpx.Timeout(None, connect=10)) as r:
                if r.status_code != 200:
                    r.read()
                    print(red(f"Error: {error_text(r)}"))
                    self.history.pop()
                    return None
                for line in _iter_lines_interruptible(r):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if obj.get("error") and "message" not in obj:
                        filt.flush()
                        print()
                        print(red(f"Error: {obj['error']}"))
                        break
                    msg = obj.get("message") or {}
                    thinking = msg.get("thinking") or ""
                    if thinking:
                        filt.feed_thinking(thinking)
                    content = msg.get("content") or ""
                    if content:
                        chunks.append(content)
                        filt.feed(content)
                    if obj.get("done"):
                        done = True
                        filt.flush()
                        print()
                        if self.show_stats:
                            print(dim(_stats_line(obj)))
                        break
        except KeyboardInterrupt:
            # Leaving the `with` block closed the response: the server cancels generation.
            interrupted = True
        except httpx.TransportError as e:
            filt.flush()
            self._unreachable(e)
            return "error"

        answer = "".join(chunks)
        if interrupted:
            filt.flush()
            print("\n[interrupted]")
        elif not done:
            filt.flush()
            print()
        if answer:
            self.history.append({"role": "assistant", "content": answer})
        else:
            # No answer came back (interrupted at once, or an error): forget the question too.
            self.history.pop()
        return None

    # -- main loop --------------------------------------------------------
    def read_message(self) -> str:
        """Read one message; handles the Ollama-style multi-line block. Raises EOFError / KeyboardInterrupt."""
        line = input(PROMPT)
        if not line.startswith(MULTILINE_MARK):
            return line.strip()
        first = line[len(MULTILINE_MARK):]
        if first.rstrip().endswith(MULTILINE_MARK):  # complete on one line: """text"""
            return first.rstrip()[:-len(MULTILINE_MARK)].strip()
        lines = [first]
        while True:
            more = input(CONT_PROMPT)
            if more.rstrip().endswith(MULTILINE_MARK):
                lines.append(more.rstrip()[:-len(MULTILINE_MARK)])
                break
            lines.append(more)
        return "\n".join(lines).strip()

    def loop(self) -> str:
        print(f"Chatting with {bold(self.model)}  (server {self.base})")
        print(f"Type a message and press Enter.  {cyan(BYE_CMD)} stops the server and exits.  "
              f"{cyan(HELP_CMD)} for commands.")
        effort = self.options.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
        print(dim("Reasoning is shown dimmed. \\set think off hides it; "
                  f"\\set reasoning_effort low|medium|high trades depth for speed (now: {effort})."))
        try:
            while True:
                try:
                    text = self.read_message()
                except (EOFError, KeyboardInterrupt):
                    print("\nStopping server ...")
                    return "bye"
                if not text:
                    continue
                if text[0] in "\\/" and len(text) > 1 and not text[1].isspace():
                    result = self.run_command(text)
                    if result:
                        return result
                    continue
                result = self.send(text)
                if result:
                    return result
        finally:
            try:
                self.client.close()
            except Exception:
                pass


def run_chat(host: str, port: int, model: str, system: str | None = None,
             options: dict | None = None, think: bool = True) -> str:
    """Run the interactive chat. Returns "bye" (user wants the server stopped) or "error"."""
    setup_console()
    session = ChatSession(host, port, model, system=system, options=options, think=think)
    return session.loop()
