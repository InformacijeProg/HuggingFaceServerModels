"""Transformers text-generation engine: load one model at a time and stream tokens from it.

Design notes
------------
* One ``threading.Lock`` guards load, unload and generation, so a load can never race a
  running generation and concurrent requests are served one after another.
* ``generate_stream`` runs ``model.generate`` in a daemon thread and consumes a
  ``TextIteratorStreamer``; deltas pass through two small filters before they are
  yielded: ``_StopFilter`` (stop strings, never prints a partial stop string) and
  ``ReasoningSplitter`` (splits ``<think>...</think>`` style reasoning from the answer).
* Some reasoning models end the *prompt* with the opening think tag, so the model output
  starts directly with reasoning; the splitter detects that from the prompt tail and
  starts in THINKING mode.
* The chat template may end assistant turns with a token that is not in
  ``generation_config.eos_token_id`` (EuroLLM: ``<|im_end|>``); it is detected at
  load time and added to the eos list, otherwise generation would run past the answer.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from . import config
from .registry import LocalModel

log = logging.getLogger("hfserve.engine")

# Opening reasoning tag at the end of a prompt / start of an output, e.g. <think>, <ifm|think_fast>
# (a closing tag such as </ifm|think> must not match: it is what we append for think=False).
_THINK_OPEN_RE = re.compile(r"(<(?!/)[^<>\s]*think[^<>\s]*>)\s*$", re.IGNORECASE)
# A complete tag of any kind at the start of the output.
_ANY_TAG_RE = re.compile(r"<[^<>\s]*>")
# How many leading characters we are willing to hold back while deciding whether the
# output starts with an opening tag.
_TAG_DETECT_LIMIT = 64
# Minimum room left for the answer when the prompt (almost) fills the context.
_MIN_NEW_TOKENS = 16


# --------------------------------------------------------------------------- device / dtype
_DTYPE_NAMES: dict[str, torch.dtype] = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower().removeprefix("torch.")
    try:
        return _DTYPE_NAMES[key]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}; use float16, bfloat16 or float32.") from None


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def pick_device() -> str:
    forced = os.environ.get("HFSERVE_DEVICE", "").strip()
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_bf16_native() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older torch without the keyword
        return bool(torch.cuda.is_bf16_supported())


def pick_dtype(device: str) -> torch.dtype:
    forced = os.environ.get("HFSERVE_DTYPE", "").strip()
    if forced:
        return parse_dtype(forced)
    if device.startswith("cuda"):
        return torch.bfloat16 if _cuda_bf16_native() else torch.float16
    return torch.float32


# --------------------------------------------------------------------------- results
@dataclass
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    prompt_seconds: float      # time to first token (tokenisation + prefill)
    gen_seconds: float         # time from first token to the end
    total_seconds: float
    finish_reason: str         # "stop" | "length" | "cancel"
    text: str                  # the answer (reasoning block removed)
    thinking: str = ""         # the reasoning block, "" when the model produced none


# --------------------------------------------------------------------------- helpers
def content_to_text(content: Any) -> str:
    """Flatten a message ``content`` (str or OpenAI-style list of parts) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in (None, "text", "input_text") and isinstance(p.get("text"), str):
                    parts.append(p["text"])
        return "\n".join(parts)
    return str(content)


def _normalise_messages(messages: list[dict] | None) -> list[dict]:
    out: list[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            raise ValueError("Each message must be an object with 'role' and 'content'.")
        role = str(m.get("role") or "user")
        out.append({"role": role, "content": content_to_text(m.get("content"))})
    return out


class _EventStoppingCriteria(StoppingCriteria):
    """Stops generation as soon as any of the given events is set (checked after every token)."""

    def __init__(self, *events: threading.Event):
        super().__init__()
        self.events = events

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        done = any(e.is_set() for e in self.events)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


class _Streamer(TextIteratorStreamer):
    """TextIteratorStreamer that records when the first generated token arrived."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.first_token_at: float | None = None
        self.token_count = 0

    def put(self, value):
        if not (self.skip_prompt and getattr(self, "next_tokens_are_prompt", False)):
            if self.first_token_at is None:
                self.first_token_at = time.perf_counter()
            self.token_count += int(value.numel()) if hasattr(value, "numel") else 1
        super().put(value)


class _StopFilter:
    """Cuts the stream at the first stop string; never lets a partial stop string through."""

    def __init__(self, stops: list[str] | None):
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.hit = False

    def feed(self, piece: str) -> str:
        """Return the text that is safe to emit (may be ""). Sets ``hit`` when a stop string was found."""
        if self.hit:
            return ""
        if not self.stops:
            return piece
        self.buf += piece
        best = -1
        for s in self.stops:
            i = self.buf.find(s)
            if i != -1 and (best == -1 or i < best):
                best = i
        if best != -1:
            out, self.buf, self.hit = self.buf[:best], "", True
            return out
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out, self.buf = self.buf[:len(self.buf) - hold], self.buf[len(self.buf) - hold:]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


class ReasoningSplitter:
    """Splits generated text into ("thinking", delta) and ("content", delta) pieces.

    * ``close_tag`` given: the prompt ended with the opening tag, so the output starts inside
      the reasoning block (Qwen3-style templates with their own opener).
    * otherwise start in CONTENT mode, but if the first non-whitespace output is an opening
      tag containing "think" (DeepSeek-R1 / Qwen3 style ``<think>``) switch to THINKING and
      swallow the opener.
    * in THINKING mode only text that cannot be part of a partial close tag is emitted.
    """

    def __init__(self, close_tag: str | None = None):
        self.close_tag = close_tag
        self.mode = "thinking" if close_tag else "content"
        self.thinking = ""
        self.content = ""
        self._buf = ""
        self._detect_opener = close_tag is None
        self._strip_newlines = bool(close_tag)  # drop newlines right after an opening tag

    @staticmethod
    def close_tag_for(opener: str) -> str:
        return "</" + opener[1:]

    @classmethod
    def from_prompt_tail(cls, prompt_tail: str) -> "ReasoningSplitter":
        m = _THINK_OPEN_RE.search(prompt_tail or "")
        return cls(cls.close_tag_for(m.group(1))) if m else cls()

    @staticmethod
    def _partial_suffix(buf: str, tag: str) -> int:
        """Length of the longest suffix of *buf* that is a proper prefix of *tag*."""
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def _emit(self, kind: str, text: str, out: list[tuple[str, str]]) -> None:
        if not text:
            return
        if kind == "thinking":
            self.thinking += text
        else:
            self.content += text
        out.append((kind, text))

    def feed(self, piece: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not piece:
            return out
        self._buf += piece
        while self._buf:
            if self.mode == "content" and self._detect_opener:
                stripped = self._buf.lstrip()
                if not stripped:
                    return out  # only whitespace so far; wait for real text
                if stripped.startswith("<") and len(stripped) <= _TAG_DETECT_LIMIT:
                    m = _ANY_TAG_RE.match(stripped)
                    if m is None:
                        if re.fullmatch(r"<[^<>\s]*", stripped):
                            return out  # could still become a tag; hold
                    elif _THINK_OPEN_RE.match(m.group(0)) and not m.group(0).startswith("</"):
                        self.close_tag = self.close_tag_for(m.group(0))
                        self.mode = "thinking"
                        self._buf = stripped[m.end():]
                        self._strip_newlines = True
                        self._detect_opener = False
                        continue
                self._detect_opener = False
            if self._strip_newlines:
                self._buf = self._buf.lstrip("\n")
                if not self._buf:
                    return out
                self._strip_newlines = False
            if self.mode == "thinking" and self.close_tag:
                idx = self._buf.find(self.close_tag)
                if idx != -1:
                    self._emit("thinking", self._buf[:idx], out)
                    self._buf = self._buf[idx + len(self.close_tag):]
                    self.mode = "content"
                    self._strip_newlines = True
                    continue
                hold = self._partial_suffix(self._buf, self.close_tag)
                emit_len = len(self._buf) - hold
                self._emit("thinking", self._buf[:emit_len], out)
                self._buf = self._buf[emit_len:]
                return out
            self._emit("content", self._buf, out)
            self._buf = ""
            return out
        return out

    def finish(self) -> list[tuple[str, str]]:
        """Flush whatever is still held back (a possible partial tag that never completed)."""
        out: list[tuple[str, str]] = []
        if self._buf and not self._strip_newlines:
            self._emit("thinking" if self.mode == "thinking" else "content", self._buf, out)
        self._buf = ""
        return out


# --------------------------------------------------------------------------- engine
class Engine:
    def __init__(self) -> None:
        self.local: LocalModel | None = None
        self.model = None
        self.tokenizer = None
        self.device: str | None = None
        self.dtype: torch.dtype | None = None
        self.loaded_at: str | None = None
        self.load_seconds: float = 0.0
        self.max_context: int = 0
        self.needs_remote_code: bool = False
        self.eos_token_ids: list[int] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- state
    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def model_name(self) -> str | None:
        return self.local.repo_id if (self.local is not None and self.is_loaded) else None

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def template_supports(self, name: str) -> bool:
        """True if the loaded tokenizer's chat template mentions *name* (e.g. "enable_thinking")."""
        tpl = getattr(self.tokenizer, "chat_template", None) if self.tokenizer is not None else None
        return isinstance(tpl, str) and name in tpl

    def _vram_bytes(self) -> int:
        if self.model is None or not torch.cuda.is_available():
            return 0
        try:
            dev = self.model.device
            if dev.type != "cuda":
                return 0
            return int(torch.cuda.memory_allocated(dev))
        except Exception:
            return int(torch.cuda.memory_allocated())

    def info(self) -> dict:
        if not self.is_loaded:
            return {
                "model": None, "device": None, "dtype": None, "vram_bytes": 0,
                "loaded_at": None, "load_seconds": 0.0, "needs_remote_code": False, "max_context": None,
            }
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": dtype_name(self.dtype) if self.dtype is not None else None,
            "vram_bytes": self._vram_bytes(),
            "loaded_at": self.loaded_at,
            "load_seconds": round(self.load_seconds, 3),
            "needs_remote_code": self.needs_remote_code,
            "max_context": self.max_context,
        }

    # ----------------------------------------------------------------- load / unload
    def load(self, local: LocalModel, device: str | None = None, dtype: str | None = None) -> dict:
        with self._lock:
            if self.is_loaded and self.local is not None and self.local.repo_id == local.repo_id:
                return self.info()
            self._unload_locked()

            dev = (device or "").strip() or pick_device()
            if dev.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but no CUDA device is available (torch.cuda.is_available() is False).")
            dt = parse_dtype(dtype) if dtype else pick_dtype(dev)
            needs = bool(local.needs_remote_code)
            if needs:
                log.warning("%s ships custom model code (auto_map in config.json): loading with trust_remote_code=True from %s",
                            local.repo_id, local.path)
            log.info("Loading %s on %s (%s) ...", local.repo_id, dev, dtype_name(dt))

            started = time.perf_counter()
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(local.path), trust_remote_code=needs)
                model = self._from_pretrained(str(local.path), dt, dev, needs)
            except Exception:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.local = local
            self.model = model
            self.tokenizer = tokenizer
            self.device = dev
            self.dtype = dt
            self.needs_remote_code = needs
            self.max_context = self._read_max_context(model)
            self.eos_token_ids = self._collect_eos_ids()
            self.load_seconds = time.perf_counter() - started
            self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            log.info("Loaded %s in %.1fs (max_context=%d, eos_token_ids=%s, vram=%d MB)",
                     local.repo_id, self.load_seconds, self.max_context, self.eos_token_ids,
                     self._vram_bytes() // (1024 * 1024))
            return self.info()

    @staticmethod
    def _from_pretrained(path: str, dt: torch.dtype, dev: str, needs: bool):
        kwargs: dict[str, Any] = {
            "device_map": dev,
            "low_cpu_mem_usage": True,
            "trust_remote_code": needs,
            "attn_implementation": "sdpa",
        }
        new_api = int(transformers.__version__.split(".")[0]) >= 5
        try:
            if new_api:
                return AutoModelForCausalLM.from_pretrained(path, dtype=dt, **kwargs)
            return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, **kwargs)
        except TypeError as e:
            log.debug("from_pretrained dtype kwarg fallback (%s)", e)
            if new_api:
                return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, **kwargs)
            return AutoModelForCausalLM.from_pretrained(path, dtype=dt, **kwargs)
        except (ImportError, ValueError) as e:
            if "accelerate" not in str(e).lower():
                raise
            log.warning("device_map unavailable (%s); loading on CPU and moving to %s", e, dev)
            kwargs.pop("device_map")
            model = AutoModelForCausalLM.from_pretrained(path, dtype=dt, **kwargs)
            return model.to(dev)

    @staticmethod
    def _read_max_context(model) -> int:
        cfg = getattr(model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
            v = getattr(cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        return 4096

    def _detect_end_of_turn_ids(self) -> list[int]:
        """Find the token the chat template uses to close an assistant turn (e.g. <|im_end|>)."""
        tok = self.tokenizer
        if not getattr(tok, "chat_template", None):
            return []
        try:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
                tokenize=True, add_generation_prompt=False,
            )
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            marker_ids = self._special_or_added_ids()
            for tid in reversed(list(ids)):
                tid = int(tid)
                if tok.decode([tid]).strip() == "":
                    continue
                return [tid] if tid in marker_ids else []
        except Exception as e:  # a broken template must not prevent loading
            log.debug("end-of-turn detection failed: %s", e)
        return []

    def _special_or_added_ids(self) -> set[int]:
        """Ids of special *and* added tokens (transformers 5's all_special_ids holds only bos/eos/pad)."""
        tok = self.tokenizer
        ids: set[int] = set()
        for getter in (
            lambda: tok.all_special_ids,
            lambda: tok.get_added_vocab().values(),
            lambda: tok.added_tokens_decoder.keys(),
        ):
            try:
                ids.update(int(i) for i in getter())
            except Exception:
                pass
        return ids

    def _collect_eos_ids(self) -> list[int]:
        ids: list[int] = []
        gen_cfg = getattr(self.model, "generation_config", None)
        cfg_eos = getattr(gen_cfg, "eos_token_id", None)
        if cfg_eos is None:
            cfg_eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(cfg_eos, int):
            ids.append(cfg_eos)
        elif isinstance(cfg_eos, (list, tuple)):
            ids.extend(int(i) for i in cfg_eos)
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        detected = self._detect_end_of_turn_ids()
        if detected:
            log.info("Chat template ends assistant turns with %r (id %d); added to eos ids.",
                     self.tokenizer.decode(detected), detected[0])
        ids.extend(detected)
        return sorted(set(ids))

    def unload(self) -> str | None:
        with self._lock:
            return self._unload_locked()

    def _unload_locked(self) -> str | None:
        name = self.model_name
        self.model = None
        self.tokenizer = None
        self.local = None
        self.device = None
        self.dtype = None
        self.loaded_at = None
        self.load_seconds = 0.0
        self.max_context = 0
        self.needs_remote_code = False
        self.eos_token_ids = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if name:
            log.info("Unloaded %s", name)
        return name

    # ----------------------------------------------------------------- prompts
    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("No model is loaded.")

    @staticmethod
    def _fallback_text(messages: list[dict]) -> str:
        labels = {"system": "System", "user": "User", "assistant": "Assistant"}
        parts = [f"{labels.get(m['role'], m['role'].capitalize())}: {m['content'].strip()}" for m in messages]
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _render_text(self, messages: list[dict], template_kwargs: dict | None, think: bool | None = None) -> str:
        """Render *messages* to the prompt text (chat template or the plain fallback).

        ``think=False`` on a template without an ``enable_thinking`` switch: when the rendered
        prompt ends with an opening reasoning tag, the matching close tag is appended so the
        model answers directly.
        """
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                           **(template_kwargs or {}))
        else:
            text = self._fallback_text(messages)
        if think is False and not self.template_supports("enable_thinking"):
            m = _THINK_OPEN_RE.search(text)
            if m:
                text = text + ReasoningSplitter.close_tag_for(m.group(1)) + "\n"
        return text

    def _encode_text(self, text: str) -> dict[str, torch.Tensor]:
        # The chat template already contains every special token; add_special_tokens=False
        # is exactly what apply_chat_template(tokenize=True) does internally.
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        out = {k: v for k, v in dict(enc).items() if isinstance(v, torch.Tensor)}
        out.pop("token_type_ids", None)  # some tokenizers emit them and generate() rejects them
        return out

    def render_prompt(self, messages: list[dict], template_kwargs: dict | None = None,
                      think: bool | None = None) -> str:
        """The prompt text the model would see for *messages* (debugging aid)."""
        self._require_loaded()
        return self._render_text(_normalise_messages(messages), template_kwargs, think)

    def build_inputs(self, messages: list[dict] | None, prompt: str | None, raw: bool = False,
                     max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS, system: str | None = None,
                     template_kwargs: dict | None = None, think: bool | None = None) -> tuple[dict, int, str]:
        """Tokenise a request and fit it into the context window.

        Returns ``(inputs, effective_max_new_tokens, prompt_tail)`` where *inputs* are on the
        model device and *prompt_tail* is the end of the prompt text (used to detect an
        opening reasoning tag). ``max_new_tokens <= 0`` means "fill the remaining context".
        """
        self._require_loaded()
        tok = self.tokenizer
        fill = max_new_tokens <= 0
        reserve = _MIN_NEW_TOKENS if fill else int(max_new_tokens)

        if raw:
            if prompt is None:
                raise ValueError("raw generation needs a 'prompt'.")
            text = prompt
            enc = dict(tok(prompt, return_tensors="pt", add_special_tokens=True))
            enc.pop("token_type_ids", None)
        else:
            msgs = _normalise_messages(messages)
            if prompt is not None and not msgs:
                msgs = [{"role": "user", "content": prompt}]
            elif prompt is not None:
                msgs.append({"role": "user", "content": prompt})
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            if not msgs:
                raise ValueError("No messages or prompt given.")
            text = self._render_text(msgs, template_kwargs, think)
            enc = self._encode_text(text)
            # Drop the oldest non-system messages until prompt + answer fits the context.
            while enc["input_ids"].shape[-1] + reserve > self.max_context:
                last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=-1)
                droppable = [i for i, m in enumerate(msgs) if m["role"] != "system" and i != last_user]
                if not droppable:
                    break
                dropped = msgs.pop(droppable[0])
                log.info("Context full: dropped oldest %s message (%d chars)", dropped["role"], len(dropped["content"]))
                text = self._render_text(msgs, template_kwargs, think)
                enc = self._encode_text(text)

        n = int(enc["input_ids"].shape[-1])
        if n + _MIN_NEW_TOKENS > self.max_context:  # last resort: keep the newest tokens only
            keep = max(1, self.max_context - _MIN_NEW_TOKENS)
            log.warning("Prompt (%d tokens) exceeds the context window (%d); truncating to the last %d tokens",
                        n, self.max_context, keep)
            enc = {k: v[..., -keep:] for k, v in enc.items()}
            n = keep
        if fill:
            effective = max(_MIN_NEW_TOKENS, self.max_context - n)
        elif n + int(max_new_tokens) > self.max_context:
            effective = max(_MIN_NEW_TOKENS, self.max_context - n)
            log.info("num_predict capped to %d to fit the context window", effective)
        else:
            effective = int(max_new_tokens)

        device = self.model.device
        inputs = {k: v.to(device) for k, v in enc.items()}
        return inputs, effective, text[-256:]

    # ----------------------------------------------------------------- generation
    def generate_stream(self, *, messages: list[dict] | None = None, prompt: str | None = None,
                        system: str | None = None, raw: bool = False, max_new_tokens: int | None = None,
                        temperature: float | None = None, top_p: float | None = None, top_k: int | None = None,
                        repetition_penalty: float | None = None, stop: list[str] | None = None,
                        seed: int | None = None, cancel: threading.Event | None = None,
                        template_kwargs: dict | None = None, think: bool | None = None,
                        ) -> Iterator[tuple[str, str] | GenerationStats]:
        """Yield ``("thinking"|"content", delta)`` tuples, then exactly one GenerationStats.

        *think*: None = template default; False = suppress reasoning (via the template's
        ``enable_thinking`` kwarg when the caller put it in *template_kwargs*, otherwise by
        closing the reasoning block in the prompt, see ``_render_text``).
        """
        cancel = cancel if cancel is not None else threading.Event()
        internal_stop = threading.Event()
        t0 = time.perf_counter()

        with self._lock:
            self._require_loaded()
            model, tok = self.model, self.tokenizer
            if cancel.is_set():
                yield GenerationStats(0, 0, 0.0, 0.0, time.perf_counter() - t0, "cancel", "", "")
                return

            max_new = config.DEFAULT_MAX_NEW_TOKENS if max_new_tokens is None else int(max_new_tokens)
            temp = config.DEFAULT_TEMPERATURE if temperature is None else float(temperature)
            p = config.DEFAULT_TOP_P if top_p is None else float(top_p)
            k = config.DEFAULT_TOP_K if top_k is None else int(top_k)
            rep = config.DEFAULT_REPETITION_PENALTY if repetition_penalty is None else float(repetition_penalty)

            inputs, max_new, prompt_tail = self.build_inputs(
                messages, prompt, raw=raw, max_new_tokens=max_new, system=system,
                template_kwargs=template_kwargs, think=think)
            input_len = int(inputs["input_ids"].shape[-1])
            if seed is not None:
                torch.manual_seed(int(seed))

            streamer = _Streamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=None)
            gen_kwargs: dict[str, Any] = dict(
                inputs,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([_EventStoppingCriteria(cancel, internal_stop)]),
            )
            if self.eos_token_ids:
                gen_kwargs["eos_token_id"] = list(self.eos_token_ids)
            if temp <= 0:
                gen_kwargs["do_sample"] = False
            else:
                # Pass every sampling knob explicitly so a model's own generation_config.json
                # cannot silently override the request. transformers treats top_k=0,
                # top_p=1.0 and repetition_penalty=1.0 as "disabled".
                gen_kwargs.update(do_sample=True, temperature=temp,
                                  top_p=p if 0 < p <= 1 else 1.0,
                                  top_k=k if k and k > 0 else 0)
            gen_kwargs["repetition_penalty"] = rep if rep and rep > 0 else 1.0

            result: dict[str, Any] = {}

            def _worker() -> None:
                try:
                    result["ids"] = model.generate(**gen_kwargs)
                except BaseException as e:  # noqa: BLE001 - must reach the consumer
                    result["error"] = e
                finally:
                    try:
                        streamer.end()  # unblocks the consumer even if generate() blew up
                    except Exception:  # noqa: BLE001 - never leave the consumer waiting
                        streamer.text_queue.put(streamer.stop_signal)

            thread = threading.Thread(target=_worker, name="hfserve-generate", daemon=True)
            stop_filter = _StopFilter(stop)
            splitter = ReasoningSplitter.from_prompt_tail(prompt_tail)
            thread.start()
            try:
                for piece in streamer:
                    if not piece:
                        continue
                    safe = stop_filter.feed(piece)
                    for kind, delta in splitter.feed(safe):
                        yield (kind, delta)
                    if stop_filter.hit:
                        internal_stop.set()
                        break
                for kind, delta in splitter.feed(stop_filter.flush()):
                    yield (kind, delta)
                for kind, delta in splitter.finish():
                    yield (kind, delta)
            finally:
                internal_stop.set()
                thread.join()

            end = time.perf_counter()
            if "error" in result:
                raise result["error"]
            ids = result.get("ids")
            seq = getattr(ids, "sequences", ids)
            if seq is not None and hasattr(seq, "shape"):
                completion_tokens = max(0, int(seq.shape[-1]) - input_len)
            else:
                completion_tokens = streamer.token_count

            if stop_filter.hit:
                finish = "stop"
            elif cancel.is_set():
                finish = "cancel"
            elif completion_tokens >= max_new:
                finish = "length"
            else:
                finish = "stop"
            first = streamer.first_token_at or end
            yield GenerationStats(
                prompt_tokens=input_len,
                completion_tokens=completion_tokens,
                prompt_seconds=first - t0,
                gen_seconds=max(0.0, end - first),
                total_seconds=end - t0,
                finish_reason=finish,
                text=splitter.content,
                thinking=splitter.thinking,
            )

    def generate(self, **kwargs) -> GenerationStats:
        stats: GenerationStats | None = None
        for item in self.generate_stream(**kwargs):
            if isinstance(item, GenerationStats):
                stats = item
        assert stats is not None
        return stats

    def count_tokens(self, text: str) -> int:
        self._require_loaded()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


engine = Engine()


# --------------------------------------------------------------------------- self-check
def _selftest() -> None:
    """Exercise ReasoningSplitter and _StopFilter with chunks split inside the tags."""
    def run(splitter: ReasoningSplitter, chunks: list[str]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for c in chunks:
            out.extend(splitter.feed(c))
        out.extend(splitter.finish())
        return out

    def joined(deltas: list[tuple[str, str]], kind: str) -> str:
        return "".join(d for k, d in deltas if k == kind)

    # 1. Prompt ends with the opener; close tag arrives split across chunks.
    s = ReasoningSplitter.from_prompt_tail("<|ifm|im_start|>assistant\n<ifm|think>\n")
    assert s.close_tag == "</ifm|think>" and s.mode == "thinking"
    d = run(s, ["\nLet me", " think <", "/ifm|th", "ink>\n\nHello", " there"])
    assert joined(d, "thinking") == "Let me think ", d
    assert joined(d, "content") == "Hello there", d
    assert all(k == "thinking" for k, _ in d[:2]) and d[-1][0] == "content", d
    assert "<" not in joined(d, "thinking"), d

    # 2. A "<" inside thinking that is not the close tag is emitted once resolved.
    s = ReasoningSplitter("</think>")
    d = run(s, ["a < b and", " x<y", "</th", "ink>ans"])
    assert joined(d, "thinking") == "a < b and x<y" and joined(d, "content") == "ans", d

    # 3. Opener detected at the start of the output (DeepSeek/Qwen3 style).
    s = ReasoningSplitter()
    d = run(s, ["<th", "ink>\nreason", "ing</think>\n", "answer"])
    assert joined(d, "thinking") == "reasoning" and joined(d, "content") == "answer", d

    # 4. Plain output that starts with a '<' that is not a tag stays content.
    s = ReasoningSplitter()
    d = run(s, ["<3 ", "you"])
    assert joined(d, "content") == "<3 you" and joined(d, "thinking") == "", d

    # 5. Never-closed reasoning ends up entirely in thinking.
    s = ReasoningSplitter("</think>")
    d = run(s, ["still thinking", " when cut <", "/thi"])
    assert joined(d, "thinking") == "still thinking when cut </thi" and joined(d, "content") == "", d

    # 5b. A prompt that ends with a *closing* tag (think=False trick) starts in CONTENT mode.
    s = ReasoningSplitter.from_prompt_tail("assistant\n<ifm|think_faster>\n</ifm|think_faster>\n")
    assert s.mode == "content" and s.close_tag is None
    assert "".join(d for k, d in run(s, ["Hello", " world"]) if k == "content") == "Hello world"

    # 6. Stop strings: partial stop never leaks, text after the stop is cut.
    f = _StopFilter(["<|end|>"])
    out = "".join(f.feed(c) for c in ["Hello <", "|en", "d|> more"])
    assert out == "Hello " and f.hit, out
    f = _StopFilter(["STOP"])
    out = "".join(f.feed(c) for c in ["abc ST", "x"]) + f.flush()
    assert out == "abc STx" and not f.hit, out
    print("engine self-check OK")


if __name__ == "__main__":
    _selftest()
