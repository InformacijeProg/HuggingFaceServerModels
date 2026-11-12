"""FastAPI server implementing docs/API.md (Ollama-style and OpenAI-compatible endpoints).

Importing this module has no side effects; ``run_server`` starts uvicorn programmatically.

Streaming design: every streaming request gets its own ``threading.Event`` (cancel) and an
``asyncio.Queue``. A worker thread drives ``engine.generate_stream`` and hands each item to
the event loop with ``call_soon_threadsafe``; the response is an async generator that reads
the queue. When the client disconnects Starlette closes the generator, its ``finally`` sets
the cancel event and the engine stops within one decoding step.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

import hfserve
from . import config, process, registry
from .engine import GenerationStats, content_to_text, engine

log = logging.getLogger("hfserve.server")

# --------------------------------------------------------------------------- shared state
uvicorn_server = None                      # uvicorn.Server, set by run_server (for /api/shutdown)
_load_lock: asyncio.Lock | None = None     # created per event loop in the lifespan
_active_cancels: set[threading.Event] = set()
_active_guard = threading.Lock()
_exit_guard = threading.Lock()
_exiting = False

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def cancel_all_generations() -> None:
    with _active_guard:
        events = list(_active_cancels)
    for ev in events:
        ev.set()


def _track(ev: threading.Event) -> None:
    with _active_guard:
        _active_cancels.add(ev)


def _untrack(ev: threading.Event) -> None:
    with _active_guard:
        _active_cancels.discard(ev)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _load_lock
    _load_lock = asyncio.Lock()
    try:
        yield
    finally:
        cancel_all_generations()


app = FastAPI(title="hfserve", version=hfserve.__version__, lifespan=_lifespan)


# --------------------------------------------------------------------------- errors
@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return JSONResponse({"error": "Invalid request: " + "; ".join(parts)}, status_code=400)


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def _as_http_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, registry.ModelNotFound):
        return HTTPException(404, str(exc))
    if isinstance(exc, (registry.ModelError, ValueError)):
        return HTTPException(400, str(exc))
    log.exception("Request failed")
    return HTTPException(500, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- request models
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str = "user"
    content: Any = ""

    def as_dict(self) -> dict:
        return {"role": self.role, "content": content_to_text(self.content)}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    think: bool | None = None
    # Server-side grounding for external clients: "zakoni" (hr-zakoni, strict) or
    # "kviz" (hr-znanje). The last user message becomes a grounded single turn.
    rag: str | None = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    prompt: str = ""
    system: str | None = None
    raw: bool = False
    stream: bool = True
    options: dict[str, Any] = Field(default_factory=dict)
    think: bool | None = None


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    device: str | None = None
    dtype: str | None = None


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    think: bool | None = None


# --------------------------------------------------------------------------- helpers
def _now_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _ns(seconds: float) -> int:
    return int(round(seconds * 1_000_000_000))


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _stop_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(s) for s in value if s]
    raise ValueError("'stop' must be a string or a list of strings.")


def _options_to_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """Map Ollama option names onto engine.generate_stream keyword arguments."""
    out: dict[str, Any] = {}
    try:
        for src, dst, conv in (
            ("num_predict", "max_new_tokens", int),
            ("max_new_tokens", "max_new_tokens", int),
            ("max_tokens", "max_new_tokens", int),
            ("temperature", "temperature", float),
            ("top_p", "top_p", float),
            ("top_k", "top_k", int),
            ("repeat_penalty", "repetition_penalty", float),
            ("repetition_penalty", "repetition_penalty", float),
            ("seed", "seed", int),
        ):
            if options.get(src) is not None and dst not in out:
                out[dst] = conv(options[src])
        if "stop" in options:
            out["stop"] = _stop_list(options.get("stop"))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid option value: {e}") from None
    return out


def _template_kwargs(options: dict[str, Any], think: bool | None,
                     reasoning_effort: str | None = None,
                     chat_template_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extra kwargs for apply_chat_template. Call after the model is loaded."""
    tk: dict[str, Any] = {}
    ctk = options.get("chat_template_kwargs") if isinstance(options.get("chat_template_kwargs"), dict) else None
    if isinstance(chat_template_kwargs, dict):
        tk.update(chat_template_kwargs)
    if ctk:
        tk.update(ctk)
    effort = options.get("reasoning_effort") or reasoning_effort
    if effort:
        tk["reasoning_effort"] = str(effort)
    if think is not None and engine.template_supports("enable_thinking"):
        tk["enable_thinking"] = bool(think)
    return tk


def _ollama_stats(st: GenerationStats, load_seconds: float) -> dict:
    return {
        "total_duration": _ns(st.total_seconds + load_seconds),
        "load_duration": _ns(load_seconds),
        "prompt_eval_count": st.prompt_tokens,
        "prompt_eval_duration": _ns(st.prompt_seconds),
        "eval_count": st.completion_tokens,
        "eval_duration": _ns(st.gen_seconds),
    }


def _openai_finish(reason: str) -> str:
    return "length" if reason == "length" else "stop"


def _iso_to_unix(value: str | None) -> int:
    if value:
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            pass
    return int(time.time())


# --------------------------------------------------------------------------- model resolution
def _resolve(ref: str | None) -> registry.LocalModel:
    """Resolve a model reference; without one fall back to the loaded / only installed model."""
    if ref is None or not str(ref).strip():
        if engine.model_name:
            ref = engine.model_name
        else:
            models = registry.list_models()
            if len(models) == 1:
                return models[0]
            raise registry.ModelError(
                "No model specified and none loaded; pass \"model\" (see GET /api/tags)."
                if models else "No models installed. Run: hfserve pull <org/name>")
    return registry.find_model(str(ref))


async def ensure_loaded(ref: str | None, device: str | None = None, dtype: str | None = None,
                        ) -> tuple[registry.LocalModel, float]:
    """Resolve *ref* and load it if it is not the current model.

    Returns ``(local, load_seconds)`` where load_seconds is 0.0 unless this call loaded it.
    """
    try:
        local = await run_in_threadpool(_resolve, ref)
    except Exception as e:
        raise _as_http_error(e) from None
    if engine.model_name == local.repo_id:
        return local, 0.0
    lock = _load_lock or asyncio.Lock()
    async with lock:
        if engine.model_name == local.repo_id:
            return local, 0.0
        try:
            info = await run_in_threadpool(engine.load, local, device, dtype)
        except Exception as e:
            raise _as_http_error(e) from None
        return local, float(info.get("load_seconds") or 0.0)


# --------------------------------------------------------------------------- generation plumbing
async def _stream_generation(gen_kwargs: dict[str, Any], cancel: threading.Event,
                             ) -> AsyncIterator[tuple[str, str] | GenerationStats]:
    """Drive engine.generate_stream in a worker thread and yield its items on the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def push(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # event loop already closed (shutdown)
            pass

    def worker() -> None:
        try:
            for item in engine.generate_stream(cancel=cancel, **gen_kwargs):
                push(item)
        except BaseException as e:  # noqa: BLE001 - forwarded to the response
            push(e)
        finally:
            push(done)

    _track(cancel)
    threading.Thread(target=worker, name="hfserve-stream", daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()
        _untrack(cancel)


async def _blocking_generation(request: Request, gen_kwargs: dict[str, Any]) -> GenerationStats:
    """Run a non-streaming generation in the threadpool; cancel it if the client goes away."""
    cancel = threading.Event()
    _track(cancel)
    try:
        task = asyncio.ensure_future(run_in_threadpool(engine.generate, cancel=cancel, **gen_kwargs))
        while True:
            finished, _ = await asyncio.wait({task}, timeout=0.5)
            if finished:
                break
            if await request.is_disconnected():
                cancel.set()
        try:
            return task.result()
        except Exception as e:
            raise _as_http_error(e) from None
    finally:
        cancel.set()
        _untrack(cancel)


# --------------------------------------------------------------------------- service endpoints
@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return "hfserve is running"


@app.get("/api/health")
async def api_health() -> dict:
    return {
        "app": config.APP_NAME,
        "version": hfserve.__version__,
        "status": "ok",
        "model": engine.model_name,
        "device": engine.device if engine.is_loaded else None,
        "pid": os.getpid(),
    }


@app.get("/api/version")
async def api_version() -> dict:
    return {"version": hfserve.__version__}


@app.get("/api/tags")
async def api_tags() -> dict:
    models = await run_in_threadpool(registry.list_models)
    return {"models": [m.to_api() for m in models]}


@app.get("/api/ps")
async def api_ps() -> dict:
    if not engine.is_loaded:
        return {"models": []}
    info = engine.info()
    return {"models": [{
        "name": info["model"], "model": info["model"], "device": info["device"], "dtype": info["dtype"],
        "vram_bytes": info["vram_bytes"], "loaded_at": info["loaded_at"], "load_seconds": info["load_seconds"],
    }]}


@app.post("/api/load")
async def api_load(req: LoadRequest) -> dict:
    await ensure_loaded(req.model, req.device, req.dtype)
    return engine.info()


@app.post("/api/unload")
async def api_unload() -> dict:
    name = await run_in_threadpool(engine.unload)
    return {"status": "unloaded", "model": name}


@app.post("/api/shutdown")
async def api_shutdown() -> dict:
    threading.Thread(target=_shutdown_worker, name="hfserve-shutdown", daemon=True).start()
    return {"status": "shutting down"}


def _shutdown_worker() -> None:
    time.sleep(0.3)  # let the response leave the socket
    if uvicorn_server is not None:
        uvicorn_server.should_exit = True
    _exit_process(0, unload=False)


_SERVER_PORT: int | None = None  # port this process serves on; selects the pid file


def _exit_process(code: int = 0, unload: bool = True) -> None:
    """Terminate the process for good: cancel generations, free the model, kill children."""
    global _exiting
    with _exit_guard:
        if _exiting:
            return
        _exiting = True
    cancel_all_generations()
    if unload and engine.is_loaded:
        t = threading.Thread(target=_safe_unload, name="hfserve-unload", daemon=True)
        t.start()
        t.join(3.0)
    try:
        process.kill_children_of_self()
    except Exception:
        pass
    process.clear_pid(port=_SERVER_PORT)
    print("hfserve server stopped.")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _safe_unload() -> None:
    try:
        engine.unload()
    except Exception:
        pass


# --------------------------------------------------------------------------- Ollama chat / generate
def _rag_messages(kind: str, messages: list[Message]) -> tuple[list[dict], dict]:
    """Grounded single-turn messages for the last user message (runs in a thread)."""
    question = next((m.as_dict()["content"] for m in reversed(messages) if m.role == "user"), "")
    if not question:
        raise HTTPException(status_code=400, detail="rag needs at least one user message")
    if kind == "zakoni":
        from hfzakon import retrieve as rag

        built, used = rag.build_messages(question)
        sources = [h.meta.get("url") or h.meta.get("law") or "?" for h in used]
        sims = [round(h.similarity, 3) for h in used]
    elif kind == "kviz":
        from hfrag import retrieve as rag

        built, used = rag.build_messages(question)
        sources = [h.source for h in used]
        sims = [round(h.similarity, 3) for h in used]
    else:
        raise HTTPException(status_code=400,
                            detail=f"unknown rag collection {kind!r} (use 'zakoni' or 'kviz')")
    return built, {"collection": kind, "chunks": len(used), "sources": sources, "similarity": sims}


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    local, load_seconds = await ensure_loaded(req.model)
    model_name = local.repo_id
    if not req.messages:  # Ollama: an empty request only loads the model
        return {"model": model_name, "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "load"}
    rag_info: dict | None = None
    if req.rag:
        try:
            grounded, rag_info = await run_in_threadpool(_rag_messages, req.rag, req.messages)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - missing index/package -> clear client error
            raise HTTPException(status_code=500, detail=f"RAG unavailable: {type(e).__name__}: {e}")
        req = req.model_copy(update={"messages": [Message(**m) for m in grounded]})
    gen_kwargs = _options_to_kwargs(req.options)
    gen_kwargs["messages"] = [m.as_dict() for m in req.messages]
    gen_kwargs["template_kwargs"] = _template_kwargs(req.options, req.think)
    gen_kwargs["think"] = req.think

    if not req.stream:
        st = await _blocking_generation(request, gen_kwargs)
        message: dict[str, Any] = {"role": "assistant", "content": st.text}
        if st.thinking:
            message["thinking"] = st.thinking
        reply = {"model": model_name, "created_at": _now_iso(), "message": message, "done": True,
                 "done_reason": st.finish_reason, **_ollama_stats(st, load_seconds)}
        if rag_info is not None:
            reply["rag"] = rag_info
        return reply

    async def body() -> AsyncIterator[str]:
        cancel = threading.Event()
        try:
            async for item in _stream_generation(gen_kwargs, cancel):
                if isinstance(item, GenerationStats):
                    final = {"model": model_name, "created_at": _now_iso(),
                             "message": {"role": "assistant", "content": ""}, "done": True,
                             "done_reason": item.finish_reason, **_ollama_stats(item, load_seconds)}
                    if rag_info is not None:
                        final["rag"] = rag_info
                    yield _ndjson(final)
                else:
                    kind, delta = item
                    msg = {"role": "assistant", "content": ""}
                    if kind == "thinking":
                        msg["thinking"] = delta
                    else:
                        msg["content"] = delta
                    yield _ndjson({"model": model_name, "created_at": _now_iso(), "message": msg, "done": False})
        except Exception as e:  # headers are already out; report inline
            log.exception("chat stream failed")
            yield _ndjson({"error": f"{type(e).__name__}: {e}"})
        finally:
            cancel.set()

    return StreamingResponse(body(), media_type="application/x-ndjson", headers=_STREAM_HEADERS)


@app.post("/api/generate")
async def api_generate(req: GenerateRequest, request: Request):
    local, load_seconds = await ensure_loaded(req.model)
    model_name = local.repo_id
    if not req.prompt and not req.raw:
        return {"model": model_name, "created_at": _now_iso(), "response": "", "done": True, "done_reason": "load"}
    gen_kwargs = _options_to_kwargs(req.options)
    gen_kwargs.update(prompt=req.prompt, system=req.system or None, raw=req.raw)
    gen_kwargs["template_kwargs"] = _template_kwargs(req.options, req.think)
    gen_kwargs["think"] = req.think

    if not req.stream:
        st = await _blocking_generation(request, gen_kwargs)
        out: dict[str, Any] = {"model": model_name, "created_at": _now_iso(), "response": st.text}
        if st.thinking:
            out["thinking"] = st.thinking
        out.update(done=True, done_reason=st.finish_reason, **_ollama_stats(st, load_seconds))
        return out

    async def body() -> AsyncIterator[str]:
        cancel = threading.Event()
        try:
            async for item in _stream_generation(gen_kwargs, cancel):
                if isinstance(item, GenerationStats):
                    yield _ndjson({"model": model_name, "created_at": _now_iso(), "response": "", "done": True,
                                   "done_reason": item.finish_reason, **_ollama_stats(item, load_seconds)})
                else:
                    kind, delta = item
                    chunk: dict[str, Any] = {"model": model_name, "created_at": _now_iso(), "response": ""}
                    if kind == "thinking":
                        chunk["thinking"] = delta
                    else:
                        chunk["response"] = delta
                    chunk["done"] = False
                    yield _ndjson(chunk)
        except Exception as e:
            log.exception("generate stream failed")
            yield _ndjson({"error": f"{type(e).__name__}: {e}"})
        finally:
            cancel.set()

    return StreamingResponse(body(), media_type="application/x-ndjson", headers=_STREAM_HEADERS)


# --------------------------------------------------------------------------- OpenAI-compatible
@app.get("/v1/models")
async def v1_models() -> dict:
    models = await run_in_threadpool(registry.list_models)
    return {"object": "list", "data": [{
        "id": m.repo_id, "object": "model", "created": _iso_to_unix(m.downloaded_at),
        "owned_by": m.repo_id.split("/")[0] if "/" in m.repo_id else "local",
    } for m in models]}


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OpenAIChatRequest, request: Request):
    local, _ = await ensure_loaded(req.model)
    model_name = local.repo_id
    if not req.messages:
        raise HTTPException(400, "'messages' must not be empty.")
    options: dict[str, Any] = {
        "temperature": req.temperature, "top_p": req.top_p, "top_k": req.top_k, "seed": req.seed,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_tokens if req.max_tokens is not None else req.max_completion_tokens,
    }
    if req.stop is not None:
        options["stop"] = req.stop
    gen_kwargs = _options_to_kwargs(options)
    gen_kwargs["messages"] = [m.as_dict() for m in req.messages]
    gen_kwargs["template_kwargs"] = _template_kwargs({}, req.think, req.reasoning_effort, req.chat_template_kwargs)
    gen_kwargs["think"] = req.think
    cmpl_id = "chatcmpl-" + secrets.token_hex(12)
    created = int(time.time())

    if not req.stream:
        st = await _blocking_generation(request, gen_kwargs)
        message: dict[str, Any] = {"role": "assistant", "content": st.text}
        if st.thinking:
            message["reasoning_content"] = st.thinking
        return {
            "id": cmpl_id, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": _openai_finish(st.finish_reason)}],
            "usage": {"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens,
                      "total_tokens": st.prompt_tokens + st.completion_tokens},
        }

    async def body() -> AsyncIterator[str]:
        cancel = threading.Event()
        first = True
        base = {"id": cmpl_id, "object": "chat.completion.chunk", "created": created, "model": model_name}
        try:
            async for item in _stream_generation(gen_kwargs, cancel):
                delta: dict[str, Any] = {"role": "assistant"} if first else {}
                first = False
                if isinstance(item, GenerationStats):
                    yield _sse({**base, "choices": [{"index": 0, "delta": delta,
                                                     "finish_reason": _openai_finish(item.finish_reason)}],
                                "usage": {"prompt_tokens": item.prompt_tokens,
                                          "completion_tokens": item.completion_tokens,
                                          "total_tokens": item.prompt_tokens + item.completion_tokens}})
                else:
                    kind, text = item
                    delta["reasoning_content" if kind == "thinking" else "content"] = text
                    yield _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.exception("v1 stream failed")
            yield _sse({"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
            yield "data: [DONE]\n\n"
        finally:
            cancel.set()

    return StreamingResponse(body(), media_type="text/event-stream", headers=_STREAM_HEADERS)


# --------------------------------------------------------------------------- run
def _setup_logging(log_level: str) -> None:
    level = {"trace": logging.DEBUG}.get(log_level.lower(), getattr(logging, log_level.upper(), logging.INFO))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    for noisy in ("httpx", "httpcore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _port_available(host: str, port: int) -> bool:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        family, _, _, _, addr = infos[0]
    except socket.gaierror:
        family, addr = socket.AF_INET, (host, port)
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.bind(addr)
        return True
    except OSError:
        return False


def _fmt_vram(nbytes: int) -> str:
    return registry.human_size(nbytes)


def run_server(host: str, port: int, model: str | None = None, log_level: str = "info") -> None:
    """Start the API server on host:port and block until it stops. Never returns normally."""
    global uvicorn_server
    import uvicorn

    config.ensure_dirs()
    _setup_logging(log_level)
    if not _port_available(host, port):
        print(f"Error: port {port} on {host} is already in use (another hfserve server?). "
              f"Stop it with 'hfserve stop' or choose another --port.", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    global _SERVER_PORT
    _SERVER_PORT = port
    process.write_pid(port=port)

    url = config.server_url(host, port)
    models = registry.list_models()
    names = ", ".join(m.repo_id for m in models[:5]) + (", ..." if len(models) > 5 else "")
    print(f"hfserve {hfserve.__version__}  (pid {os.getpid()})")
    print(f"Listening on {url}")
    print(f"  Ollama-style chat API : POST {url}/api/chat")
    print(f"  OpenAI-compatible API : POST {url}/v1/chat/completions")
    print(f"  Models directory      : {config.MODELS_DIR}")
    print(f"  Installed models      : {len(models)}" + (f"  ({names})" if names else ""))
    sys.stdout.flush()

    code = 0
    try:
        if model:
            try:
                local = registry.find_model(model, models)
                print(f"Loading {local.repo_id} ...")
                sys.stdout.flush()
                info = engine.load(local)
                print(f"Loaded on {info['device']} ({info['dtype']}) in {info['load_seconds']:.1f}s, "
                      f"VRAM {_fmt_vram(info['vram_bytes'])}")
            except registry.ModelError as e:
                print(f"Error: {e}", file=sys.stderr)
                code = 1
                return
            except Exception as e:
                print(f"Error: failed to load {model}: {type(e).__name__}: {e}", file=sys.stderr)
                code = 1
                return
        print("Press Ctrl+C to stop.")
        sys.stdout.flush()

        class _Server(uvicorn.Server):
            def handle_exit(self, sig, frame):  # Ctrl+C: stop running generations right away
                cancel_all_generations()
                super().handle_exit(sig, frame)

        cfg = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=False, log_config=None)
        server = _Server(cfg)
        uvicorn_server = server
        try:
            server.run()
        except KeyboardInterrupt:
            pass
        except SystemExit as e:  # uvicorn exits this way when it cannot bind
            code = int(e.code or 0) if isinstance(e.code, int) else 1
            if code and not _port_available(host, port):
                print(f"Error: port {port} on {host} is already in use.", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.exception("server crashed")
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        code = 1
    finally:
        _exit_process(code)
