# hfserve HTTP API

The server (`hfserve serve`) listens on `http://127.0.0.1:11435` by default
(`HFSERVE_HOST` / `HFSERVE_PORT`, or `--host` / `--port`). All bodies are JSON.
Every error response is `{"error": "<human readable message>"}` with a 4xx/5xx status.
An error that happens after a stream has started arrives inline as a final
`{"error": "..."}` line (NDJSON) or `data: {"error": {"message": "..."}}` event (SSE).

Model references (`"model"` fields) accept anything `registry.find_model` accepts:
`org/name`, a huggingface.co URL, the bare `name` when unique, or the 1-based index
from `hfserve list`.

## Service

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | `text/plain` → `hfserve is running` |
| GET | `/api/health` | `{"app":"hfserve","version":"0.1.0","status":"ok","model":"<repo_id>"|null,"device":"cuda"|"cpu"|null,"pid":1234}` |
| GET | `/api/tags` | `{"models":[ LocalModel.to_api(), ... ]}` (Ollama-compatible list of installed models) |
| GET | `/api/ps` | `{"models":[{"name","model","device","dtype","vram_bytes","loaded_at","load_seconds"}]}` — empty list if nothing loaded |
| POST | `/api/load` | body `{"model":"<ref>","device"?: "cuda"|"cpu","dtype"?: "float16"|"bfloat16"|"float32"}` → `{"model","device","dtype","load_seconds","vram_bytes","needs_remote_code","max_context"}`; 404 if not installed, 400 if ambiguous, 500 on load failure. Loading a model that is already loaded is a no-op (returns same info). Loading a different model unloads the previous one first. |
| POST | `/api/unload` | `{"status":"unloaded","model":"<repo_id>"|null}` |
| POST | `/api/shutdown` | `{"status":"shutting down"}` then the process exits within ~1 s: cancels any running generation, kills child processes, clears the pid file, `os._exit(0)`. |

## Chat (Ollama-compatible)

`POST /api/chat`

```json
{
  "model": "utter-project/EuroLLM-1.7B-Instruct",
  "messages": [{"role": "system|user|assistant", "content": "..."}],
  "stream": true,
  "think": true,
  "options": {
    "temperature": 0.6, "top_p": 0.95, "top_k": 0,
    "num_predict": 4096, "repeat_penalty": 1.0,
    "stop": ["..."], "seed": 42,
    "reasoning_effort": "high",
    "chat_template_kwargs": {}
  }
}
```

Option defaults live in `config.py` (`top_k` 0 and `repeat_penalty` 1.0 mean "off").
`reasoning_effort` (`high`/`medium`/`low`) and `chat_template_kwargs` are passed to the
tokenizer's chat template; templates that do not use them (such as EuroLLM's) ignore them.
`think: false` maps to the template's `enable_thinking=False` when the template supports it
(Qwen3 style). For templates without that switch whose generation prompt ends with an
opening think tag, the engine appends the matching closing tag to the prompt so the model
answers without reasoning.

* If the requested model is not the loaded one it is loaded first (like Ollama).
* Generations are serialised with a lock; concurrent requests wait their turn.
* `stream: true` (default) → `application/x-ndjson`, one JSON object per line:

```json
{"model":"utter-project/EuroLLM-1.7B-Instruct","created_at":"2026-09-12T06:00:00.000Z","message":{"role":"assistant","content":"","thinking":"The user wants"},"done":false}
{"model":"utter-project/EuroLLM-1.7B-Instruct","created_at":"...","message":{"role":"assistant","content":"","thinking":" a greeting."},"done":false}
{"model":"utter-project/EuroLLM-1.7B-Instruct","created_at":"...","message":{"role":"assistant","content":"Hel"},"done":false}
{"model":"utter-project/EuroLLM-1.7B-Instruct","created_at":"...","message":{"role":"assistant","content":"lo!"},"done":false}
{"model":"utter-project/EuroLLM-1.7B-Instruct","created_at":"...","message":{"role":"assistant","content":""},"done":true,
 "done_reason":"stop","total_duration":123456789,"load_duration":0,
 "prompt_eval_count":21,"prompt_eval_duration":1000000,"eval_count":42,"eval_duration":2000000000}
```

Durations are nanoseconds (Ollama convention). `done_reason` is `stop` (eos or
stop string), `length` (hit `num_predict`), or `cancel` (client disconnected / shutdown).

* Reasoning models: text the model produces inside its think block is streamed as
  `message.thinking` deltas (with `content` empty); the answer follows as `message.content`
  deltas. The engine detects the think tags from the chat template (Qwen3/DeepSeek
  `<think>`…`</think>`). Models without a think block (such as EuroLLM) stream only `content`.
* `stream: false` → one JSON object of the final shape, with the full answer in
  `message.content` and the reasoning in `message.thinking` (omitted when empty).
* If the client disconnects mid-stream the generation is cancelled immediately.

### Server-side RAG (`"rag"` field)

`"rag": "zakoni"` (Croatian legislation, strict) or `"rag": "kviz"` (quiz knowledge)
grounds the answer server-side: the **last user message** is used as the question, the
matching chunks are retrieved from the local Chroma index and the conversation is
replaced by a grounded single turn (any system/history you sent is ignored for that
request). With `"zakoni"`, answers come only from retrieved law text with a source
citation; without a confident hit the model answers `Ne znam.` The final response
object (stream and non-stream) carries a `rag` block:

```json
"rag": {"collection": "zakoni", "chunks": 2,
        "sources": ["https://www.zakon.hr/z/4002/zakon-o-naplati-cestarine"],
        "similarity": [0.92]}
```

Errors: unknown collection → 400; index/package unavailable → 500 with a clear
message. Requires the RAG index under `rag/` (see `hfzakon`/`hfrag`).

`POST /api/generate` — same as chat but with `"prompt": "..."`, optional
`"system": "..."`, optional `"raw": true` (send prompt verbatim, no chat template).
Chunks use `"response"` (answer) and `"thinking"` (reasoning) instead of `"message"`:
`{"model","created_at","response":"<delta>","done":false}` … final `{"response":"","done":true,"done_reason":...,stats}`.

## OpenAI-compatible

`POST /v1/chat/completions` with `{"model","messages","stream","temperature","top_p","top_k","max_tokens"` (or `max_completion_tokens`)`,"repetition_penalty","stop","seed","reasoning_effort","chat_template_kwargs","think"}`.
The last streamed chunk also carries a `usage` object.
Reasoning is returned as `reasoning_content` (streaming: `delta.reasoning_content`;
non-streaming: `message.reasoning_content`), the vLLM/DeepSeek convention.

* non-stream → `{"id":"chatcmpl-<hex>","object":"chat.completion","created":<unix>,"model":..,"choices":[{"index":0,"message":{"role":"assistant","content":..},"finish_reason":"stop"|"length"}],"usage":{"prompt_tokens","completion_tokens","total_tokens"}}`
* stream → `text/event-stream`; each event is `data: {"id","object":"chat.completion.chunk","created","model","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}\n\n`.
  The first chunk's delta also carries `"role":"assistant"`. The last chunk has an empty delta and `finish_reason` set, followed by `data: [DONE]\n\n`.

`GET /v1/models` → `{"object":"list","data":[{"id":"<repo_id>","object":"model","created":<unix>,"owned_by":"<org>"}]}`

## Engine behaviour

* Device: `HFSERVE_DEVICE` or `cuda` if available else `cpu`.
* Dtype: `HFSERVE_DTYPE` or, on CUDA, `bfloat16` when natively supported else `float16`; on CPU `float32`.
* `trust_remote_code=True` is passed only when the model's `config.json` has an `auto_map` (custom code); the server logs a notice when it does.
* Prompts are built with the tokenizer chat template (`apply_chat_template(..., add_generation_prompt=True)`); models without a template get a plain `System:/User:/Assistant:` fallback.
* If the prompt would not fit in the context window (`max_position_embeddings`) the oldest non-system messages are dropped, and `num_predict` is capped to the remaining room.
* `temperature <= 0` means greedy decoding.
* End of turn: besides the model's `eos_token_id`, the engine detects the chat template's
  end-of-turn token (for EuroLLM `<|im_end|>`, id 4) and stops on it too.
* Message `content` given as a list of OpenAI-style parts is flattened to text.
