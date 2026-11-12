# Using hfserve from other projects

hfserve is a local, Ollama-style model server for HuggingFace transformers models.
It serves **EuroLLM-1.7B-Instruct** (and any other installed model) over HTTP, with an
optional strict Croatian-legislation RAG layer. This guide covers two ways to use it:

1. **Headless — from another project** (like you would use Ollama): server + model only, no GUI.
2. **As a chat bot** with the built-in GUI.

The full HTTP contract (every endpoint, every field) is in [docs/API.md](docs/API.md).

---

## The port

The server always listens on **`http://127.0.0.1:11435`**.

Port **11435** is the fixed contract for this machine — it was chosen so that a real
Ollama install keeps its own default **11434** untouched; the two can run side by side.
Override only if you must, with `--port` or the `HFSERVE_PORT` environment variable,
and then every client has to be told about it.

---

## 1. Headless use from another project

### Start the server (no GUI)

From a terminal, or spawned by your project:

```bat
cd /d D:\HuggingFaceServerModels
.vemv\Scripts\python.exe -m hfserve serve --model utter-project/EuroLLM-1.7B-Instruct
```

- `serve` runs the API server in the **foreground** (log also goes to `state\server.log`).
- `--model` preloads the model at start-up, so the first request does not pay the load time.
  Without it, the first `/api/chat` request loads the model automatically (Ollama behavior).
- `hfserve.cmd serve --model ...` is the same thing via the bundled launcher.

From another program, spawn it as a child process, e.g. in Python:

```python
import subprocess
subprocess.Popen(
    [r"D:\HuggingFaceServerModels\.vemv\Scripts\python.exe", "-m", "hfserve",
     "serve", "--model", "utter-project/EuroLLM-1.7B-Instruct"],
    cwd=r"D:\HuggingFaceServerModels")
```

### Check / stop

```bat
.vemv\Scripts\python.exe -m hfserve ps     &rem who is running, which model, VRAM
.vemv\Scripts\python.exe -m hfserve stop   &rem graceful stop (kills stragglers if needed)
```

or over HTTP: `GET /api/health` (ready check), `POST /api/shutdown` (the process exits
within ~1 s). Wait for `/api/health` to return 200 before sending the first request.

### Talk to it — Ollama-style API

`POST /api/chat` with the same shape Ollama uses:

```bash
curl http://127.0.0.1:11435/api/chat -d "{
  \"model\": \"utter-project/EuroLLM-1.7B-Instruct\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Pozdrav! Tko si ti?\"}],
  \"stream\": false,
  \"options\": {\"temperature\": 0.3, \"num_predict\": 256, \"repeat_penalty\": 1.1}
}"
```

Python (any HTTP library works; no SDK needed):

```python
import httpx

r = httpx.post("http://127.0.0.1:11435/api/chat", json={
    "model": "utter-project/EuroLLM-1.7B-Instruct",
    "messages": [{"role": "user", "content": "Pozdrav!"}],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 256, "repeat_penalty": 1.1},
}, timeout=300)
print(r.json()["message"]["content"])
```

The official **Ollama Python client** also works — point it at this port:

```python
from ollama import Client
client = Client(host="http://127.0.0.1:11435")
r = client.chat(model="utter-project/EuroLLM-1.7B-Instruct",
                messages=[{"role": "user", "content": "Pozdrav!"}])
```

With `"stream": true` (the default) you get NDJSON lines exactly like Ollama:
`{"message": {"content": "<delta>"}, "done": false}` … final line with `"done": true`
and timing stats. `GET /api/tags` lists installed models, Ollama-style.

### Talk to it — OpenAI-compatible API

`POST /v1/chat/completions` works with any OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="utter-project/EuroLLM-1.7B-Instruct",
    messages=[{"role": "user", "content": "Pozdrav!"}])
print(r.choices[0].message.content)
```

### Croatian legislation answers over HTTP (strict RAG)

Add `"rag": "zakoni"` to an `/api/chat` request and the server grounds the answer in
the local legislation index (all consolidated laws from zakon.hr + Narodne novine
2025→today) **before** the model sees the question:

```python
r = httpx.post("http://127.0.0.1:11435/api/chat", json={
    "model": "utter-project/EuroLLM-1.7B-Instruct",
    "messages": [{"role": "user", "content": "Što kaže članak 5. Zakona o naplati cestarine?"}],
    "stream": False, "rag": "zakoni",
    "options": {"temperature": 0.2, "num_predict": 220, "repeat_penalty": 1.1},
}, timeout=300)
d = r.json()
print(d["message"]["content"])   # grounded answer, or "Ne znam."
print(d["rag"])                  # {"collection": "zakoni", "chunks": 2, "sources": [...], ...}
```

Rules of the `zakoni` collection: the answer comes **only** from retrieved law text and
cites its source; when nothing confident is found the model says **"Ne znam."** instead
of guessing. The `rag` block in the response tells you which sources grounded the
answer and how similar they were — log it if your project needs an audit trail.
`"rag": "kviz"` selects the quiz knowledge base instead. Without a `rag` field the
request is a plain chat — **do not ask the bare 1.7B model Croatian legal questions
without `"rag": "zakoni"`**; ungrounded answers are guesses.

Keep the legislation index fresh with `python -m hfzakon refresh` (incremental,
usually seconds to minutes; the GUI does this automatically on start).

### Generation settings that work well for EuroLLM-1.7B

- `temperature` 0.2–0.6 (lower for factual/RAG answers), `repeat_penalty` 1.1
  (small models loop without it), `num_predict` sized to the expected answer.
- The model occasionally appends a stray "Ne znam." after a correct grounded answer;
  trim it client-side if it bothers your UI, or lower `num_predict`.
- One model is loaded at a time (8 GB VRAM); requesting a different `model` unloads
  the current one first. Concurrent requests queue — they do not fail.

---

## 2. Chat bot with the GUI

For a human chatting with the model, start the window instead:

```bat
HFServe-Chat.cmd           &rem double-click works too
rem  or: .vemv\Scripts\python.exe -m hfserve gui
```

What the GUI does:

- **Starts the server itself** when none is running (and stops it again on close —
  only when it was the one that started it; a server you started stays up).
- Pick the model, set temperature / max tokens / repeat penalty, chat. `New chat`
  clears the history.
- **"Zakoni (RAG)"** checkbox: strict legislation answers with citations (same engine
  as `"rag": "zakoni"` above). **"Kviz baza (RAG)"**: quiz knowledge base. When both
  are checked, Zakoni wins.
- **Automatic legislation refresh**: on start, the GUI checks (in the background)
  whether new Narodne novine editions or new laws exist. If yes, a progress row with
  a time estimate appears above the chat while it downloads and indexes them — chat
  stays usable; if you ask a Zakoni question mid-refresh, the transcript notes that
  the newest editions may not be included yet. Skipped when refreshed < 24 h ago.
- Console alternative: `.vemv\Scripts\python.exe -m hfserve run EuroLLM-1.7B-Instruct`
  (chat in the terminal; `\bye` ends the chat and stops the server it started).

The GUI and any external project can use the **same running server** at the same time —
they are just HTTP clients of port 11435.

---

## Managing models

```bat
.vemv\Scripts\python.exe -m hfserve list                 &rem installed models
.vemv\Scripts\python.exe -m hfserve pull <HF URL or org/name>
.vemv\Scripts\python.exe -m hfserve show <model>
.vemv\Scripts\python.exe -m hfserve rm <model>
```

Models live in `models\<org>\<name>\`. Anything plain-transformers from
huggingface.co can be pulled; EuroLLM-1.7B-Instruct is the working default.

## Environment overrides

| Variable | Meaning | Default |
|---|---|---|
| `HFSERVE_PORT` | server port | `11435` |
| `HFSERVE_HOST` | bind address | `127.0.0.1` (localhost only) |
| `HFSERVE_HOME` | root for models and state | project folder |
| `HFSERVE_DEVICE` / `HFSERVE_DTYPE` | force `cuda`/`cpu`, `float16`/… | auto (fp16 on this GPU) |

The server binds to localhost only by default — other machines cannot reach it unless
you set `HFSERVE_HOST` to a LAN address, which also exposes it to that network.
Server log: `state\server.log`.
