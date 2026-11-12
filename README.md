# hfserve

Serve HuggingFace models on your own machine, Ollama style. Paste a model URL such as
<https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct>, download it, start a local
API server, and chat with the model from the console. `\bye` stops the server and kills
every process the app started.

```
$ hfserve run https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct
Starting server ...
Serving utter-project/EuroLLM-1.7B-Instruct at http://127.0.0.1:11435
  API: POST /api/chat  ·  POST /v1/chat/completions  ·  GET /api/tags
Loading model ... done: cuda / float16 in 3.4s (VRAM 3.3 GB)
Chatting with utter-project/EuroLLM-1.7B-Instruct  (server http://127.0.0.1:11435)
Type a message and press Enter.  \bye stops the server and exits.  \help for commands.
>>> hello
Hello! How can I help you today?
>>> \bye
Stopping server ...
Server stopped (graceful shutdown); no hfserve processes left running. Bye.
```

## Install

Requires Python 3.10+ and, for GPU inference, an NVIDIA driver. Tested on Windows 10
with Python 3.12 and an RTX 2070 SUPER (8 GB).

```powershell
cd E:\HuggingFaceServerModels
python -m venv .vemv
.vemv\Scripts\activate
# PyTorch with CUDA (pick the index that matches your driver; cu128 works with any 12.8+ driver)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Optional: `pip install -e .` registers an `hfserve` command in the venv. Otherwise use
`hfserve.cmd` (Windows) / `./hfserve.sh` (Linux, macOS) from the project folder, or
`python -m hfserve`.

## Commands

| Command | What it does |
|---|---|
| `hfserve` | Interactive shell: shows installed models and the server address, accepts the commands below |
| `hfserve list` | List downloaded models (with an index you can use instead of the name) |
| `hfserve pull <url or org/name>` | Download a model from huggingface.co (resumable) |
| `hfserve run <model>` | Start the server, load the model, and chat in the console. Pulls the model first if it is not installed |
| `hfserve gui [--model M]` | Ask which installed model to use (and offer to download another from huggingface.co), then open a chat window — starts the server if needed; `HFServe-Chat.cmd` does the same without a console. `--model` skips the question |
| `hfserve serve [--model M]` | Run only the API server in the foreground (like `ollama serve`) |
| `hfserve stop [--all]` | Stop the server on the current port (`--all`: every hfserve server on the machine) and kill every process it started |
| `hfserve ps` | Show what the server has loaded |
| `hfserve show <model>` | Details and files of an installed model |
| `hfserve rm <model>` | Delete a downloaded model |

A `<model>` may be `utter-project/EuroLLM-1.7B-Instruct`, the bare name
`EuroLLM-1.7B-Instruct` when it is unique, the number printed by `hfserve list`, or a
full huggingface.co URL.

`hfserve list --json` prints the model index; the same data is kept in `models/models.json`,
rewritten on every pull, remove, and list.

## Windows starter script

`Start-HFServe.ps1` wraps the commands above in a menu and reads the model list from
`models/models.json`:

```powershell
.\Start-HFServe.ps1                       # menu: start/stop server, chat, switch model, download, list
.\Start-HFServe.ps1 -Start                # serve the first installed model in this window (Ctrl+C stops)
.\Start-HFServe.ps1 -Start -Background    # start it minimized in its own window and return
.\Start-HFServe.ps1 -Start -Model EuroLLM-1.7B-Instruct -Port 11436
.\Start-HFServe.ps1 -Chat                 # console chat with the first model (\bye stops the server)
.\Start-HFServe.ps1 -Pull https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct
.\Start-HFServe.ps1 -List                 # installed models from models.json
.\Start-HFServe.ps1 -Stop
.\Stop-HFServe.ps1                        # clear the model from VRAM, then stop the server
.\Stop-HFServe.ps1 -All                   # same, for every hfserve server on the machine
```

`-Model` takes a name or the list number from `models.json`. While the server is running,
menu option 1 switches the loaded model through the API without a restart. If PowerShell
refuses to run scripts, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Chatting

Inside `hfserve run` the prompt is `>>> `. Commands start with a backslash (a slash works too):

| Command | Effect |
|---|---|
| `\bye` (also `\quit`, `\exit`, Ctrl+C at the prompt, Ctrl+Z+Enter) | Stop the server, kill its processes, and exit |
| `\clear` | Start a new conversation |
| `\system <text>` | Set the system prompt |
| `\set temperature 0.3` | Change a generation option (`temperature`, `top_p`, `top_k`, `num_predict`, `repeat_penalty`, `seed`) |
| `\set think off` | Hide the model's reasoning (shown dimmed by default); also asks models that support it to skip thinking |
| `\set reasoning_effort low` | For reasoning models whose chat template supports it: `high` thinks longest, `low` answers fastest. Ignored by models without it (such as EuroLLM) |
| `\models`, `\load <model>` | List installed models, switch to another one |
| `\stats off` | Hide the tokens-per-second line |
| `"""` | Begin or end a multi-line message |

Ctrl+C while an answer is streaming interrupts that answer only. `\bye` also stops a server
that was started separately with `hfserve serve` and that `run` merely connected to.

## API

The server speaks both the Ollama and the OpenAI chat protocols, so existing clients work:

```bash
curl http://127.0.0.1:11435/api/chat -d '{"model":"utter-project/EuroLLM-1.7B-Instruct","messages":[{"role":"user","content":"hi"}]}'
curl http://127.0.0.1:11435/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"utter-project/EuroLLM-1.7B-Instruct","messages":[{"role":"user","content":"hi"}],"stream":false}'
curl http://127.0.0.1:11435/api/tags
```

Full reference: [docs/API.md](docs/API.md).

## Where things live

| Path | Content |
|---|---|
| `models/<org>/<name>/` | Downloaded model files plus an `.hfserve.json` manifest |
| `.hfserve/server.pid` | PID of the server on the default port (`server-<port>.pid` for other ports) |
| `.hfserve/server.log` | Server output when started by `hfserve run` |
| `.hfserve/last_model.json` | The model loaded last in the chat window, preselected next time |
| `files/in/` | Files uploaded in the chat window (copies; originals are never touched) |
| `files/out/` | Workbooks, documents, decks and PDFs hffiles produced |

Environment variables: `HFSERVE_HOME`, `HFSERVE_MODELS`, `HFSERVE_HOST`, `HFSERVE_PORT`
(default `127.0.0.1:11435`), `HFSERVE_DEVICE` (`cuda`/`cpu`), `HFSERVE_DTYPE`
(`float16`/`bfloat16`/`float32`), `HF_TOKEN` for gated repositories, `HFFILES_HOME` for the file workspace.

A `.env` file in the project root is read for `HFSERVE_HOME` / `HFSERVE_MODELS` / `MODELS_DIR`
(the models folder) and `HFSERVE_HOST` / `HFSERVE_PORT` — or a whole `SERVER_URL` such as
`http://127.0.0.1:11435` — for the address the server listens on and every client connects to:

```
MODELS_DIR=./models
HFSERVE_HOST=127.0.0.1
HFSERVE_PORT=11435
```

So `MODELS_DIR=./models` is the folder the server, the registry, the chat window and
`Start-HFServe.ps1` all load models from. A relative value is relative to the project, not to
the shell's working directory. A real environment variable always wins over `.env`, and only
those keys are read — a secret such as `HF_TOKEN` in `.env` is deliberately ignored and still
has to be a real environment variable. A port that is not a number is reported at start-up and
falls back to 11435 instead of being applied silently; if the models folder does not exist, the
chat window says so and names the setting to fix instead of showing an empty model list.

**Freeing the GPU.** `Stop-HFServe.ps1` (or `Stop-HFServe.cmd`) unloads the model through
`POST /api/unload` *before* stopping the server, and prints VRAM in use before and after.
`Start-HFServe.ps1` also clears whatever a previous session left loaded when it starts, and
clears it again when it exits — however it exits, including Ctrl+C. Nothing is lost by this:
the next request reloads the model by itself, since every `/api/chat` goes through
`ensure_loaded`.

## Office files and PDF

The chat window has a **Datoteka** row: attach an `.xlsx`, `.docx`, `.pptx` or `.pdf`, ask
for what you want in plain language, and Python does the work. The model only produces a
*plan* — a JSON list of operations such as `sort_rows`, `add_slide` or `keep_pages` — which
`hffiles` validates and then executes with openpyxl, python-docx, python-pptx or
pypdf/reportlab. The model never writes file contents, so totals, dates and slide layout are
as correct as the code that produces them, and the same request twice gives the same file.

**Spremi kao …** downloads the file being worked on, **U PDF** renders it to PDF (through
Microsoft Office or LibreOffice, whichever is installed).

```powershell
python -m hffiles describe prihodi.xlsx                      # what the model gets to see
python -m hffiles ask "dodaj stupac PDV, 25% iznosa" --file prihodi.xlsx
python -m hffiles ask "dodaj slajd s tri tocke" --file prezentacija.pptx
python -m hffiles pdf ponuda.docx                            # export to PDF
python -m hffiles new roba.xlsx --csv roba.csv --sheet Roba   # no model involved
```

Uploads are copied to `files/in/` and results written to `files/out/`; the original file is
never opened for writing, and a plan that fails writes nothing. Details, the operation list
per format and the library limits (openpyxl stores formulas but does not calculate them) are
in [docs/FILES.md](docs/FILES.md).

## Notes

* Models whose `config.json` declares custom code (an `auto_map`) are loaded with
  `trust_remote_code=True`. That runs Python shipped inside the model repo, so only pull
  models from publishers you trust. EuroLLM is a plain Llama-style model with no custom code.
* On GPUs without native bfloat16 (Turing and older) the model runs in float16; on CPU in float32.
* Measured with EuroLLM-1.7B-Instruct on an RTX 2070 SUPER: about 3.3 GB of VRAM, loads in
  about 3.5 s.
* transformers is pinned to 5.14 to 5.x (the tested range).
* Only one model is held in memory at a time. Requesting a different model swaps it in.

## Fine-tuning (Croatian)

`hftrain` is a separate package that fine-tunes an installed model with LoRA adapters and
installs the merged result as a new hfserve model. Question-answer examples for
instruction tuning go into `training/QA/` (see `training/QA/README.txt` for the formats).

```powershell
pip install -r requirements-train.txt
python -m hftrain prepare --name qa-v1                        # from training/QA (the default source)
python -m hftrain train --data qa-v1 --run qa-lora-1 --lr 1e-4
python -m hftrain eval --run qa-lora-1 --compare-base
python -m hftrain export --run qa-lora-1 --name local/EuroLLM-1.7B-hr
python -m hfserve run local/EuroLLM-1.7B-hr
```

Or use the window: `python -m hftrain gui` / `HFTrain-GUI.cmd` — Prepare QA data,
Start/Stop training with a progress bar, and a Chat tab to try the result.

The plan, the data sources and every flag are described in [docs/TRAINING.md](docs/TRAINING.md).

## Asking it maths in a loop

`math_loop.py` keeps the model busy with random English arithmetic and word problems, sending
the next one the moment the previous answer ends. The problems are generated locally, so the
right answer is known and every reply is graded:

```
python math_loop.py                  # until ESC (or q, or Ctrl+C)
python math_loop.py --count 20       # twenty problems, then a score
python math_loop.py --temperature 0  # steadier answers
python math_loop.py --no-stream      # print whole answers instead of streaming them
```

It starts the server if none is running (and stops it again only in that case), unloads the
model from VRAM on the way out unless `--keep-loaded` is given, and takes the address from
`.env` like everything else. A reply counts as correct when any number in it matches the
expected value, because a small model tends to answer in a sentence.

## Development

```
python -m pytest tests -q
```
