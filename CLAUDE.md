# hfserve — project notes

Ollama-style local server for HuggingFace transformers models. See README.md for usage,
docs/API.md for the HTTP contract.

## Layout
- `hfserve/config.py`    paths, ports, env overrides (models live in `models/<org>/<name>/`); reads `.env`
                         for `HFSERVE_HOME`/`HFSERVE_MODELS`/`MODELS_DIR` and `HFSERVE_HOST`/`HFSERVE_PORT`/
                         `HFSERVE_URL`/`SERVER_URL` (real env wins; only `DOTENV_KEYS` are read, so an
                         HF_TOKEN in .env is ignored on purpose). `require_models_dir()` fails loud,
                         `SETTINGS_PROBLEMS` collects unreadable settings instead of raising at import,
                         `read/write_last_model()` remember the GUI's choice
- `hfserve/registry.py`  parse HF URLs/ids, list/find/pull/remove models, `.hfserve.json` manifest;
                         `hub_download_plan()` = what a pull would fetch (size/files/servable), without pulling
- `hfserve/engine.py`    transformers loading + streaming generation (one model at a time, lock)
- `hfserve/server.py`    FastAPI: /api/chat, /api/generate (Ollama), /v1/chat/completions (OpenAI), /api/shutdown;
                         /api/chat takes optional `"rag": "zakoni"|"kviz"` for server-side grounding with a
                         `rag` sources block in the reply. Port contract 11435 (11434 stays Ollama's);
                         integration guide for other projects: API_README.md
- `hfserve/process.py`   spawn/find/stop the server, pid file, process-tree kill, Windows job object
- `hfserve/chat.py`      console chat client (`\bye` stops the server)
- `hfserve/gui.py`       tkinter chat window over the HTTP API (`hfserve gui` / `HFServe-Chat.cmd`); starts the
                         server when none runs and stops it again on close (only when it started it).
                         Opens picker.py first unless `--model` was given; the Model combo box hot-swaps
                         (POST /api/load unloads the old one), and every load is remembered
- `hfserve/picker.py`    the startup "which model?" window: installed models from `config.MODELS_DIR` (the
                         folder from .env), last choice preselected, plus Hub search / "Check fit"
                         (`hub_download_plan` + detected VRAM via nvidia-smi) / Download (`pull_model`,
                         which rewrites models.json so the new model works everywhere)
- `hfserve/cli.py`       argparse commands + interactive shell (`python -m hfserve`)
- `math_loop.py`         console loop that posts random English math problems to the server
                         (`python math_loop.py [--count N]`): problems are generated locally so the
                         expected answer is known and every reply is graded; streams the answer and
                         stops on ESC/q (watcher thread) or Ctrl+C. Unloads the model on exit and
                         stops the server only if it started it
- `graphify-animation/`  "Living Map" web app over `graphify-out/graph.json`: a ~63 s animated tour of
                         what happens when `math_loop.py` runs (12 blocks, each anchored to a real node)
                         plus a 3D explorer. `build_data.py` regenerates `graph-data.js` (needs networkx +
                         numpy); `story.js` holds every text. Rationale lives as separate nodes joined by
                         `rationale_for`, not as a node attribute, so build_data folds it back in.
                         `node smoke_test.mjs` drives it in headless Chrome. See its README.md
- `tests/`               pytest, no network
- `Start-HFServe.ps1`    Windows menu/starter (PS 5.1 compatible); reads `models/models.json`,
                         which registry.write_index() rewrites on pull/rm and `hfserve list [--json]`.
                         Resolves the models folder like config.py does (env, then `.env`), and warns
                         when it is missing. `Clear-ModelMemory` (POST /api/unload) runs before any
                         action and again in a `finally`, so no run leaves weights in VRAM
- `Stop-HFServe.ps1`     unload the model first, then `hfserve stop` (`-All` for every server);
                         prints nvidia-smi VRAM before/after. Host/port come from `hfserve.config`
                         (asked via python), never parsed a second time. `Stop-HFServe.cmd` wraps it
- `hfrag/`               local RAG over Croatian knowledge: Chroma index under `rag/index` (collection
                         hr-znanje), multilingual-e5-small embeddings (query:/passage: prefixes),
                         `python -m hfrag ingest-qa | ingest-chunks | query | stats`. retrieve.build_messages()
                         routes by confidence: similarity >= 0.88 grounds the answer in matching chunks
                         (few-shot "copy or say Ne znam"), below that the model answers from its own
                         knowledge — weak context misleads the 1.7B. GUI chat has a "Kviz baza (RAG)" toggle;
                         the eval runner has --rag.
- `hfzakon/`             Croatian legislation RAG: scrapes zakon.hr (all ~900 consolidated laws, chunked
                         per Članak with law title + article title in every chunk) and narodne-novine.nn.hr
                         (sluzbeni dio, NN 1/2025 onward; docs chunked per Članak or windowed prose, header
                         = title + NN issue/date/vrsta/donositelj). Raw HTML cached under `rag/zakoni/raw/`
                         (resumable: re-run pull to fetch only new gazette editions), chunks under
                         `rag/zakoni/chunks/`, Chroma collection `hr-zakoni` in the same `rag/index` store.
                         `python -m hfzakon pull-zakon | discover-zakon | pull-nn | chunk | ingest |
                         refresh | query | stats`. retrieve.build_messages() is strict: context-only
                         answers with source citation, no confident hit (>= 0.84, measured) -> "Ne znam.";
                         "clanak N" questions get an exact metadata lookup. GUI toggle "Zakoni (RAG)"
                         (wins over kviz). `refresh` is incremental (NN is append-only: only new editions
                         + a tail probe for new law ids get fetched/embedded; state in
                         rag/zakoni/state/last_refresh.json). The GUI runs refresh.maybe_run() on every
                         start (skips if refreshed < 24 h ago) in a background thread with a visible
                         progress row + ETA; chat stays usable, and a Zakoni-RAG question asked
                         mid-refresh gets a transcript note that newest editions may be missing.
- `hffiles/`             file work done by Python, planned by the model: Excel (openpyxl), Word
                         (python-docx), PowerPoint (python-pptx) and PDF (pypdf + reportlab).
                         See docs/FILES.md. The model only returns a *plan* (JSON list of ops);
                         `ops.py` validates every op and parameter against its registry, and the
                         per-format module executes it. Op names are unique across formats, so the
                         plan's format follows from its ops and a plan may not mix two.
  - `config.py`  workspace under `files/` (`in/` uploads, `out/` results, `HFFILES_HOME`), per-kind
                 suffixes/labels/defaults, `safe_name`/`out_path` (a model-supplied name is reduced
                 to a basename under out/), `kind_of`
  - `files.py`   the switchboard: `kind_of`, `describe`, `digest_text`, `summary`, `upload`,
                 `copy_file`, `to_pdf` - what the GUI and CLI call
  - `excel.py`   open/save/copy, describe+digest, cell coercion, column-by-header, sort/filter/
                 format/chart/csv mechanics
  - `word.py`    block-stream document: headings, paragraphs, bullets, tables, run-aware
                 replace_text, outline/word-count reads (.docx only; .doc/.docm refused)
  - `slides.py`  layout picked from the template by placeholder inspection (never coordinates),
                 bullet overflow onto continuation slides, table/chart slides, 16:9 by default
  - `pdf.py`     pypdf for pages/metadata (lossless) + reportlab for new pages from Word-style
                 blocks; registers a system TTF so Croatian diacritics survive
  - `convert.py` PDF export: Office COM in a **child process** with DispatchEx (never drives the
                 user's open Office), else LibreOffice headless; no renderer -> clear error
  - `ops.py`     the op REGISTRY (`@op(..., kind=...)`), plan parsing (aliases, unknown params are
                 dropped with a note), `apply_plan` (read-only plans save nothing; writes go to
                 out/, in place only inside out/; `export_pdf` runs convert after saving)
  - `plan.py`    per-format prompt (rules + op reference + digest + anonymous few-shot) and
                 `extract_plan` (fences, prose, Python literals, trailing commas)
  - `cli.py`     `python -m hffiles`: describe, ops --kind, copy, upload, new, pdf, apply, ask
                 (full loop via the server), where
- `hftrain/`             fine-tuning package (LoRA via peft), separate from hfserve; see docs/TRAINING.md.
                         Trains EuroLLM on question-answer data from `training/QA/`.
  - `config.py`  paths under `training/` (QA, data, runs), training defaults
  - `data.py`    source specs (`file:`, `dir:` with `@max=N,sample=...`), QA record formats
                 (question/answer, prompt/response, alpaca, messages), txt/md passages, prepare corpus, tokenize+pack
  - `model.py`   load base via hfserve registry, LoRA, adapter, end-of-turn ids, chat prompt rendering
  - `train.py`   TrainConfig + Trainer run (fp16 on Turing), run.json/log.jsonl, resume
  - `evaluate.py` perplexity + sample generations, base vs adapter
  - `export.py`  merge adapter -> new model folder under models/ with .hfserve.json manifest
  - `cli.py`     `python -m hftrain` / `hftrain.cmd`: prepare, tokenize, data, train, runs, gui, eval, export
  - `gui.py`     tkinter window (`hftrain gui` / `HFTrain-GUI.cmd`): Prepare QA data (child `hftrain prepare`),
                 Start/Stop training as a child process, progress bar from `<run>/progress.json`, console panel,
                 GPU row (nvidia-smi), chat with a model or run adapter. Stop writes `<run>/STOP`: checkpoint,
                 then exit 130 / "interrupted", Resume continues

## Environment
- Windows 10, venv is `.vemv` (note the spelling), Python 3.12; run with `python -m hfserve` or `hfserve.cmd`.
- GPU: RTX 2070 SUPER 8 GB (Turing → no native bf16, engine picks float16). torch cu128 build.
- Default server port 11435 (11434 is left for Ollama).

## Conventions
- Never start a test server on the default port; use another port and always POST /api/shutdown afterwards.
- The working model is utter-project/EuroLLM-1.7B-Instruct (plain llama arch, no remote code). The engine
  enables `trust_remote_code` only when a model's config.json has `auto_map`.
- Training extras (peft, datasets) are optional: `pip install -r requirements-train.txt`. hftrain tests use a
  fake tokenizer and temp dirs (`train_dir` fixture); never train against the real `training/` in tests.
- EuroLLM's chat template has NO `{% generation %}` markers, so chat examples currently train on every token
  (data.py warns once per run); add markers to a template copy before serious QA tuning if assistant-only
  loss is wanted. Its tokenizer adds BOS `<s>` on encode; eos/end-of-turn is `<|im_end|>` (id 4). Croatian
  tokenizes at ~0.315 tokens/char (3.2 chars/token).
- `training/QA/` holds the question-answer data for fine-tuning EuroLLM: `.jsonl` records
  `{"system", "question", "answer"}` (system prompt "Ti si kviz-majstor..." gates the quiz behavior),
  used via `--source dir:training/QA`; provenance in `training/QA/SOURCES.txt`.
- `training/eval/` is the Croatian test suite (kviz-test.jsonl + run_kviz_test.py): run it with
  `--run <run>` after training and compare against the stored base-model baseline results
  (12 PASS / 9 FAIL / 3 REVIEW; Slovenian-drift detector included). It is not training data.
- `training/chat/` holds authored Croatian conversation examples (razgovor.jsonl, no system prompt)
  and honesty examples (ne-znam.jsonl); `training/text/wiki-hr/` holds the Croatian Wikipedia
  training slice in `<|doc|>` format. Both feed qa-v2-style mixed corpora via explicit --source.
- hftrain's load_tokenizer auto-patches ChatML templates with `{% generation %}` markers
  (model.add_generation_markers), so chat examples now train on assistant tokens only.
- Training defaults are batch 1 x grad-accum 16: on the 1.7B model, batch 2 with PACKED seq-1024 sequences
  already reserves ~8.1 GB and Windows spills VRAM into shared system RAM (everything waits on PCIe; measured
  on qa-lora-2, 2026-10-16). Short padded chat-only corpora tolerate batch 2. Never run the hfrag embedder and
  training on the GPU at the same time. Estimate run times from a benchmark of the exact config.
- File work never goes through the model: it emits a plan, Python executes it. openpyxl does not
  evaluate formulas (`data_only=True` gives Excel's cached value, `None` in a file only we wrote), and
  saving a `data_only` workbook would drop its formulas - mutating plans always load with
  `data_only=False`. Editing loses charts/pivots openpyxl cannot round-trip; `hffiles copy` is the
  byte-for-byte copy. The GUI's "Priloži Excel" copies the upload into `files/in/` and writes results
  to `files/out/`, then keeps working on the result file; "Spremi kao ..." downloads it and
  "U PDF" exports it.
- PDF export is a *rendering*, not a conversion: it needs Office (COM, Windows) or LibreOffice.
  Office automation runs in a child process with DispatchEx so it never quits the user's open
  Word/Excel; an in-process `app.Quit()` also raised RPC_S_CALL_FAILED (0x800706BE) as a Windows
  fatal-exception dump in pytest. Never fall back to re-typesetting in reportlab - a lookalike
  PDF that differs from the source is worse than an error.
- reportlab's built-in Helvetica is WinAnsi: it has no c-caron/d-stroke, so pdf.fonts() registers
  Calibri/Arial/DejaVuSans from the system before rendering Croatian text.
- `.txt`/`.md` sources with `<|doc|>` lines are split into one document per part; without them they are cut
  into passages of ~3000 characters.
