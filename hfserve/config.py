"""Paths, defaults and environment overrides shared by every hfserve module.

Environment variables:
  HFSERVE_HOME     root folder for models and state (default: project root,
                   i.e. the folder that contains the ``hfserve`` package)
  HFSERVE_MODELS   folder where downloaded models live (default: HOME/models)
  HFSERVE_HOST     bind address of the API server (default: 127.0.0.1)
  HFSERVE_PORT     port of the API server (default: 11435)
  HFSERVE_DEVICE   force "cuda", "cuda:0" or "cpu" (default: auto)
  HFSERVE_DTYPE    force "float16", "bfloat16" or "float32" (default: auto)
  HF_TOKEN         HuggingFace token for gated/private repos (read by huggingface_hub)

A ``.env`` file in the project root is read for HFSERVE_HOME / HFSERVE_MODELS and
for the plain ``MODELS_DIR`` name, so ``MODELS_DIR=./models`` there is what the
server, the registry and the GUI actually load models from. A real environment
variable always wins over .env (standard dotenv precedence); nothing else from
.env is injected into the process environment.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "hfserve"


class ConfigError(RuntimeError):
    """A setting points somewhere unusable; the message is meant for the user."""


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines out of a .env file. A missing/unreadable file gives {}."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # A trailing "  # note" is a comment, not part of a path or a port.
            comment = value.find(" #")
            if comment != -1:
                value = value[:comment].rstrip()
        if key:
            values[key] = value
    return values


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
# Only these keys are taken from .env. Anything else in the file (an HF_TOKEN, say) is
# left where it is and never enters this process's environment, so it cannot leak through
# a settings dump or a child process - huggingface_hub still wants a real HF_TOKEN.
DOTENV_KEYS = ("HFSERVE_HOME", "HFSERVE_MODELS", "MODELS_DIR",
               "HFSERVE_HOST", "HFSERVE_PORT", "HFSERVE_URL", "SERVER_URL")
DOTENV = {k: v for k, v in _load_env_file(ENV_FILE).items() if k in DOTENV_KEYS}

# Settings that had to be ignored because they could not be read. Collected instead of
# raised: a bad port must not make `import hfserve` fail, but it must not pass silently
# either - the GUI shows these at start-up and `hfserve serve` prints them.
SETTINGS_PROBLEMS: list[str] = []


def _setting(*names: str) -> tuple[str | None, str]:
    """First non-empty value of *names* in the real environment, then in .env.

    Returns ``(value, source)``; *source* names where it came from so an error
    message can point at the thing the user actually has to fix.
    """
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip(), f"environment variable {name}"
    for name in names:
        value = DOTENV.get(name)
        if value and value.strip():
            return value.strip(), f"{name} in {ENV_FILE}"
    return None, "built-in default"


def _as_dir(value: str, base: Path) -> Path:
    """Resolve a configured folder; a relative value is relative to *base*, not the cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


_home, HOME_DIR_SOURCE = _setting("HFSERVE_HOME")
HOME_DIR = _as_dir(_home, PROJECT_ROOT) if _home else PROJECT_ROOT

# HFSERVE_MODELS is the documented name; MODELS_DIR is accepted so the plain key in
# .env works. A relative value such as "./models" resolves against HOME_DIR.
_models, MODELS_DIR_SOURCE = _setting("HFSERVE_MODELS", "MODELS_DIR")
MODELS_DIR = _as_dir(_models, HOME_DIR) if _models else (HOME_DIR / "models").resolve()
STATE_DIR = HOME_DIR / ".hfserve"

PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"
# Last model loaded from the GUI, so the next start can preselect it.
LAST_MODEL_FILE = STATE_DIR / "last_model.json"

def _split_server_url(value: str) -> tuple[str | None, int | None]:
    """``(host, port)`` out of "http://127.0.0.1:11435", "127.0.0.1:11435" or ":11435"."""
    from urllib.parse import urlsplit

    text = value.strip()
    if "://" not in text:
        text = "//" + text.lstrip("/")  # urlsplit only finds a netloc after "//"
    parts = urlsplit(text)
    try:
        port = parts.port
    except ValueError:
        port = None
    return (parts.hostname or None), port


# The address the server listens on and every client (GUI, console chat, Start-HFServe)
# connects to. HFSERVE_HOST / HFSERVE_PORT win; otherwise a whole URL is accepted as
# HFSERVE_URL or SERVER_URL, in the environment or in .env. 11434 stays Ollama's.
_url, URL_SOURCE = _setting("HFSERVE_URL", "SERVER_URL")
_url_host, _url_port = _split_server_url(_url) if _url else (None, None)
if _url and _url_host is None and _url_port is None:
    SETTINGS_PROBLEMS.append(f"No host or port could be read from {_url!r} ({URL_SOURCE}).")

_host, HOST_SOURCE = _setting("HFSERVE_HOST")
if not _host and _url_host:
    _host, HOST_SOURCE = _url_host, URL_SOURCE
DEFAULT_HOST = _host or "127.0.0.1"

_port, PORT_SOURCE = _setting("HFSERVE_PORT")
DEFAULT_PORT = 11435
if _port:
    try:
        DEFAULT_PORT = int(_port)
    except ValueError:
        SETTINGS_PROBLEMS.append(f"{_port!r} is not a port number ({PORT_SOURCE}); using 11435.")
        PORT_SOURCE = "built-in default, after an unreadable setting"
elif _url_port:
    DEFAULT_PORT, PORT_SOURCE = _url_port, URL_SOURCE
if not 1 <= DEFAULT_PORT <= 65535:
    SETTINGS_PROBLEMS.append(f"Port {DEFAULT_PORT} is out of range ({PORT_SOURCE}); using 11435.")
    DEFAULT_PORT, PORT_SOURCE = 11435, "built-in default, after an out-of-range setting"

# Written into every downloaded model folder; marks the folder as an hfserve model.
MANIFEST_NAME = ".hfserve.json"
# Index of every downloaded model, kept at MODELS_DIR/models.json for scripts
# (see Start-HFServe.ps1); rewritten by pull/rm and by `hfserve list`.
INDEX_NAME = "models.json"

# Files never worth downloading for a transformers-based server.
DEFAULT_IGNORE_PATTERNS = [
    "*.gguf", "*.onnx", "*.tflite", "*.h5", "*.msgpack", "*.ot",
    "onnx/*", "coreml/*", "*.mlpackage/*",
]
# Dropped when the repo also ships safetensors (avoids downloading weights twice).
LEGACY_WEIGHT_PATTERNS = ["*.bin", "*.pth", "*.pt", "*.ckpt"]

# Generation defaults (Ollama-like option names are mapped onto these in server.py).
# 0 / 1.0 mean "disabled" for top_k / repetition_penalty. Reasoning models think
# before answering, hence the generous token budget.
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.0

# How long the chat client waits for a freshly spawned server to answer /api/health.
SERVER_START_TIMEOUT = 60.0
# How long a graceful shutdown may take before processes are killed.
SHUTDOWN_GRACE = 8.0


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def require_models_dir() -> Path:
    """MODELS_DIR, or a ConfigError naming the setting that points at the wrong place.

    Fail loud on purpose: a models folder that does not exist (a typo in .env, a
    drive that is not mounted) otherwise surfaces much later, and only as the far
    more confusing "no models installed".
    """
    if not MODELS_DIR.exists():
        raise ConfigError(
            f"The models folder does not exist:\n  {MODELS_DIR}\n\n"
            f"That path comes from {MODELS_DIR_SOURCE}.\n"
            f"Fix the setting, or create the folder and download a model with:\n"
            f"  hfserve pull <org/name>")
    if not MODELS_DIR.is_dir():
        raise ConfigError(
            f"The models path is not a folder:\n  {MODELS_DIR}\n\n"
            f"That path comes from {MODELS_DIR_SOURCE}.")
    return MODELS_DIR


def read_last_model() -> str | None:
    """The model last loaded from the GUI, or None when nothing is remembered."""
    try:
        data = json.loads(LAST_MODEL_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("model") if isinstance(data, dict) else None
    return str(name) if name else None


def write_last_model(name: str) -> None:
    """Remember *name* for the next start. Never fatal - it is only a convenience."""
    if not name:
        return
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_MODEL_FILE.write_text(
            json.dumps({"model": name,
                        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                       indent=2),
            encoding="utf-8")
    except OSError:
        pass


def server_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"
