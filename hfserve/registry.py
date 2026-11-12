"""Local model registry: parse HuggingFace references, download, list and remove models.

Layout on disk (see config.MODELS_DIR)::

    models/
      utter-project/
        EuroLLM-1.7B-Instruct/
          .hfserve.json          <- manifest written by pull_model()
          config.json
          model-00000-of-00001.safetensors
          ...
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

from . import config

_HF_HOSTS = ("huggingface.co", "www.huggingface.co", "hf.co")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt", ".gguf")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class ModelError(Exception):
    """Base class for registry errors; message is safe to print to the user."""


class ModelNotFound(ModelError):
    pass


class AmbiguousModel(ModelError):
    pass


# --------------------------------------------------------------------------- refs
def parse_model_ref(text: str) -> tuple[str, str | None]:
    """Turn a user-supplied reference into ``(repo_id, revision)``.

    Accepted forms::

        https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct
        https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct/tree/main
        https://hf.co/utter-project/EuroLLM-1.7B-Instruct
        hf.co/utter-project/EuroLLM-1.7B-Instruct
        utter-project/EuroLLM-1.7B-Instruct
        utter-project/EuroLLM-1.7B-Instruct@main   (revision after '@')
        gpt2                                 (single-segment repo id)
    """
    s = text.strip().strip("\"'")
    if not s:
        raise ModelError("Empty model reference.")

    revision: str | None = None
    lowered = s.lower()
    if lowered.startswith(("http://", "https://")) or any(lowered.startswith(h + "/") for h in _HF_HOSTS):
        s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
        host, _, rest = s.partition("/")
        if host.lower() not in _HF_HOSTS:
            raise ModelError(f"Not a HuggingFace URL: {text!r}")
        rest = rest.split("?", 1)[0].split("#", 1)[0].strip("/")
        parts = [p for p in rest.split("/") if p]
        if parts and parts[0].lower() in ("datasets", "spaces"):
            raise ModelError("Only model repositories can be served (datasets/spaces are not supported).")
        if parts and parts[0].lower() == "models":
            parts = parts[1:]
        if len(parts) < 2:
            raise ModelError(f"Cannot find '<org>/<name>' in URL: {text!r}")
        repo_id = f"{parts[0]}/{parts[1]}"
        # /tree/<rev>, /blob/<rev>/..., /resolve/<rev>/...
        if len(parts) >= 4 and parts[2] in ("tree", "blob", "resolve", "commit"):
            revision = unquote(parts[3])  # e.g. refs%2Fpr%2F1
        s = repo_id
    if "@" in s:
        s, revision = s.rsplit("@", 1)
        revision = revision or None
    s = s.strip("/")
    if not _REPO_ID_RE.match(s) or s.count("/") > 1:
        raise ModelError(f"Invalid model reference: {text!r} (expected 'org/name' or a huggingface.co URL)")
    return s, revision


def looks_like_ref(text: str) -> bool:
    """True if *text* is a URL or an ``org/name`` id (i.e. something pull_model can fetch)."""
    t = text.strip().lower()
    return "/" in t or t.startswith(("http://", "https://", "hf.co"))


def model_dir(repo_id: str) -> Path:
    parts = repo_id.split("/")
    return config.MODELS_DIR.joinpath(*parts)


# --------------------------------------------------------------------------- data
@dataclass
class LocalModel:
    repo_id: str
    path: Path
    size_bytes: int = 0
    downloaded_at: str | None = None
    revision: str | None = None
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    needs_remote_code: bool = False
    has_chat_template: bool = False
    files: list[str] = field(default_factory=list)
    # True when the folder looks like an interrupted download (no manifest and either
    # leftover *.incomplete blobs or no weight file at all).
    incomplete: bool = False

    @property
    def name(self) -> str:
        return self.repo_id

    @property
    def short_name(self) -> str:
        return self.repo_id.rsplit("/", 1)[-1]

    def to_manifest(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    def to_api(self) -> dict:
        """Ollama-compatible ``/api/tags`` entry."""
        return {
            "name": self.repo_id,
            "model": self.repo_id,
            "modified_at": self.downloaded_at,
            "size": self.size_bytes,
            "digest": self.revision,
            "details": {
                "family": self.model_type,
                "architectures": self.architectures,
                "format": "safetensors" if any(f.endswith(".safetensors") for f in self.files) else "pytorch",
                "needs_remote_code": self.needs_remote_code,
                "has_chat_template": self.has_chat_template,
                "path": str(self.path),
            },
        }


def human_size(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _inspect_dir(path: Path, repo_id: str) -> LocalModel:
    """Build a LocalModel from a folder, using the manifest when present."""
    manifest = _read_json(path / config.MANIFEST_NAME)
    cfg = _read_json(path / "config.json")
    files = sorted(
        p.name for p in path.iterdir()
        if p.is_file() and p.name != config.MANIFEST_NAME
    )
    tok_cfg = _read_json(path / "tokenizer_config.json")
    has_template = (path / "chat_template.jinja").exists() or bool(tok_cfg.get("chat_template"))
    size = manifest.get("size_bytes") or dir_size(path)
    incomplete = not manifest and (
        not any(f.endswith(WEIGHT_SUFFIXES) for f in files)
        or any((path / ".cache").rglob("*.incomplete"))
    )
    return LocalModel(
        repo_id=manifest.get("repo_id") or repo_id,
        path=path,
        size_bytes=size,
        downloaded_at=manifest.get("downloaded_at")
        or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        revision=manifest.get("revision"),
        model_type=cfg.get("model_type") or manifest.get("model_type"),
        architectures=list(cfg.get("architectures") or manifest.get("architectures") or []),
        needs_remote_code=bool(cfg.get("auto_map")),
        has_chat_template=has_template,
        files=files,
        incomplete=incomplete,
    )


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and ((path / config.MANIFEST_NAME).exists() or (path / "config.json").exists())


def list_models() -> list[LocalModel]:
    """All models under MODELS_DIR, sorted by repo id (case-insensitive)."""
    root = config.MODELS_DIR
    if not root.exists():
        return []
    found: list[LocalModel] = []
    for org in sorted(root.iterdir()):
        if org.name.startswith(".") or not org.is_dir():
            continue
        if _is_model_dir(org):  # single-segment repo id, e.g. "gpt2"
            found.append(_inspect_dir(org, org.name))
            continue
        for name in sorted(org.iterdir()):
            if name.name.startswith(".") or not _is_model_dir(name):
                continue
            found.append(_inspect_dir(name, f"{org.name}/{name.name}"))
    found.sort(key=lambda m: m.repo_id.lower())
    return found


def find_model(ref: str, models: Iterable[LocalModel] | None = None) -> LocalModel:
    """Resolve *ref* to an installed model.

    *ref* may be a list index (1-based, as printed by ``hfserve list``), a full
    ``org/name`` id, a huggingface.co URL, or just the ``name`` part when it is
    unique among installed models. Raises ModelNotFound / AmbiguousModel.
    """
    models = list(models) if models is not None else list_models()
    ref = ref.strip()
    if not ref:
        raise ModelNotFound("No model given.")
    if ref.isdigit():
        idx = int(ref)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        raise ModelNotFound(f"No model with index {idx} (have {len(models)}).")
    if looks_like_ref(ref):
        repo_id, _ = parse_model_ref(ref)
        for m in models:
            if m.repo_id.lower() == repo_id.lower():
                return m
        raise ModelNotFound(f"Model '{repo_id}' is not downloaded. Run: hfserve pull {repo_id}")
    matches = [m for m in models if m.short_name.lower() == ref.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.repo_id for m in matches)
        raise AmbiguousModel(f"'{ref}' matches several models: {names}. Use the full org/name.")
    subs = [m for m in models if ref.lower() in m.repo_id.lower()]
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 1:
        names = ", ".join(m.repo_id for m in subs)
        raise AmbiguousModel(f"'{ref}' matches several models: {names}. Use the full org/name.")
    raise ModelNotFound(f"No installed model matches '{ref}'. Run 'hfserve list' or 'hfserve pull {ref}'.")


# --------------------------------------------------------------------------- pull
def _select_ignore_patterns(filenames: list[str]) -> list[str]:
    patterns = list(config.DEFAULT_IGNORE_PATTERNS)
    if any(f.endswith(".safetensors") for f in filenames):
        patterns += config.LEGACY_WEIGHT_PATTERNS
    return patterns


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def hub_download_plan(ref: str, revision: str | None = None) -> dict:
    """What pulling *ref* would fetch, without fetching it.

    Returns ``{"repo_id", "revision", "files", "total_bytes", "servable"}``. The
    file filter is the one ``pull_model`` applies, so ``total_bytes`` is the real
    download size - which is also what the weights occupy in VRAM for a float16
    safetensors repo. ``servable`` is False when the repo has no config.json and
    therefore cannot be served by transformers.

    Raises the same ModelError/ModelNotFound as pull_model for a bad reference.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError,
    )

    repo_id, ref_rev = parse_model_ref(ref)
    revision = revision or ref_rev
    try:
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError as e:  # subclass of RepositoryNotFoundError: must come first
        raise ModelError(
            f"'{repo_id}' is a gated repository. Accept its terms on huggingface.co and set the HF_TOKEN "
            "environment variable (or run `huggingface-cli login`)."
        ) from e
    except RepositoryNotFoundError as e:
        raise ModelNotFound(
            f"Repository '{repo_id}' was not found on huggingface.co (or it is private and HF_TOKEN is not set)."
        ) from e
    except RevisionNotFoundError as e:
        raise ModelError(f"Revision '{revision}' does not exist in '{repo_id}'.") from e
    except HfHubHTTPError as e:
        raise ModelError(f"HuggingFace Hub error for '{repo_id}': {e}") from e

    siblings = list(info.siblings or [])
    all_names = [s.rfilename for s in siblings]
    ignore = _select_ignore_patterns(all_names)
    wanted = [s for s in siblings if not _matches_any(s.rfilename, ignore)]
    return {
        "repo_id": repo_id,
        "revision": getattr(info, "sha", None) or revision,
        "files": len(wanted),
        "total_bytes": sum((s.size or 0) for s in wanted),
        "servable": "config.json" in all_names,
    }


def pull_model(ref: str, revision: str | None = None, quiet: bool = False, force: bool = False) -> LocalModel:
    """Download a model from the Hub into MODELS_DIR and return its LocalModel.

    Downloads resume if interrupted. Prints tqdm progress bars unless *quiet*.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError,
    )

    repo_id, ref_rev = parse_model_ref(ref)
    revision = revision or ref_rev
    config.ensure_dirs()
    target = model_dir(repo_id)

    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError as e:  # subclass of RepositoryNotFoundError: must come first
        raise ModelError(
            f"'{repo_id}' is a gated repository. Accept its terms on huggingface.co and set the HF_TOKEN "
            "environment variable (or run `huggingface-cli login`)."
        ) from e
    except RepositoryNotFoundError as e:
        raise ModelNotFound(
            f"Repository '{repo_id}' was not found on huggingface.co (or it is private and HF_TOKEN is not set)."
        ) from e
    except RevisionNotFoundError as e:
        raise ModelError(f"Revision '{revision}' does not exist in '{repo_id}'.") from e
    except HfHubHTTPError as e:
        raise ModelError(f"HuggingFace Hub error for '{repo_id}': {e}") from e

    siblings = list(info.siblings or [])
    all_names = [s.rfilename for s in siblings]
    ignore = _select_ignore_patterns(all_names)
    wanted = [s for s in siblings if not _matches_any(s.rfilename, ignore)]
    total = sum((s.size or 0) for s in wanted)

    if "config.json" not in all_names:
        raise ModelError(
            f"'{repo_id}' has no config.json; it does not look like a transformers model and cannot be served."
        )

    if not quiet:
        print(f"Pulling {repo_id}" + (f"@{revision}" if revision else "")
              + f"  ({len(wanted)} files, {human_size(total)}) -> {target}")
        sys.stdout.flush()

    started = time.time()
    kwargs = {}
    if quiet:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        ignore_patterns=ignore,
        force_download=force,
        **kwargs,
    )
    elapsed = time.time() - started

    model = _inspect_dir(target, repo_id)
    model.revision = info.sha
    model.downloaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model.size_bytes = dir_size(target)
    with open(target / config.MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(model.to_manifest(), f, indent=2)

    write_index()
    if not quiet:
        print(f"Done: {repo_id} ({human_size(model.size_bytes)}) in {elapsed:.0f}s")
    return model


# --------------------------------------------------------------------------- index
def index_file() -> Path:
    return config.MODELS_DIR / config.INDEX_NAME


def build_index(models: Iterable[LocalModel] | None = None) -> dict:
    """JSON-ready description of every installed model (what models.json contains)."""
    models = list_models() if models is None else list(models)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_dir": str(config.MODELS_DIR),
        "server": config.server_url(),
        "models": [
            {
                "name": m.repo_id,
                "short_name": m.short_name,
                "path": str(m.path),
                "size_bytes": m.size_bytes,
                "size": human_size(m.size_bytes),
                "model_type": m.model_type,
                "architectures": m.architectures,
                "revision": m.revision,
                "downloaded_at": m.downloaded_at,
                "needs_remote_code": m.needs_remote_code,
                "has_chat_template": m.has_chat_template,
                "incomplete": m.incomplete,
            }
            for m in models
        ],
    }


def write_index(models: Iterable[LocalModel] | None = None) -> Path:
    """Rewrite MODELS_DIR/models.json; returns its path. Never raises on I/O trouble."""
    path = index_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(build_index(models), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except OSError:
        pass
    return path


# --------------------------------------------------------------------------- remove
def remove_model(model: LocalModel) -> None:
    """Delete a model folder (and its then-empty org folder)."""
    path = model.path.resolve()
    root = config.MODELS_DIR.resolve()
    if root not in path.parents:
        raise ModelError(f"Refusing to delete {path}: it is outside {root}")
    shutil.rmtree(path)
    parent = path.parent
    if parent != root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
    write_index()
