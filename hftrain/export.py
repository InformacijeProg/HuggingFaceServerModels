"""Merge a run's LoRA adapter into the base weights and install the result as a new hfserve
model (``models/<org>/<name>/`` with an ``.hfserve.json`` manifest), so that
``hfserve run <org>/<name>`` serves it like any downloaded model."""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from hfserve import registry
from hfserve.registry import LocalModel

from . import model as model_mod
from .train import TrainError, adapter_dir, read_run

log = logging.getLogger("hftrain.export")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(name: str) -> str:
    name = name.strip().strip("/")
    if not _NAME_RE.match(name):
        raise TrainError(f"Model name {name!r} must look like org/name, e.g. local/EuroLLM-1.7B-hr.")
    return name


def _copy_support_files(base: Path, target: Path) -> list[str]:
    """Custom modeling code, chat templates and the generation config from the base folder."""
    copied = []
    for f in sorted(base.iterdir()):
        if not f.is_file():
            continue
        if f.suffix in (".py", ".jinja") or f.name == "generation_config.json" and not (target / f.name).exists():
            shutil.copy2(f, target / f.name)
            copied.append(f.name)
    return copied


def _ensure_auto_map(base: Path, target: Path) -> None:
    """save_pretrained of a trust_remote_code model may drop ``auto_map``; restore it from the
    base config so hfserve loads the custom classes again."""
    base_cfg = registry._read_json(base / "config.json")
    if not base_cfg.get("auto_map"):
        return
    cfg_path = target / "config.json"
    cfg = registry._read_json(cfg_path)
    changed = False
    if not cfg.get("auto_map"):
        cfg["auto_map"] = base_cfg["auto_map"]
        changed = True
    if cfg.get("architectures") != base_cfg.get("architectures") and base_cfg.get("architectures"):
        cfg["architectures"] = base_cfg["architectures"]
        changed = True
    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def write_manifest(target: Path, repo_id: str, base: LocalModel, run: str, extra: dict | None = None) -> LocalModel:
    files = sorted(p.name for p in target.iterdir() if p.is_file() and p.name != registry.config.MANIFEST_NAME)
    cfg = registry._read_json(target / "config.json")
    local = LocalModel(
        repo_id=repo_id, path=target, size_bytes=registry.dir_size(target),
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        revision=f"hftrain:{run}", model_type=cfg.get("model_type") or base.model_type,
        architectures=list(cfg.get("architectures") or base.architectures),
        needs_remote_code=bool(cfg.get("auto_map")),
        has_chat_template=(target / "chat_template.jinja").exists()
        or bool(registry._read_json(target / "tokenizer_config.json").get("chat_template")),
        files=files,
    )
    manifest = local.to_manifest()
    manifest["base_model"] = base.repo_id
    manifest["base_revision"] = base.revision
    manifest.update(extra or {})
    with open(target / registry.config.MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return local


def export_merged(run: str, name: str, *, dtype: str = "bfloat16", force: bool = False,
                  model_ref: str | None = None, progress=print) -> LocalModel:
    name = validate_name(name)
    info = read_run(run)
    adir = adapter_dir(run)
    base = model_mod.find_local(model_ref or info["model"])
    target = registry.model_dir(name)
    if target.exists():
        if not force:
            raise TrainError(f"{name} already exists at {target}; use --force to replace it.")
        if base.path.resolve() == target.resolve():
            raise TrainError("Refusing to overwrite the base model itself.")
        shutil.rmtree(target)
    out_dtype = model_mod.engine.parse_dtype(dtype)

    progress(f"loading {base.repo_id} in float32 on cpu for the merge ...")
    tok = model_mod.load_tokenizer(base)
    model = model_mod.load_model(base, "cpu", torch.float32, training=False)
    progress(f"merging adapter {adir} ...")
    model = model_mod.load_adapter(model, adir)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model = model.to(out_dtype)

    target.mkdir(parents=True)
    progress(f"saving {name} ({model_mod.engine.dtype_name(out_dtype)}) to {target} ...")
    model.save_pretrained(str(target), safe_serialization=True)
    tok.save_pretrained(str(target))
    copied = _copy_support_files(base.path, target)
    _ensure_auto_map(base.path, target)
    _write_readme(target, name, base, run, info)
    local = write_manifest(target, name, base, run, {
        "run": run, "adapter": str(adir), "train_loss": info.get("train_loss"),
        "eval_loss_start": info.get("eval_loss_start"), "eval_loss_end": info.get("eval_loss_end"),
        "data": info.get("data"), "copied_files": copied,
    })
    registry.write_index()
    return local


def _write_readme(target: Path, name: str, base: LocalModel, run: str, info: dict) -> None:
    lines = [
        f"# {name}",
        "",
        f"Merged LoRA fine-tune of `{base.repo_id}` (revision `{base.revision}`) produced by hftrain run `{run}`.",
        "",
        f"* data: `{info.get('data')}`",
        f"* steps: {info.get('steps')}, tokens seen: {info.get('tokens_seen')}",
        f"* train loss: {info.get('train_loss')}, eval loss {info.get('eval_loss_start')} -> {info.get('eval_loss_end')}",
        "",
        "Serve it with `hfserve run " + name + "`. The base model's licence applies to the weights;",
        "the training data's licences apply to what the model learned from it.",
        "",
    ]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")
