"""hftrain.export: manifest writing and support-file handling (no model weights involved)."""
from __future__ import annotations

import json

import pytest

from hfserve import config, registry
from hftrain import export
from hftrain.train import TrainError


@pytest.mark.parametrize("name", ["local/K2-hr", "IFM/K2-Horizon-0.9B-hr", " org/name "])
def test_validate_name_ok(name):
    assert "/" in export.validate_name(name)


@pytest.mark.parametrize("name", ["", "noslash", "a/b/c", "/x", "-bad/name", "org/na me"])
def test_validate_name_rejects(name):
    with pytest.raises(TrainError):
        export.validate_name(name)


def test_ensure_auto_map_restores_from_base(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    base.mkdir(), target.mkdir()
    auto_map = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    (base / "config.json").write_text(json.dumps({"auto_map": auto_map, "architectures": ["XForCausalLM"]}), encoding="utf-8")
    (target / "config.json").write_text(json.dumps({"architectures": ["Other"]}), encoding="utf-8")
    export._ensure_auto_map(base, target)
    cfg = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert cfg["auto_map"] == auto_map and cfg["architectures"] == ["XForCausalLM"]


def test_ensure_auto_map_noop_without_custom_code(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    base.mkdir(), target.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (target / "config.json").write_text("{}", encoding="utf-8")
    export._ensure_auto_map(base, target)
    assert (target / "config.json").read_text(encoding="utf-8") == "{}"


def test_copy_support_files_copies_code_and_templates_only(tmp_path):
    base, target = tmp_path / "base", tmp_path / "target"
    base.mkdir(), target.mkdir()
    for name in ("modeling_x.py", "configuration_x.py", "chat_template.jinja", "generation_config.json",
                 "model.safetensors", "README.md"):
        (base / name).write_text("x", encoding="utf-8")
    copied = export._copy_support_files(base, target)
    assert set(copied) == {"modeling_x.py", "configuration_x.py", "chat_template.jinja", "generation_config.json"}
    (target / "generation_config.json").write_text("mine", encoding="utf-8")
    export._copy_support_files(base, target)
    assert (target / "generation_config.json").read_text(encoding="utf-8") == "mine"  # not overwritten


def test_write_manifest_makes_model_visible_to_hfserve(make_model, models_dir):
    base = make_model("IFM/K2", model_type="k2_horizon", architectures=("K2ForCausalLM",),
                      auto_map={"AutoModelForCausalLM": "modeling_k2.K2ForCausalLM"},
                      manifest={"repo_id": "IFM/K2", "revision": "abc"}, files={"model.safetensors": 10})
    base_local = registry.find_model("IFM/K2")
    target = registry.model_dir("local/K2-hr")
    target.mkdir(parents=True)
    (target / "config.json").write_text(json.dumps({
        "model_type": "k2_horizon", "architectures": ["K2ForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_k2.K2ForCausalLM"}}), encoding="utf-8")
    (target / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"x" * 1000)

    local = export.write_manifest(target, "local/K2-hr", base_local, "run1", {"data": "hr-v1"})
    manifest = json.loads((target / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["repo_id"] == "local/K2-hr" and manifest["base_model"] == "IFM/K2"
    assert manifest["revision"] == "hftrain:run1" and manifest["data"] == "hr-v1"
    assert local.needs_remote_code and local.has_chat_template and local.size_bytes >= 1000
    assert config.MANIFEST_NAME not in local.files and "model.safetensors" in local.files

    names = [m.repo_id for m in registry.list_models()]
    assert names == ["IFM/K2", "local/K2-hr"]
    found = registry.find_model("K2-hr")
    assert found.path == target and not found.incomplete


def test_export_refuses_existing_target_without_force(train_dir, make_model, monkeypatch):
    from hftrain import train as ht

    make_model("IFM/K2", files={"model.safetensors": 10}, manifest={"repo_id": "IFM/K2"})
    run = train_dir / "runs" / "r1"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    ht.write_run("r1", {"run": "r1", "model": "IFM/K2", "status": "finished"})
    target = registry.model_dir("local/K2-hr")
    target.mkdir(parents=True)
    with pytest.raises(TrainError, match="--force"):
        export.export_merged("r1", "local/K2-hr")
    with pytest.raises(TrainError, match="org/name"):
        export.export_merged("r1", "bad")
    with pytest.raises(TrainError, match="Refusing"):
        export.export_merged("r1", "IFM/K2", force=True)
