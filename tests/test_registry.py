"""Unit tests for hfserve.registry. No network access: huggingface_hub is replaced by fakes."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from hfserve import config, registry
from hfserve.registry import AmbiguousModel, LocalModel, ModelError, ModelNotFound

REPO = "IFM/K2-Horizon-0.9B"


# =========================================================================== parse_model_ref
@pytest.mark.parametrize(
    "text, expected",
    [
        # full URLs
        ("https://huggingface.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("http://huggingface.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/", (REPO, None)),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/tree/main", (REPO, "main")),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/tree/main/", (REPO, "main")),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/tree", (REPO, None)),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/blob/abc123/config.json", (REPO, "abc123")),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/resolve/v2/model.safetensors", (REPO, "v2")),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/commit/deadbeef", (REPO, "deadbeef")),
        ("https://huggingface.co/models/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("HTTPS://HuggingFace.CO/IFM/K2-Horizon-0.9B", (REPO, None)),
        # query strings / fragments
        ("https://huggingface.co/IFM/K2-Horizon-0.9B?not-for-all-audiences=true", (REPO, None)),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/tree/main?library=transformers#readme", (REPO, "main")),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B/?x=1", (REPO, None)),
        # alternative hosts, with and without scheme
        ("https://hf.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("hf.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("HF.CO/IFM/K2-Horizon-0.9B/tree/dev", (REPO, "dev")),
        ("huggingface.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("https://www.huggingface.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        ("www.huggingface.co/IFM/K2-Horizon-0.9B", (REPO, None)),
        # bare ids
        ("IFM/K2-Horizon-0.9B", (REPO, None)),
        ("IFM/K2-Horizon-0.9B/", (REPO, None)),
        ("IFM/K2-Horizon-0.9B@main", (REPO, "main")),
        ("IFM/K2-Horizon-0.9B@refs/pr/1", (REPO, "refs/pr/1")),
        ("IFM/K2-Horizon-0.9B@", (REPO, None)),
        ("gpt2", ("gpt2", None)),
        ("gpt2@main", ("gpt2", "main")),
        ("bert-base_uncased.v1", ("bert-base_uncased.v1", None)),
        # surrounding quotes / whitespace
        ('"IFM/K2-Horizon-0.9B"', (REPO, None)),
        ("'IFM/K2-Horizon-0.9B'", (REPO, None)),
        ("  IFM/K2-Horizon-0.9B \n", (REPO, None)),
        ('  "https://huggingface.co/IFM/K2-Horizon-0.9B/tree/main"  ', (REPO, "main")),
        ("\t'hf.co/IFM/K2-Horizon-0.9B'\t", (REPO, None)),
    ],
)
def test_parse_model_ref_accepted_forms(text, expected):
    assert registry.parse_model_ref(text) == expected


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "Empty model reference"),
        ("   ", "Empty model reference"),
        ('""', "Empty model reference"),
        ("https://github.com/IFM/K2-Horizon-0.9B", "Not a HuggingFace URL"),
        ("https://huggingface.com/IFM/K2-Horizon-0.9B", "Not a HuggingFace URL"),
        ("http://example.com/x/y", "Not a HuggingFace URL"),
        ("https://huggingface.co/datasets/IFM/some-data", "datasets/spaces"),
        ("https://huggingface.co/spaces/IFM/demo", "datasets/spaces"),
        ("hf.co/datasets/IFM/some-data", "datasets/spaces"),
        ("https://huggingface.co/", "Cannot find"),
        ("https://huggingface.co/IFM", "Cannot find"),
        ("https://huggingface.co/models", "Cannot find"),
        ("IFM/K2 Horizon", "Invalid model reference"),
        ("IFM/K2-Horizon-0.9B!", "Invalid model reference"),
        ("IFM/K2:latest", "Invalid model reference"),
        ("-IFM/K2", "Invalid model reference"),
        (".IFM/K2", "Invalid model reference"),
        ("IFM/.K2", "Invalid model reference"),
        ("IFM//K2", "Invalid model reference"),
        ("a/b/c", "Invalid model reference"),
        ("org/name@rev@rev2", "Invalid model reference"),
        ("ftp://huggingface.co/IFM/K2", "Invalid model reference"),
    ],
)
def test_parse_model_ref_errors(text, match):
    with pytest.raises(ModelError, match=match) as excinfo:
        registry.parse_model_ref(text)
    assert excinfo.type is ModelError  # a parse failure is neither NotFound nor Ambiguous


def test_parse_model_ref_error_quotes_original_text():
    with pytest.raises(ModelError) as excinfo:
        registry.parse_model_ref("https://github.com/IFM/K2")
    assert "'https://github.com/IFM/K2'" in str(excinfo.value)


# =========================================================================== looks_like_ref
@pytest.mark.parametrize(
    "text, expected",
    [
        ("IFM/K2-Horizon-0.9B", True),
        ("https://huggingface.co/IFM/K2-Horizon-0.9B", True),
        ("http://huggingface.co/IFM/K2-Horizon-0.9B", True),
        ("hf.co/IFM/K2-Horizon-0.9B", True),
        ("HF.CO/IFM/K2-Horizon-0.9B", True),
        ("  IFM/K2-Horizon-0.9B  ", True),
        ("gpt2", False),
        ("K2-Horizon-0.9B", False),
        ("", False),
        ("   ", False),
        ("3", False),
        ("huggingface", False),
    ],
)
def test_looks_like_ref(text, expected):
    assert registry.looks_like_ref(text) is expected


# =========================================================================== model_dir
def test_model_dir_two_segments(models_dir):
    assert registry.model_dir("IFM/K2-Horizon-0.9B") == models_dir / "IFM" / "K2-Horizon-0.9B"


def test_model_dir_single_segment(models_dir):
    assert registry.model_dir("gpt2") == models_dir / "gpt2"


def test_model_dir_reads_config_at_call_time(models_dir, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    monkeypatch.setattr(config, "MODELS_DIR", other)
    assert registry.model_dir("a/b") == other / "a" / "b"


# =========================================================================== human_size / dir_size
@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1536.0, "1.5 KB"),
        (10 * 1024, "10.0 KB"),
        (1024 ** 2 - 1, "1024.0 KB"),
        (1024 ** 2, "1.0 MB"),
        (int(2.5 * 1024 ** 2), "2.5 MB"),
        (1024 ** 3, "1.0 GB"),
        (int(7.5 * 1024 ** 3), "7.5 GB"),
        (1024 ** 4, "1.0 TB"),
        (1024 ** 5, "1024.0 TB"),
    ],
)
def test_human_size(n, expected):
    assert registry.human_size(n) == expected


def test_dir_size_counts_nested_files(make_model):
    path = make_model("org/m", files={"a.safetensors": 100, "sub/b.txt": 23})
    cfg = (path / "config.json").stat().st_size
    assert registry.dir_size(path) == 100 + 23 + cfg


def test_dir_size_empty_dir(tmp_path):
    assert registry.dir_size(tmp_path) == 0


# =========================================================================== _inspect_dir
def test_inspect_dir_reads_config_json(make_model):
    path = make_model(REPO, model_type="k2", architectures=("K2ForCausalLM",),
                      files={"model.safetensors": 100})
    m = registry._inspect_dir(path, REPO)
    assert m.repo_id == REPO
    assert m.name == REPO
    assert m.short_name == "K2-Horizon-0.9B"
    assert m.path == path
    assert m.model_type == "k2"
    assert m.architectures == ["K2ForCausalLM"]
    assert m.files == ["config.json", "model.safetensors"]
    assert m.revision is None
    assert m.needs_remote_code is False
    assert m.has_chat_template is False
    # no manifest: downloaded_at derives from the folder mtime, as an aware ISO timestamp
    assert datetime.fromisoformat(m.downloaded_at).tzinfo is not None


def test_inspect_dir_size_computed_without_manifest(make_model):
    path = make_model("org/m", files={"a.safetensors": 1000, "b.bin": 24})
    cfg = (path / "config.json").stat().st_size
    m = registry._inspect_dir(path, "org/m")
    assert m.size_bytes == 1000 + 24 + cfg
    assert m.size_bytes == registry.dir_size(path)


def test_inspect_dir_size_from_manifest_wins(make_model):
    path = make_model("org/m", manifest={"repo_id": "org/m", "size_bytes": 987_654_321},
                      files={"a.safetensors": 10})
    m = registry._inspect_dir(path, "org/m")
    assert m.size_bytes == 987_654_321


def test_inspect_dir_manifest_zero_size_falls_back_to_computed(make_model):
    path = make_model("org/m", manifest={"repo_id": "org/m", "size_bytes": 0},
                      files={"a.safetensors": 10})
    m = registry._inspect_dir(path, "org/m")
    assert m.size_bytes == registry.dir_size(path) > 10


def test_inspect_dir_manifest_fields(make_model):
    manifest = {
        "repo_id": REPO,
        "revision": "abc123",
        "downloaded_at": "2026-01-02T03:04:05+00:00",
        "model_type": "from-manifest",
        "architectures": ["ManifestArch"],
        "size_bytes": 5,
    }
    path = make_model("ifm/k2-horizon-0.9b", manifest=manifest, config_json=False)
    m = registry._inspect_dir(path, "ifm/k2-horizon-0.9b")
    assert m.repo_id == REPO  # manifest id wins over the folder-derived id
    assert m.revision == "abc123"
    assert m.downloaded_at == "2026-01-02T03:04:05+00:00"
    assert m.model_type == "from-manifest"
    assert m.architectures == ["ManifestArch"]
    assert m.files == []  # the manifest itself is never listed


def test_inspect_dir_config_json_beats_manifest(make_model):
    path = make_model("org/m", model_type="llama", architectures=("LlamaForCausalLM",),
                      manifest={"model_type": "old", "architectures": ["Old"]})
    m = registry._inspect_dir(path, "org/m")
    assert m.model_type == "llama"
    assert m.architectures == ["LlamaForCausalLM"]


def test_inspect_dir_manifest_excluded_from_files(make_model):
    path = make_model("org/m", manifest={"repo_id": "org/m"}, files={"z.safetensors": 1, "README.md": 1})
    m = registry._inspect_dir(path, "org/m")
    assert config.MANIFEST_NAME not in m.files
    assert m.files == sorted(["config.json", "z.safetensors", "README.md"])


@pytest.mark.parametrize(
    "auto_map, expected",
    [
        ({"AutoModelForCausalLM": "modeling_k2.K2ForCausalLM"}, True),
        ({}, False),
        (None, False),
    ],
)
def test_inspect_dir_needs_remote_code_from_auto_map(make_model, auto_map, expected):
    path = make_model("org/m", auto_map=auto_map)
    assert registry._inspect_dir(path, "org/m").needs_remote_code is expected


def test_inspect_dir_has_chat_template_from_jinja_file(make_model):
    path = make_model("org/m", chat_template_file=True)
    assert registry._inspect_dir(path, "org/m").has_chat_template is True


def test_inspect_dir_has_chat_template_from_tokenizer_config(make_model):
    path = make_model("org/m", tokenizer_chat_template="{% for m in messages %}{{ m.content }}{% endfor %}")
    assert registry._inspect_dir(path, "org/m").has_chat_template is True


@pytest.mark.parametrize("tok_cfg", [None, {}, {"chat_template": ""}, {"chat_template": None}])
def test_inspect_dir_has_chat_template_false_without_template(make_model, tok_cfg):
    path = make_model("org/m", tokenizer_config=tok_cfg)
    assert registry._inspect_dir(path, "org/m").has_chat_template is False


def test_inspect_dir_tolerates_corrupt_json(make_model):
    path = make_model("org/m", config_json=False, manifest={"repo_id": "org/m"})
    (path / "config.json").write_text("{not json", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("[1, 2]", encoding="utf-8")  # valid JSON, not a dict
    m = registry._inspect_dir(path, "org/m")
    assert m.model_type is None
    assert m.architectures == []
    assert m.needs_remote_code is False
    assert m.has_chat_template is False


def test_local_model_to_api_and_manifest(make_model):
    path = make_model("org/m", files={"model.safetensors": 1})
    m = registry._inspect_dir(path, "org/m")
    api = m.to_api()
    assert api["name"] == api["model"] == "org/m"
    assert api["size"] == m.size_bytes
    assert api["details"]["format"] == "safetensors"
    assert api["details"]["path"] == str(path)
    manifest = m.to_manifest()
    assert manifest["repo_id"] == "org/m"
    assert manifest["path"] == str(path)
    json.dumps(manifest)  # must be serialisable as written by pull_model


def test_local_model_to_api_format_pytorch_without_safetensors():
    m = LocalModel(repo_id="org/m", path=Path("x"), files=["config.json", "pytorch_model.bin"])
    assert m.to_api()["details"]["format"] == "pytorch"


# =========================================================================== list_models
def test_list_models_missing_root(models_dir, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", models_dir / "does-not-exist")
    assert registry.list_models() == []


def test_list_models_empty_root(models_dir):
    assert registry.list_models() == []


def test_list_models_sorted_case_insensitive(make_model):
    for rid in ("Zeta/m", "alpha/m", "Beta/m", "gamma/m"):
        make_model(rid)
    assert [m.repo_id for m in registry.list_models()] == ["alpha/m", "Beta/m", "gamma/m", "Zeta/m"]


def test_list_models_populates_fields(make_model):
    make_model(REPO, model_type="k2", auto_map={"AutoModelForCausalLM": "x.Y"},
               chat_template_file=True, files={"model.safetensors": 300},
               manifest={"repo_id": REPO, "revision": "rev1", "size_bytes": 42})
    [m] = registry.list_models()
    assert m.repo_id == REPO
    assert m.path == registry.model_dir(REPO)
    assert m.size_bytes == 42
    assert m.revision == "rev1"
    assert m.model_type == "k2"
    assert m.needs_remote_code is True
    assert m.has_chat_template is True


def test_list_models_single_segment_repo_folder(make_model, models_dir):
    make_model("gpt2", model_type="gpt2")
    make_model("org/name")
    models = registry.list_models()
    assert [m.repo_id for m in models] == ["gpt2", "org/name"]
    assert models[0].path == models_dir / "gpt2"
    assert models[0].short_name == "gpt2"


def test_list_models_skips_hidden_and_non_model_dirs(make_model, models_dir):
    make_model("org/real")
    hidden_top = models_dir / ".cache" / "x"
    hidden_top.mkdir(parents=True)
    (hidden_top / "config.json").write_text("{}", encoding="utf-8")
    hidden_nested = models_dir / "org" / ".hidden"
    hidden_nested.mkdir()
    (hidden_nested / "config.json").write_text("{}", encoding="utf-8")
    incomplete = models_dir / "org" / "incomplete"  # no config.json, no manifest
    incomplete.mkdir()
    (incomplete / "README.md").write_text("x", encoding="utf-8")
    (models_dir / "emptyorg").mkdir()
    (models_dir / "stray.txt").write_text("x", encoding="utf-8")
    (models_dir / "org" / "stray.txt").write_text("x", encoding="utf-8")
    assert [m.repo_id for m in registry.list_models()] == ["org/real"]


def test_list_models_manifest_only_dir_is_listed(make_model):
    make_model("org/m", config_json=False, manifest={"repo_id": "org/m"})
    assert [m.repo_id for m in registry.list_models()] == ["org/m"]


# =========================================================================== find_model
def _lm(repo_id: str) -> LocalModel:
    return LocalModel(repo_id=repo_id, path=Path("models").joinpath(*repo_id.split("/")))


@pytest.fixture
def installed() -> list[LocalModel]:
    return [
        _lm("IFM/K2-Horizon-0.9B"),
        _lm("meta-llama/Llama-3.2-1B-Instruct"),
        _lm("Qwen/Qwen2.5-0.5B-Instruct"),
        _lm("gpt2"),
        _lm("org-a/shared-name"),
        _lm("org-b/shared-name"),
    ]


@pytest.mark.parametrize("ref, index", [("1", 0), ("2", 1), ("6", 5), (" 3 ", 2), ("01", 0)])
def test_find_model_by_index(installed, ref, index):
    assert registry.find_model(ref, installed) is installed[index]


@pytest.mark.parametrize("ref", ["0", "7", "999"])
def test_find_model_index_out_of_range(installed, ref):
    with pytest.raises(ModelNotFound, match="have 6"):
        registry.find_model(ref, installed)


def test_find_model_index_with_no_models():
    with pytest.raises(ModelNotFound, match="have 0"):
        registry.find_model("1", [])


@pytest.mark.parametrize("ref", ["IFM/K2-Horizon-0.9B", "ifm/k2-horizon-0.9b", "IFM/K2-HORIZON-0.9B",
                                 "IFM/K2-Horizon-0.9B@main", "IFM/K2-Horizon-0.9B/"])
def test_find_model_by_full_id_case_insensitive(installed, ref):
    assert registry.find_model(ref, installed) is installed[0]


@pytest.mark.parametrize("ref", [
    "https://huggingface.co/IFM/K2-Horizon-0.9B",
    "https://huggingface.co/IFM/K2-Horizon-0.9B/tree/main",
    "hf.co/ifm/k2-horizon-0.9b",
    '"https://huggingface.co/IFM/K2-Horizon-0.9B"',
])
def test_find_model_by_url(installed, ref):
    assert registry.find_model(ref, installed) is installed[0]


def test_find_model_url_not_installed(installed):
    with pytest.raises(ModelNotFound, match="hfserve pull Foo/Bar"):
        registry.find_model("https://huggingface.co/Foo/Bar", installed)


def test_find_model_invalid_ref_raises_model_error(installed):
    with pytest.raises(ModelError) as excinfo:
        registry.find_model("https://github.com/Foo/Bar", installed)
    assert excinfo.type is ModelError


@pytest.mark.parametrize("ref, index", [
    ("K2-Horizon-0.9B", 0), ("k2-horizon-0.9b", 0), ("Llama-3.2-1B-Instruct", 1), ("gpt2", 3), ("GPT2", 3),
])
def test_find_model_by_unique_short_name(installed, ref, index):
    assert registry.find_model(ref, installed) is installed[index]


def test_find_model_ambiguous_short_name(installed):
    with pytest.raises(AmbiguousModel) as excinfo:
        registry.find_model("shared-name", installed)
    msg = str(excinfo.value)
    assert "org-a/shared-name" in msg and "org-b/shared-name" in msg


@pytest.mark.parametrize("ref, index", [("horizon", 0), ("llama-3", 1), ("qwen2.5", 2), ("org-a", 4)])
def test_find_model_by_unique_substring(installed, ref, index):
    assert registry.find_model(ref, installed) is installed[index]


def test_find_model_ambiguous_substring(installed):
    with pytest.raises(AmbiguousModel) as excinfo:
        registry.find_model("instruct", installed)
    msg = str(excinfo.value)
    assert "meta-llama/Llama-3.2-1B-Instruct" in msg and "Qwen/Qwen2.5-0.5B-Instruct" in msg


def test_find_model_exact_short_name_beats_substring():
    models = [_lm("A/foo"), _lm("B/foobar")]
    assert registry.find_model("foo", models) is models[0]


@pytest.mark.parametrize("ref", ["does-not-exist", "zzz"])
def test_find_model_no_match(installed, ref):
    with pytest.raises(ModelNotFound, match="No installed model matches"):
        registry.find_model(ref, installed)


@pytest.mark.parametrize("ref", ["", "   ", "\t\n"])
def test_find_model_empty_ref(installed, ref):
    with pytest.raises(ModelNotFound, match="No model given"):
        registry.find_model(ref, installed)


def test_find_model_uses_list_models_by_default(make_model):
    make_model("Zeta/last")
    make_model("alpha/first")
    assert registry.find_model("2").repo_id == "Zeta/last"
    assert registry.find_model("first").repo_id == "alpha/first"
    with pytest.raises(ModelNotFound):
        registry.find_model("3")


# =========================================================================== remove_model
def test_remove_model_deletes_model_and_empty_org(models_dir, make_model):
    path = make_model("org/name", files={"w.safetensors": 10})
    registry.remove_model(LocalModel(repo_id="org/name", path=path))
    assert not path.exists()
    assert not (models_dir / "org").exists()
    assert models_dir.exists()


def test_remove_model_keeps_org_with_other_models(models_dir, make_model):
    victim = make_model("org/victim")
    other = make_model("org/other")
    registry.remove_model(LocalModel(repo_id="org/victim", path=victim))
    assert not victim.exists()
    assert other.is_dir()
    assert (models_dir / "org").is_dir()


def test_remove_model_single_segment_keeps_root(models_dir, make_model):
    path = make_model("gpt2")
    registry.remove_model(LocalModel(repo_id="gpt2", path=path))
    assert not path.exists()
    assert models_dir.is_dir()


def test_remove_model_refuses_path_outside_models_dir(models_dir, tmp_path):
    outside = tmp_path / "elsewhere" / "victim"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ModelError, match="Refusing to delete"):
        registry.remove_model(LocalModel(repo_id="x/victim", path=outside))
    assert (outside / "keep.txt").exists()


def test_remove_model_refuses_sibling_with_common_prefix(models_dir):
    evil = models_dir.parent / (models_dir.name + "-evil") / "org" / "name"
    evil.mkdir(parents=True)
    with pytest.raises(ModelError, match="Refusing to delete"):
        registry.remove_model(LocalModel(repo_id="org/name", path=evil))
    assert evil.exists()


def test_remove_model_refuses_models_root_itself(models_dir, make_model):
    make_model("org/name")
    with pytest.raises(ModelError, match="Refusing to delete"):
        registry.remove_model(LocalModel(repo_id="root", path=models_dir))
    assert (models_dir / "org" / "name" / "config.json").exists()


# =========================================================================== pull_model (mocked hub)
class _Sibling:
    def __init__(self, rfilename: str, size: int) -> None:
        self.rfilename = rfilename
        self.size = size


class _Info:
    def __init__(self, files: dict[str, int], sha: str) -> None:
        self.siblings = [_Sibling(n, s) for n, s in files.items()]
        self.sha = sha


class FakeHub:
    """State shared between the fake HfApi / snapshot_download and the test."""

    def __init__(self) -> None:
        self.files: dict[str, int] = {
            "config.json": 700,
            "model.safetensors": 5000,
            "pytorch_model.bin": 5000,
            "README.md": 300,
            "tokenizer.json": 1200,
        }
        self.sha = "0123456789abcdef0123456789abcdef01234567"
        self.error: Exception | None = None  # raised by model_info when set
        self.info_calls: list[dict] = []
        self.download_calls: list[dict] = []
        self.progress_bars_disabled = 0


@pytest.fixture
def hub(monkeypatch, models_dir) -> FakeHub:
    """Replace huggingface_hub.HfApi / snapshot_download (imported inside pull_model)."""
    import huggingface_hub
    import huggingface_hub.utils as hub_utils

    state = FakeHub()

    class FakeHfApi:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def model_info(self, repo_id, revision=None, files_metadata=False, **kwargs):
            state.info_calls.append({"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata})
            if state.error is not None:
                raise state.error
            return _Info(state.files, state.sha)

    def fake_snapshot_download(*, repo_id, local_dir, revision=None, ignore_patterns=None,
                               force_download=False, **kwargs):
        patterns = list(ignore_patterns or [])
        state.download_calls.append({
            "repo_id": repo_id, "revision": revision, "local_dir": local_dir,
            "ignore_patterns": patterns, "force_download": force_download, "extra": kwargs,
        })
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name, size in state.files.items():
            if not registry._matches_any(name, patterns):
                (target / name).write_bytes(b"x" * size)
        return str(target)

    def fake_disable_progress_bars(*args, **kwargs):
        state.progress_bars_disabled += 1

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub_utils, "disable_progress_bars", fake_disable_progress_bars)
    return state


def _hub_error(cls, message: str, status: int = 404):
    """huggingface_hub >= 1.0 requires a keyword-only httpx.Response."""
    import httpx

    request = httpx.Request("GET", "https://huggingface.co/api/models/org/name")
    return cls(message, response=httpx.Response(status, request=request))


def test_pull_model_downloads_into_models_dir_and_writes_manifest(hub, models_dir, capsys):
    model = registry.pull_model(REPO)

    target = models_dir / "IFM" / "K2-Horizon-0.9B"
    assert model.path == target
    assert target.is_dir()

    [call] = hub.download_calls
    assert call["repo_id"] == REPO
    assert call["revision"] is None
    assert Path(call["local_dir"]) == target
    assert Path(call["local_dir"]).is_relative_to(models_dir)
    assert "*.bin" in call["ignore_patterns"]  # safetensors present -> legacy weights skipped
    assert set(config.LEGACY_WEIGHT_PATTERNS) <= set(call["ignore_patterns"])
    assert set(config.DEFAULT_IGNORE_PATTERNS) <= set(call["ignore_patterns"])
    assert call["force_download"] is False
    assert hub.info_calls == [{"repo_id": REPO, "revision": None, "files_metadata": True}]

    expected_size = 700 + 5000 + 300 + 1200  # everything except pytorch_model.bin
    assert not (target / "pytorch_model.bin").exists()
    assert model.files == sorted(["README.md", "config.json", "model.safetensors", "tokenizer.json"])
    assert model.size_bytes == expected_size
    assert model.revision == hub.sha

    manifest = json.loads((target / config.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["repo_id"] == REPO
    assert manifest["revision"] == hub.sha
    assert manifest["size_bytes"] == expected_size
    assert manifest["path"] == str(target)
    assert datetime.fromisoformat(manifest["downloaded_at"]).tzinfo is not None

    out = capsys.readouterr().out
    assert f"Pulling {REPO}  (4 files, 7.0 KB) -> {target}" in out
    assert f"Done: {REPO} (7.0 KB)" in out

    # the pulled model is now visible to list/find, with manifest values
    found = registry.find_model(REPO)
    assert found.size_bytes == expected_size
    assert found.revision == hub.sha
    assert [m.repo_id for m in registry.list_models()] == [REPO]


def test_pull_model_keeps_bin_when_repo_has_no_safetensors(hub, models_dir):
    hub.files = {"config.json": 10, "pytorch_model.bin": 50, "model.onnx": 99}
    model = registry.pull_model("org/legacy", quiet=True)
    [call] = hub.download_calls
    assert "*.bin" not in call["ignore_patterns"]
    assert not (set(config.LEGACY_WEIGHT_PATTERNS) & set(call["ignore_patterns"]))
    assert "*.onnx" in call["ignore_patterns"]
    assert (models_dir / "org" / "legacy" / "pytorch_model.bin").exists()
    assert not (models_dir / "org" / "legacy" / "model.onnx").exists()
    assert model.size_bytes == 60
    assert model.to_api()["details"]["format"] == "pytorch"


def test_pull_model_quiet_prints_nothing_and_disables_progress_bars(hub, capsys):
    registry.pull_model("org/name", quiet=True)
    assert capsys.readouterr().out == ""
    assert hub.progress_bars_disabled == 1


def test_pull_model_not_quiet_keeps_progress_bars(hub, capsys):
    registry.pull_model("org/name")
    assert hub.progress_bars_disabled == 0
    assert "Pulling org/name" in capsys.readouterr().out


def test_pull_model_revision_from_ref(hub):
    registry.pull_model("org/name@v1", quiet=True)
    assert hub.info_calls[-1]["revision"] == "v1"
    assert hub.download_calls[-1]["revision"] == "v1"


def test_pull_model_explicit_revision_overrides_ref(hub):
    registry.pull_model("org/name@v1", revision="v2", quiet=True)
    assert hub.info_calls[-1]["revision"] == "v2"
    assert hub.download_calls[-1]["revision"] == "v2"


def test_pull_model_accepts_url_ref(hub, models_dir):
    model = registry.pull_model("https://huggingface.co/org/name/tree/dev", quiet=True)
    assert hub.info_calls[-1] == {"repo_id": "org/name", "revision": "dev", "files_metadata": True}
    assert model.path == models_dir / "org" / "name"
    assert model.repo_id == "org/name"


def test_pull_model_force_download(hub):
    registry.pull_model("org/name", quiet=True, force=True)
    assert hub.download_calls[-1]["force_download"] is True


def test_pull_model_single_segment_repo(hub, models_dir):
    model = registry.pull_model("gpt2", quiet=True)
    assert model.path == models_dir / "gpt2"
    assert (models_dir / "gpt2" / config.MANIFEST_NAME).exists()


def test_pull_model_invalid_ref_fails_before_hub(hub):
    with pytest.raises(ModelError, match="Not a HuggingFace URL"):
        registry.pull_model("https://github.com/org/name")
    assert hub.info_calls == []
    assert hub.download_calls == []


def test_pull_model_without_config_json_raises_and_downloads_nothing(hub, models_dir):
    hub.files = {"model.gguf": 100, "README.md": 5}
    with pytest.raises(ModelError, match="no config.json") as excinfo:
        registry.pull_model("org/name", quiet=True)
    assert excinfo.type is ModelError
    assert hub.download_calls == []
    assert not (models_dir / "org" / "name").exists()


def test_pull_model_repository_not_found(hub):
    from huggingface_hub.errors import RepositoryNotFoundError

    hub.error = _hub_error(RepositoryNotFoundError, "404 Client Error: Repository Not Found")
    with pytest.raises(ModelNotFound, match="'org/missing' was not found") as excinfo:
        registry.pull_model("org/missing", quiet=True)
    assert excinfo.value.__cause__ is hub.error
    assert hub.download_calls == []


def test_pull_model_revision_not_found(hub):
    from huggingface_hub.errors import RevisionNotFoundError

    hub.error = _hub_error(RevisionNotFoundError, "404 Client Error: Revision Not Found")
    with pytest.raises(ModelError, match="Revision 'nope' does not exist in 'org/name'") as excinfo:
        registry.pull_model("org/name@nope", quiet=True)
    assert excinfo.type is ModelError
    assert hub.download_calls == []


def test_pull_model_generic_hub_error(hub):
    from huggingface_hub.errors import HfHubHTTPError

    hub.error = _hub_error(HfHubHTTPError, "500 Server Error", status=500)
    with pytest.raises(ModelError, match="HuggingFace Hub error for 'org/name'") as excinfo:
        registry.pull_model("org/name", quiet=True)
    assert excinfo.type is ModelError
    assert "500 Server Error" in str(excinfo.value)


def test_pull_model_gated_repo_reports_gated_not_missing(hub):
    from huggingface_hub.errors import GatedRepoError

    hub.error = _hub_error(GatedRepoError, "403 Client Error: Gated repository", status=403)
    with pytest.raises(ModelError) as excinfo:
        registry.pull_model("org/gated", quiet=True)
    assert not isinstance(excinfo.value, ModelNotFound)
    assert "gated" in str(excinfo.value)
    assert "HF_TOKEN" in str(excinfo.value)


# =========================================================================== ignore-pattern helpers
def test_select_ignore_patterns_with_and_without_safetensors():
    with_st = registry._select_ignore_patterns(["config.json", "model-00001-of-00002.safetensors"])
    without = registry._select_ignore_patterns(["config.json", "pytorch_model.bin"])
    assert with_st == config.DEFAULT_IGNORE_PATTERNS + config.LEGACY_WEIGHT_PATTERNS
    assert without == config.DEFAULT_IGNORE_PATTERNS


@pytest.mark.parametrize("name, expected", [
    ("model.gguf", True), ("onnx/model.onnx", True), ("pytorch_model.bin", False),
    ("config.json", False), ("model.safetensors", False),
])
def test_matches_any_default_patterns(name, expected):
    assert registry._matches_any(name, config.DEFAULT_IGNORE_PATTERNS) is expected


# --------------------------------------------------------------------------- hub_download_plan
# The GUI's "Check fit" asks what a pull would cost before paying for it, so the
# figure has to match what pull_model actually downloads - same ignore patterns.
def test_hub_download_plan_counts_only_what_pull_would_fetch(hub, models_dir):
    plan = registry.hub_download_plan("org/name")

    assert plan["repo_id"] == "org/name"
    assert plan["servable"] is True
    assert plan["revision"] == hub.sha
    # pytorch_model.bin is dropped because the repo also ships safetensors.
    assert plan["files"] == 4
    assert plan["total_bytes"] == 700 + 5000 + 300 + 1200
    assert hub.info_calls[-1]["files_metadata"] is True
    # Nothing was downloaded and no folder was created: this is a question, not a pull.
    assert hub.download_calls == []
    assert not (models_dir / "org" / "name").exists()


def test_hub_download_plan_ignores_other_runtime_weights(hub):
    hub.files = {"README.md": 100, "model.gguf": 4_000_000, "model.onnx": 2_000_000}

    plan = registry.hub_download_plan("org/gguf-only")

    # No config.json: transformers cannot serve it, and the GUI says so instead of pulling.
    assert plan["servable"] is False
    assert plan["files"] == 1
    assert plan["total_bytes"] == 100


def test_hub_download_plan_keeps_bin_weights_when_there_are_no_safetensors(hub):
    hub.files = {"config.json": 700, "pytorch_model.bin": 5000}

    plan = registry.hub_download_plan("org/legacy")

    assert plan["files"] == 2
    assert plan["total_bytes"] == 5700


def test_hub_download_plan_accepts_url_ref_with_revision(hub):
    registry.hub_download_plan("https://huggingface.co/org/name/tree/dev")

    assert hub.info_calls[-1] == {"repo_id": "org/name", "revision": "dev", "files_metadata": True}


def test_hub_download_plan_repository_not_found(hub):
    from huggingface_hub.errors import RepositoryNotFoundError

    hub.error = _hub_error(RepositoryNotFoundError, "404 Client Error: Repository Not Found")
    with pytest.raises(ModelNotFound, match="'org/missing' was not found"):
        registry.hub_download_plan("org/missing")


def test_hub_download_plan_gated_repo_explains_the_token(hub):
    from huggingface_hub.errors import GatedRepoError

    hub.error = _hub_error(GatedRepoError, "403 Client Error: Gated repository", status=403)
    with pytest.raises(ModelError, match="HF_TOKEN"):
        registry.hub_download_plan("org/gated")
