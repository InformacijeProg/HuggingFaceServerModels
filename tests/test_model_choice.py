"""The models folder comes from .env, and the GUI asks which model to use.

Two things are covered here, both without Tk, network or the real models folder:

* ``hfserve.config``: parsing .env, the precedence between a real environment
  variable and .env, relative paths resolved against HOME_DIR, the loud error for
  a models folder that is not there, and the remembered last model.
* ``hfserve.picker``: the pure helpers behind the startup window - reading VRAM
  out of nvidia-smi output and turning a download size into a fits/too-big verdict.

The picker's widgets are not exercised: a Tk window needs a display, and the logic
worth testing is in these helpers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hfserve import config

picker = pytest.importorskip("hfserve.picker", reason="tkinter is not available")


# --------------------------------------------------------------------------- .env parsing
def test_load_env_file_reads_plain_keys(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("MODELS_DIR=./models\nHFSERVE_PORT=11435\n", encoding="utf-8")

    assert config._load_env_file(env) == {"MODELS_DIR": "./models", "HFSERVE_PORT": "11435"}


def test_load_env_file_skips_comments_and_blank_lines(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\n  \nMODELS_DIR=D:/models\n# MODELS_DIR=ignored\n",
                   encoding="utf-8")

    assert config._load_env_file(env) == {"MODELS_DIR": "D:/models"}


@pytest.mark.parametrize("line, expected", [
    ('MODELS_DIR="./models"', "./models"),
    ("MODELS_DIR='./models'", "./models"),
    ("export MODELS_DIR=./models", "./models"),
    ("MODELS_DIR =  ./models  ", "./models"),
])
def test_load_env_file_strips_quotes_export_and_spaces(tmp_path: Path, line, expected):
    env = tmp_path / ".env"
    env.write_text(line + "\n", encoding="utf-8")

    assert config._load_env_file(env) == {"MODELS_DIR": expected}


def test_load_env_file_keeps_a_value_containing_an_equals_sign(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("TOKEN=abc=def=ghi\n", encoding="utf-8")

    assert config._load_env_file(env) == {"TOKEN": "abc=def=ghi"}


def test_load_env_file_tolerates_a_bom(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("MODELS_DIR=./models\n", encoding="utf-8-sig")

    assert config._load_env_file(env) == {"MODELS_DIR": "./models"}


def test_load_env_file_missing_file_is_empty(tmp_path: Path):
    assert config._load_env_file(tmp_path / "nope.env") == {}


def test_load_env_file_ignores_a_line_without_an_equals_sign(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("this is not a setting\nMODELS_DIR=./models\n", encoding="utf-8")

    assert config._load_env_file(env) == {"MODELS_DIR": "./models"}


# --------------------------------------------------------------------------- precedence
def test_setting_prefers_a_real_environment_variable_over_dotenv(monkeypatch):
    monkeypatch.setattr(config, "DOTENV", {"MODELS_DIR": "from-dotenv"})
    monkeypatch.setenv("HFSERVE_MODELS", "from-environment")

    value, source = config._setting("HFSERVE_MODELS", "MODELS_DIR")

    assert value == "from-environment"
    assert "environment variable HFSERVE_MODELS" in source


def test_setting_falls_back_to_dotenv(monkeypatch):
    monkeypatch.setattr(config, "DOTENV", {"MODELS_DIR": "./models"})
    monkeypatch.delenv("HFSERVE_MODELS", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)

    value, source = config._setting("HFSERVE_MODELS", "MODELS_DIR")

    assert value == "./models"
    assert "MODELS_DIR in" in source  # the message points at the .env file


def test_setting_prefers_the_documented_name_within_dotenv(monkeypatch):
    monkeypatch.setattr(config, "DOTENV", {"MODELS_DIR": "plain", "HFSERVE_MODELS": "documented"})
    monkeypatch.delenv("HFSERVE_MODELS", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)

    assert config._setting("HFSERVE_MODELS", "MODELS_DIR")[0] == "documented"


def test_setting_ignores_an_empty_value(monkeypatch):
    monkeypatch.setattr(config, "DOTENV", {"MODELS_DIR": "   "})
    monkeypatch.delenv("HFSERVE_MODELS", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)

    value, source = config._setting("HFSERVE_MODELS", "MODELS_DIR")

    assert value is None
    assert source == "built-in default"


# --------------------------------------------------------------------------- path resolution
def test_as_dir_resolves_a_relative_value_against_the_base_not_the_cwd(tmp_path: Path):
    # "./models" in .env must mean "next to the project", wherever the shell happens to be.
    assert config._as_dir("./models", tmp_path) == (tmp_path / "models").resolve()


def test_as_dir_keeps_an_absolute_value(tmp_path: Path):
    absolute = (tmp_path / "elsewhere").resolve()

    assert config._as_dir(str(absolute), tmp_path / "ignored") == absolute


def test_as_dir_expands_a_home_relative_value(tmp_path: Path):
    assert config._as_dir("~/models", tmp_path) == (Path.home() / "models").resolve()


# --------------------------------------------------------------------------- fail loud
def test_require_models_dir_returns_the_folder_when_it_exists(models_dir: Path):
    assert config.require_models_dir() == models_dir


def test_require_models_dir_names_the_setting_when_the_folder_is_missing(state_dir, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR_SOURCE", "MODELS_DIR in D:/project/.env")

    with pytest.raises(config.ConfigError) as excinfo:
        config.require_models_dir()  # state_dir patched MODELS_DIR but never created it

    message = str(excinfo.value)
    assert "does not exist" in message
    assert "MODELS_DIR in D:/project/.env" in message  # tells the user what to fix
    assert str(config.MODELS_DIR) in message


def test_require_models_dir_rejects_a_file(state_dir, monkeypatch, tmp_path: Path):
    not_a_dir = tmp_path / "models.txt"
    not_a_dir.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "MODELS_DIR", not_a_dir)

    with pytest.raises(config.ConfigError, match="not a folder"):
        config.require_models_dir()


# --------------------------------------------------------------------------- last model
def test_last_model_round_trip(state_dir: Path, monkeypatch):
    monkeypatch.setattr(config, "LAST_MODEL_FILE", state_dir / "last_model.json")

    assert config.read_last_model() is None  # nothing remembered yet

    config.write_last_model("utter-project/EuroLLM-1.7B-Instruct")

    assert config.read_last_model() == "utter-project/EuroLLM-1.7B-Instruct"


def test_write_last_model_ignores_an_empty_name(state_dir: Path, monkeypatch):
    target = state_dir / "last_model.json"
    monkeypatch.setattr(config, "LAST_MODEL_FILE", target)

    config.write_last_model("")

    assert not target.exists()


def test_read_last_model_survives_a_corrupt_file(state_dir: Path, monkeypatch):
    target = state_dir / "last_model.json"
    monkeypatch.setattr(config, "LAST_MODEL_FILE", target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    assert config.read_last_model() is None  # a convenience file must never be fatal


# --------------------------------------------------------------------------- GPU detection
def _fake_run(stdout: str, returncode: int = 0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr="")
    return run


def test_gpu_vram_gb_reads_name_and_total(monkeypatch):
    monkeypatch.setattr(picker.subprocess, "run",
                        _fake_run("NVIDIA GeForce RTX 2070 SUPER, 8192\n"))

    description, vram = picker.gpu_vram_gb()

    assert vram == pytest.approx(8.0)
    assert "RTX 2070 SUPER" in description
    assert "8.0 GB" in description


def test_gpu_vram_gb_without_nvidia_smi(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(picker.subprocess, "run", boom)

    description, vram = picker.gpu_vram_gb()

    assert vram is None
    assert "CPU" in description


def test_gpu_vram_gb_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(picker.subprocess, "run", _fake_run("", returncode=9))

    assert picker.gpu_vram_gb()[1] is None


def test_gpu_vram_gb_with_unparsable_memory(monkeypatch):
    monkeypatch.setattr(picker.subprocess, "run", _fake_run("Some GPU, [N/A]\n"))

    description, vram = picker.gpu_vram_gb()

    assert vram is None
    assert "Some GPU" in description


# --------------------------------------------------------------------------- fit verdict
GB = 1024 ** 3


@pytest.mark.parametrize("weights_gb, expected", [
    (3.1, "fits comfortably"),   # EuroLLM-1.7B on an 8 GB card
    (6.5, "tight"),              # a 3B model: fits, but nothing else on the GPU
    (15.0, "too big"),           # a 7B model in float16
])
def test_fit_verdict_against_an_8gb_card(weights_gb, expected):
    assert expected in picker.fit_verdict(int(weights_gb * GB), 8.0)


def test_fit_verdict_without_a_gpu_talks_about_system_ram():
    verdict = picker.fit_verdict(3 * GB, None)

    assert "CPU" in verdict
    assert "RAM" in verdict


def test_fit_verdict_reports_the_size_it_judged():
    assert "3.1 GB" in picker.fit_verdict(int(3.1 * GB), 8.0)


# --------------------------------------------------------------------------- server address
@pytest.mark.parametrize("value, expected", [
    ("http://127.0.0.1:11435", ("127.0.0.1", 11435)),
    ("https://box.lan:8000", ("box.lan", 8000)),
    ("127.0.0.1:11435", ("127.0.0.1", 11435)),
    (":11435", (None, 11435)),
    ("  http://127.0.0.1:11435/  ", ("127.0.0.1", 11435)),
    ("127.0.0.1", ("127.0.0.1", None)),
])
def test_split_server_url(value, expected):
    assert config._split_server_url(value) == expected


def test_split_server_url_with_a_bad_port_keeps_the_host():
    # urlsplit raises on ".port" here; the host is still usable and the port falls back.
    assert config._split_server_url("http://127.0.0.1:not-a-port") == ("127.0.0.1", None)


def test_load_env_file_reads_the_server_address(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("HFSERVE_HOST=127.0.0.1\nHFSERVE_PORT=11435\n", encoding="utf-8")

    assert config._load_env_file(env) == {"HFSERVE_HOST": "127.0.0.1", "HFSERVE_PORT": "11435"}


def test_load_env_file_drops_a_trailing_inline_comment(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("HFSERVE_PORT=11435  # 11434 stays Ollama's\n", encoding="utf-8")

    assert config._load_env_file(env) == {"HFSERVE_PORT": "11435"}


def test_load_env_file_keeps_a_hash_inside_a_quoted_value(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text('MODELS_DIR="./models # really"\n', encoding="utf-8")

    assert config._load_env_file(env) == {"MODELS_DIR": "./models # really"}


def test_dotenv_only_keeps_the_keys_hfserve_consumes():
    # A token or any other secret in .env must not be held in module memory.
    assert set(config.DOTENV) <= set(config.DOTENV_KEYS)
    assert "HF_TOKEN" not in config.DOTENV_KEYS


def test_the_configured_port_is_usable():
    assert 1 <= config.DEFAULT_PORT <= 65535
    assert config.server_url() == f"http://{config.DEFAULT_HOST}:{config.DEFAULT_PORT}"
    assert config.SETTINGS_PROBLEMS == []  # the project's own .env must be readable


def test_suggested_models_are_well_formed():
    # Every suggestion must be a pullable "org/name" reference with a note beside it.
    assert picker.SUGGESTED
    for repo_id, note in picker.SUGGESTED:
        assert repo_id.count("/") == 1, repo_id
        assert not repo_id.startswith("http"), repo_id
        assert note.strip()
