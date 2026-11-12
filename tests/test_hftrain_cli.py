"""hftrain.cli: argument parsing and the commands that need neither a model nor the network."""
from __future__ import annotations

import json

from hftrain import cli, config as tconfig
from hftrain.train import list_runs, read_run, write_run, adapter_dir, latest_checkpoint, TrainError


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "prepare" in capsys.readouterr().out


def test_prepare_unknown_source_is_an_error(train_dir, capsys):
    assert cli.main(["prepare", "--name", "x", "--source", "does-not-exist"]) == 1
    assert "Error:" in capsys.readouterr().out


def test_data_and_runs_when_empty(train_dir, capsys):
    assert cli.main(["data"]) == 0
    assert cli.main(["runs"]) == 0
    out = capsys.readouterr().out
    assert "No corpora" in out and "No runs" in out


def test_runs_table(train_dir, capsys):
    write_run("r1", {"run": "r1", "status": "finished", "model": "IFM/K2", "data": "hr-v1", "steps": 120,
                     "train_loss": 1.2345, "eval_loss_start": 2.5, "eval_loss_end": 1.9})
    write_run("r2", {"run": "r2", "status": "running", "model": "IFM/K2", "data": "hr-v1"})
    assert cli.main(["runs"]) == 0
    out = capsys.readouterr().out
    assert "r1" in out and "2.500->1.900" in out and "1.234" in out and "running" in out
    assert [r["run"] for r in list_runs()] == ["r1", "r2"]
    assert read_run("r1")["steps"] == 120


def test_adapter_dir_falls_back_to_latest_checkpoint(train_dir):
    write_run("r3", {"run": "r3", "status": "interrupted"})
    try:
        adapter_dir("r3")
    except TrainError:
        pass
    else:
        raise AssertionError("expected TrainError")
    cps = tconfig.run_dir("r3") / "checkpoints"
    for step in (100, 250, 50):
        d = cps / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert latest_checkpoint("r3").name == "checkpoint-250"
    assert adapter_dir("r3").name == "checkpoint-250"
    final = tconfig.run_dir("r3") / "adapter"
    final.mkdir()
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_dir("r3") == final


def test_train_parser_defaults():
    args = cli.build_parser().parse_args(["train", "--data", "d", "--run", "r"])
    assert args.seq_len == tconfig.DEFAULT_SEQ_LEN and args.lora_r == tconfig.DEFAULT_LORA_R
    assert args.target_modules.split(",") == list(tconfig.DEFAULT_TARGET_MODULES)
    assert args.max_steps == -1 and not args.resume and not args.train_embeddings
    args = cli.build_parser().parse_args(["eval", "--run", "r", "--compare-base", "--prompt", "Bok"])
    assert args.compare_base and args.prompt == ["Bok"] and args.temperature == 0.6 and args.seed == 1
    args = cli.build_parser().parse_args(["export", "--run", "r", "--name", "local/x"])
    assert args.dtype == "bfloat16" and not args.force


def test_eval_load_prompts_file(tmp_path):
    from hftrain.evaluate import load_prompts

    f = tmp_path / "p.txt"
    f.write_text("completion: Zagreb je\nNapiši pjesmu.\n\n" + json.dumps({"kind": "chat", "text": "Bok", "system": "S"}) + "\n",
                 encoding="utf-8")
    prompts = load_prompts(f, ["extra"])
    assert prompts[0] == {"kind": "completion", "text": "Zagreb je"}
    assert prompts[1] == {"kind": "chat", "text": "Napiši pjesmu."}
    assert prompts[2]["system"] == "S" and prompts[3] == {"kind": "chat", "text": "extra"}
    assert load_prompts(None, []) == tconfig.EVAL_PROMPTS
