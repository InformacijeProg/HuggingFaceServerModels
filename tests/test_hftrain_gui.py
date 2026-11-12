"""hftrain.gui helpers (no window) and the train.py pieces the GUI relies on: progress.json and
the STOP file."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hftrain import config as tconfig, gui
from hftrain.train import ConsoleProgress, StopFile


def test_split_output_lines_and_redraws():
    parts, rest = gui.split_output("loading\r\nbar 1%\rbar 2%\r\x1b[Astep 5: eval\nhalf")
    assert parts == [("line", "loading"), ("live", "bar 1%"), ("live", "bar 2%"), ("line", "step 5: eval")]
    assert rest == "half"


def test_split_output_holds_trailing_carriage_return():
    parts, rest = gui.split_output("done\r")
    assert parts == [] and rest == "done\r"
    parts, rest = gui.split_output(rest + "\nnext")
    assert parts == [("line", "done")] and rest == "next"


def test_console_filters_bars_and_remote_code_noise():
    assert gui.NOISE_RE.search("evaluating:  50%|█████     | 2/4 [00:00<00:00,  3.56batch/s]")
    assert gui.NOISE_RE.search("training  57%|█████▋    | 8/14 [01:03 elapsed, 01:35 left], loss 3.144")
    assert gui.NOISE_RE.search("Loading weights: 100%|██████████| 255/255 [00:01<00:00, 182.11it/s]")
    assert gui.NOISE_RE.search("[ERROR] `cache_position` is part of K2HorizonModel.forward's signature")
    assert not gui.NOISE_RE.search("step 6: eval loss 2.8761, perplexity 17.74 (new best)")
    assert not gui.NOISE_RE.search("training gui-check stopped.")


def test_cli_gui_passes_form_and_start(monkeypatch):
    from hftrain import cli

    seen = {}
    monkeypatch.setattr(gui, "main", lambda form=None, start=False: seen.update(form=form, start=start) or 0)
    assert cli.main(["gui", "--data", "hr-v2", "--model", "local/K2-hr1", "--run", "hr-lora-3",
                     "--lora-r", "64", "--start"]) == 0
    assert seen == {"form": {"data": "hr-v2", "model": "local/K2-hr1", "run": "hr-lora-3", "lora_r": 64},
                    "start": True}


def test_prefill_accepts_every_cli_field():
    import inspect

    params = set(inspect.signature(gui.prefill).parameters) - {"app"}
    assert params == {"data", "run", "model", "lora_r", "lr", "epochs"}
    assert {"form", "start"} <= set(inspect.signature(gui.main).parameters)


def test_next_run_name():
    assert gui.next_run_name([]) == "qa-lora-1"
    assert gui.next_run_name(["qa-lora-1", "smoke-1", "qa-lora-3"]) == "qa-lora-4"
    assert gui.next_corpus_name(["qa-v1", "other"]) == "qa-v2"


def test_build_train_argv():
    argv = gui.build_train_argv("py", data="hr-v2", run="hr-lora-2", model="IFM/K2", lora_r=64, lr=2e-4,
                                epochs=1.0, resume=True)
    assert argv[:5] == ["py", "-u", "-m", "hftrain", "train"]
    assert argv[argv.index("--lora-alpha") + 1] == "128"
    assert argv[argv.index("--lr") + 1] == "0.0002" and argv[-1] == "--resume"


def test_progress_line():
    line = gui.progress_line({"step": 50, "max_steps": 200, "loss": 2.5, "eval_loss": 2.25, "tokens_per_s": 1400.4,
                              "vram_reserved_mb": 5427, "vram_total_mb": 8192, "elapsed_s": 3725, "remaining_s": 11175})
    assert line == ("step 50/200 (25.0%) | loss 2.500 | eval 2.250 | 1,400 tok/s | VRAM 5.3/8.0 GB | "
                    "1:02:05 elapsed, 3:06:15 left")


def test_gpu_status_line():
    line = gui.gpu_status_line("NVIDIA GeForce RTX 2070 SUPER, 97, 5427, 8192, 71, 205.33\n")
    assert line == ("GPU: NVIDIA GeForce RTX 2070 SUPER   |   load 97%   |   VRAM 5.3 / 8.0 GB (66%)   |   "
                    "71 °C   |   205 W")
    assert gui.gpu_status_line("Some GPU, 3, 100, 1024, 40, [N/A]").endswith("40 °C")
    assert gui.gpu_status_line("") == "GPU: no NVIDIA GPU found"


def test_list_runs_light_and_corpus_tokens(train_dir):
    rd = tconfig.run_dir("hr-lora-9")
    (rd / "checkpoints" / "checkpoint-200").mkdir(parents=True)
    (rd / "checkpoints" / "checkpoint-200" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (rd / "run.json").write_text(json.dumps({"run": "hr-lora-9", "status": "interrupted"}), encoding="utf-8")
    runs = gui.list_runs_light()
    assert [(r["run"], r["has_checkpoint"], r["has_adapter"]) for r in runs] == [("hr-lora-9", True, True)]
    corpus = {"tokenized": [{"seq_len": 512, "train": {"tokens": 1}}, {"seq_len": 1024, "train": {"tokens": 99}}]}
    assert gui.corpus_tokens(corpus) == 99 and gui.corpus_tokens({"tokenized": []}) is None


def _state(step=0, tokens=0, max_steps=100):
    return SimpleNamespace(global_step=step, max_steps=max_steps, num_input_tokens_seen=tokens,
                           is_world_process_zero=True)


def test_console_progress_writes_progress_file(tmp_path):
    pf = tmp_path / "progress.json"
    cb = ConsoleProgress(pf)
    cb.on_train_begin(None, _state(), None)
    assert json.loads(pf.read_text(encoding="utf-8"))["phase"] == "starting"
    cb.on_step_begin(None, _state(), None)
    cb._t_start -= 10  # ten seconds for ten steps
    cb._last = (cb._last[0] - 10, 0)
    for s in range(1, 11):
        cb.on_step_end(None, _state(s), None)
    cb.on_log(None, _state(10, tokens=16384), None, logs={"loss": 2.5})
    p = json.loads(pf.read_text(encoding="utf-8"))
    assert p["phase"] == "training" and p["step"] == 10 and p["loss"] == 2.5 and p["tokens_per_s"] > 0
    assert 80 <= p["remaining_s"] <= 100  # 90 steps left at about one second each
    cb.on_train_end(None, _state(10), None)
    assert json.loads(pf.read_text(encoding="utf-8"))["phase"] == "finished"


def test_stop_file_saves_then_interrupts(tmp_path):
    stop = tmp_path / "STOP"
    cb = StopFile(stop)
    control = SimpleNamespace(should_save=False)
    cb.on_step_end(None, _state(5), control)
    assert control.should_save is False
    stop.write_text("x", encoding="utf-8")
    cb.on_step_end(None, _state(6), control)
    assert control.should_save is True
    with pytest.raises(KeyboardInterrupt):
        cb.on_save(None, _state(6), control)
    assert not stop.exists()
