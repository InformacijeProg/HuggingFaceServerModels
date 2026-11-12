"""hftrain.train.ConsoleProgress: the console progress bar, driven by fake Trainer state."""
from __future__ import annotations

from types import SimpleNamespace

from hftrain.train import ConsoleProgress


def _state(step=0, tokens=0, max_steps=100):
    return SimpleNamespace(global_step=step, max_steps=max_steps, num_input_tokens_seen=tokens,
                           is_world_process_zero=True)


def test_bar_follows_steps_and_reports(capsys):
    cb = ConsoleProgress()
    cb.on_step_begin(None, _state(), None)
    assert cb.bar is not None and cb.bar.total == 100
    for step in range(1, 11):
        cb.on_step_end(None, _state(step), None)
    assert cb.bar.n == 10
    cb._last = (cb._last[0] - 1.0, 0)  # the steps took a second, not one clock tick
    cb.on_log(None, _state(10, tokens=163_840), None, logs={"loss": 2.3456, "learning_rate": 1e-4})
    assert cb._post["loss"] == "2.346" and "tok/s" in cb._post
    cb.on_evaluate(None, _state(10), None, metrics={"eval_loss": 2.0})
    cb.on_evaluate(None, _state(20), None, metrics={"eval_loss": 1.9})
    cb.on_evaluate(None, _state(30), None, metrics={"eval_loss": 1.95})
    cb.on_save(None, _state(30), None)
    cb.on_train_end(None, _state(30), None)
    assert cb.bar is None
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "step 10: eval loss 2.0000, perplexity 7.39" in text
    assert "(new best)" in text and "(best 1.9000)" in text
    assert "checkpoint saved" in text


def test_resume_starts_bar_at_checkpoint_step():
    cb = ConsoleProgress()
    cb.on_step_begin(None, _state(step=400, tokens=6_553_600), None)
    assert cb.bar.n == 400
    cb.on_step_end(None, _state(step=401), None)
    assert cb.bar.n == 401
    cb.on_train_end(None, _state(401), None)


def test_eval_bar_before_training_and_log_without_bar():
    cb = ConsoleProgress()
    loader = [0] * 5
    for _ in loader:
        cb.on_prediction_step(None, _state(), None, eval_dataloader=loader)
    assert cb.eval_bar.n == 5
    cb.on_evaluate(None, _state(), None, metrics={"eval_loss": 3.9})
    assert cb.eval_bar is None
    cb.on_log(None, _state(), None, logs={"eval_loss": 3.9})  # no bar yet: ignored
