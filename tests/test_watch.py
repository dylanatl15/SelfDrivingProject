"""Checkpoint discovery for the live viewer. No window, no torch."""

import os
import time

from selfdrive.eval.watch import FINAL, SETTLE_S, newest_checkpoint, newest_run


def _touch(path, age_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_ranks_by_step_count_not_name_or_mtime(tmp_path):
    # "ppo_10000000" < "ppo_2000000" as strings, and the 10M file is also older here.
    _touch(tmp_path / "checkpoints" / "ppo_2000000_steps.zip", age_s=50)
    _touch(tmp_path / "checkpoints" / "ppo_10000000_steps.zip", age_s=100)
    steps, path = newest_checkpoint(tmp_path)
    assert steps == 10_000_000
    assert path.name == "ppo_10000000_steps.zip"


def test_skips_a_checkpoint_still_being_written(tmp_path):
    _touch(tmp_path / "checkpoints" / "ppo_500000_steps.zip", age_s=100)
    _touch(tmp_path / "checkpoints" / "ppo_1000000_steps.zip", age_s=0)
    assert newest_checkpoint(tmp_path)[0] == 500_000
    assert newest_checkpoint(tmp_path, now=time.time() + SETTLE_S + 1)[0] == 1_000_000


def test_final_model_outranks_checkpoints(tmp_path):
    _touch(tmp_path / "checkpoints" / "ppo_20000000_steps.zip", age_s=100)
    _touch(tmp_path / "final_model.zip", age_s=100)
    assert newest_checkpoint(tmp_path)[0] == FINAL


def test_a_resumed_run_passes_its_old_final_model(tmp_path):
    _touch(tmp_path / "final_model.zip", age_s=3600)  # left behind when the run was stopped
    _touch(tmp_path / "checkpoints" / "ppo_1000000_steps.zip", age_s=3700)
    _touch(tmp_path / "checkpoints" / "ppo_4000000_steps.zip", age_s=100)
    steps, path = newest_checkpoint(tmp_path)
    assert steps == 4_000_000
    assert path.name == "ppo_4000000_steps.zip"


def test_nothing_yet(tmp_path):
    assert newest_checkpoint(tmp_path) is None
    assert newest_run(tmp_path / "missing") is None


def test_newest_run_uses_start_time(tmp_path):
    _touch(tmp_path / "ppo_b" / "train_config.json", age_s=500)
    _touch(tmp_path / "ppo_a" / "train_config.json", age_s=10)
    (tmp_path / "not_a_run").mkdir()
    assert newest_run(tmp_path).name == "ppo_a"
