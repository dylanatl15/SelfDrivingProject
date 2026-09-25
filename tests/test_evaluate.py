"""The evaluation scorecard, including the number the trainer keeps its best model by."""

import numpy as np

from selfdrive.eval.evaluate import ConstantPolicy, run_episodes


class ScriptedEnv:
    """Ends every episode on its first step, with the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def reset(self, seed=None, options=None):
        return np.zeros(2, np.float32), {}

    def step(self, action):
        m = self.outcomes.pop(0)
        crashed = bool(m["collided"])
        return np.zeros(2, np.float32), 0.0, crashed, not crashed, {"episode_metrics": m}


def test_clean_coverage_counts_crashed_and_stuck_episodes_as_zero():
    # A policy that covers a lot of floor and then crashes must not outscore one that
    # covers less and never fails: that is what the best model is chosen by.
    outcomes = [
        {"coverage_m2": 10.0, "collided": 1.0, "stuck": 0.0},
        {"coverage_m2": 8.0, "collided": 0.0, "stuck": 1.0},
        {"coverage_m2": 6.0, "collided": 0.0, "stuck": 0.0},
        {"coverage_m2": 2.0, "collided": 0.0, "stuck": 0.0},
    ]
    result = run_episodes(ConstantPolicy(0.0, 1.0), ScriptedEnv(outcomes), n_episodes=4)
    assert result.mean_coverage_m2 == 6.5
    assert result.clean_coverage_m2 == 2.0
    assert result.success_rate == 0.5
