"""The patience clock (`GoalPatience`, `obs.goal_patience`).

waypoint_pay5 shuffled back and forth in dead ends, because backing out of one lost range
and nothing in its input changed while it dithered. The clock gives it an input that does
change: seconds since the range to the goal last fell to a new low. These tests pin that
dithering never restarts it, real progress does, and that the phone-side rule in
`docs/goal-block.md` is what the environment computes.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from selfdrive.config import load_env_config, load_yaml
from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.envs.goals import GoalPatience
from selfdrive.envs.obs import ObsConfig, ObservationBuilder
from selfdrive.envs.randomize import DomainRandConfig
from selfdrive.world.generators import ArenaParams
from selfdrive.world.geometry import World

DT = 1.0 / 30.0
FIELD = World(segments=[[0, 0, 30, 0], [30, 0, 30, 30], [30, 30, 0, 30], [0, 30, 0, 0]],
              bounds=(0.0, 0.0, 30.0, 30.0))


def test_the_clock_counts_while_the_car_gets_no_closer():
    clock = GoalPatience(gain=0.25)
    clock.reset(5.0)
    for _ in range(30):
        clock.update(5.0, DT)
    assert clock.seconds == pytest.approx(1.0)
    clock.update(5.6, DT)  # backing away counts too
    assert clock.seconds == pytest.approx(1.0 + DT)


def test_the_clock_restarts_once_the_range_falls_by_the_gain():
    clock = GoalPatience(gain=0.25)
    clock.reset(5.0)
    clock.update(4.9, DT)
    clock.update(4.8, DT)
    assert clock.seconds == pytest.approx(2 * DT)
    clock.update(4.75, DT)
    assert clock.seconds == 0.0
    clock.update(4.6, DT)  # the new best is 4.75, so 0.15 m closer is not enough
    assert clock.seconds == pytest.approx(DT)


def test_dithering_back_and_forth_never_restarts_the_clock():
    clock = GoalPatience(gain=0.25)
    clock.reset(5.0)
    for k in range(90):
        clock.update(4.8 if k % 2 else 5.3, DT)
    assert clock.seconds == pytest.approx(3.0)


def test_the_fourth_goal_float_puts_ten_seconds_at_the_top_of_the_scale():
    config = ObsConfig(goal_block=True, goal_patience=True)
    assert config.goal_size == 4
    assert config.size == 68 + 4
    assert ObsConfig(goal_patience=True).goal_size == 0
    builder = ObservationBuilder(config)
    waited = [builder.goal(3.0, 0.0, s)[3] for s in (0.0, 5.0, 10.0, 30.0)]
    assert waited == pytest.approx([-1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="seconds waited"):
        builder.goal(3.0, 0.0)


def patience_env(max_steps: int = 600) -> CarEnv:
    cfg = EnvConfig(obs=ObsConfig(goal_block=True, goal_patience=True),
                    arena=ArenaParams(kind="outdoor"),
                    domain_rand=DomainRandConfig(enabled=False), max_steps=max_steps)
    return CarEnv(cfg)


def run(env: CarEnv, actions, goal) -> tuple[list[float], list[bool]]:
    """The clock float after reset and after every step, and whether each step arrived."""
    obs, _ = env.reset(seed=0, options={"world": FIELD, "pose": (15.0, 15.0, 0.0), "goal": goal})
    waited, reached = [float(obs[-1])], []
    for action in actions:
        obs, _, terminated, truncated, info = env.step(np.array(action, np.float32))
        waited.append(float(obs[-1]))
        reached.append(bool(info.get("goal_reached")))
        if terminated or truncated:
            break
    return waited, reached


def test_a_parked_car_watches_its_clock_run():
    env = patience_env()
    waited, _ = run(env, [[0.0, 0.0]] * 60, goal=(22.0, 15.0))
    assert waited[0] == -1.0
    assert waited[60] == pytest.approx(60 * env.dt / 10.0 * 2 - 1, abs=1e-5)


def test_driving_at_the_goal_keeps_the_clock_near_zero():
    waited, _ = run(patience_env(), [[0.0, 0.6]] * 150, goal=(27.0, 15.0))
    seconds = (np.array(waited) + 1.0) / 2.0 * 10.0
    assert seconds[60:].max() < 0.5


def test_a_new_goal_restarts_the_clock():
    # Parked 0.6 m short of the goal for a second, then creeping in: arrival at 0.5 m comes
    # before the range has fallen the 0.25 m that would restart the clock by itself.
    waited, reached = run(patience_env(), [[0.0, 0.0]] * 30 + [[0.0, 0.3]] * 120,
                          goal=(15.6, 15.0))
    step = reached.index(True)
    assert waited[step] > -0.85  # at least 0.75 s on the clock the step before arriving
    assert waited[step + 1] == -1.0


def test_the_clock_needs_the_goal_block():
    with pytest.raises(ValueError, match="goal_block"):
        CarEnv(EnvConfig(obs=ObsConfig(goal_patience=True)))


def test_patience_env_passes_the_gymnasium_checker():
    env = patience_env()
    assert env.observation_space.shape == (72,)
    check_env(env, skip_render_check=True)


def test_waypoint_s2_patience_config_is_the_stage2_base_with_the_clock():
    base = load_yaml("configs/env_waypoint_s2.yaml")
    ours = load_yaml("configs/env_waypoint_s2_patience.yaml")
    added = ("goal_patience", "norm_patience_max", "patience_gain")
    assert {k: ours["obs"].pop(k) for k in added} == {
        "goal_patience": True, "norm_patience_max": 10.0, "patience_gain": 0.25}
    assert ours == base
    assert load_env_config("configs/env_waypoint_s2_patience.yaml").obs.size == 120
