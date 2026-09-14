# Working notes for this repository

## What this is

The simulation and RL half of a senior project self-driving toy car. Read `README.md`
first, then `docs/protocol.md`.

## Ground rules

- **`docs/protocol.md` is frozen.** Two teammates are writing ESP32 firmware and an
  Android app against it without running this code. Changing the codec silently breaks
  their work with no compile error. Propose a v2; do not edit v1. The two example frames
  in that document are pinned byte-for-byte in `tests/test_protocol.py` so they cannot
  quietly rot.
- **`ObsConfig` normalization constants are a published interface.** They are mirrored
  by hand in the Android app. Changing one invalidates every trained checkpoint.
- **Never import pygame at module scope.** Twenty training workers must not load it.
  `CarEnv.render()` imports it lazily; keep it that way.
- **Rewards read ground truth, observations read noisy sensors.** Do not "simplify" the
  reward to use sensor readings — that trains the policy to chase its own noise.
- **Domain randomization must be drawn from `self.np_random`.** Anything sampled from a
  global RNG breaks seed reproducibility, which `tests/test_determinism.py` guards.

## Conventions

- SI units everywhere in code (metres, seconds, radians). YAML accepts `*_deg` aliases,
  converted on load in `config.py`.
- Steering positive = **left** (counter-clockwise). Throttle positive = **forward**.
  Same in the sim, the protocol and the firmware.
- Geometry is vectorized numpy. Adding a Python loop over rays or primitives to
  `world/geometry.py` is a throughput regression, not a style question — it is the hot
  path. Check with `python -m selfdrive.train.bench`.

## Tests worth understanding before changing the reward

`tests/test_rewards.py` pins the *ordering* the reward is meant to express: forward
driving must beat parking, tight circles, wide fast loops, a slow weave, shuffling in
place, and reversing. Wide loops are pinned because `phase1_v1` found them: a 2 s
displacement window paid a 1-2 m orbit almost as well as a straight line. The weave is
pinned because a per-cell coverage grid, which looks isotropic, paid it more than a line. If a reward tweak
breaks one of those, the tweak is wrong, not the test.

`phase1_v2` found the next hole. Paying only for new ground makes every lap after the
first free, and a collision costs as much as 100 m of new ground, so it circled open
patches at near-full steering lock. `w_retrace` charges for ground covered again: it is 0
in `env_phase1.yaml` and 0.5 in `env_phase1_retrace.yaml`. `phase1_v3b` did not show that
it helps, so it stays off. Best models are kept by `clean_coverage_m2`, not success rate,
because a car that circles never fails.

`phase1_v3a` and `phase1_v3b` circled again for a different reason: the action
distribution. The env clips Gaussian actions to [-1, 1], so extra std costs nothing.
`ent_coef` 0.004 let v2's std collapse to 0.06; 0.01 ran v3's past 3, with the steering
mean near |5|, and both ended at full lock. `train_ppo_v4.yaml` bounds actions with tanh
(squashed gSDE) instead. On a clipped-Gaussian run, watch `train/std` and the raw policy
mean. Under gSDE, `train/std` is the noise-matrix scale, not the action std.

`phase1_v5` and `phase1_mem_v2` then plateaued near 13 m² of clean coverage by 5M steps.
Three things there are easy to misread. The arenas are cramped: 59 % of reachable free
space lies within 1 m of an obstacle, so driving that close is not wall-hugging, and
v5's 20 % meant it was avoiding clutter. Coverage is capped by speed, not arena size: a
50 s episode sweeps at most ~25 m² per 1 m/s. And the eval's `reverse` column counts
negative throttle commands, which are mostly braking; `backing` counts the car rolling
backwards. mem_v2 held ~0.7 m/s by flipping the sign of its throttle 4.5 times a second,
and its steering between full locks as often. `env_phase1_smooth.yaml` charges changes on
both actuators (`w_throttle_oscillation`).

That charge is also a charge on exploration. gSDE resampled its noise every 4 steps, so the
commands changed on a quarter of all steps whatever the policy meant, and at 0.15 the
charge cost more than exploring paid. `phase1_v6` and `phase1_v6r` hid from it by pinning
both actuators at a limit, where tanh flattens the noise, and orbited at full lock; v6r did
it in reverse. Keep smoothness weights small next to explore pay, and resample the noise
rarely: `train_ppo_v6.yaml` uses `sde_sample_freq` 30, `env_phase1_light.yaml` 0.05 on
each actuator. The eval `lock` column counts forward driving only, so it misses reverse
orbits; check `backing`.

`env_phase1_light_big.yaml` is the same reward in arenas of 3.4 times the floor area. Wall
count scales with area there (`walls_per_100m2`) and wall length is absolute, so obstacle
density matches `env_phase1.yaml`. A 32-seed sample said a sparser layout matched; the
test uses 300.

Squashed policies also saturate by degrees. Every gen7 run held its pre-tanh throttle mean
at +2 to +3.6 with a wall under 1 m ahead, where every noise sample squashes to near-full
throttle: exploration almost never braked, so PPO had no advantage signal to learn braking,
and the crashes came at full lock and full throttle. The entropy bonus cannot pull a mean
back. Under squashed gSDE SB3 estimates entropy from buffer actions, and that only grows the
noise. `train_ppo_v7.yaml` sets `mean_penalty`, a quadratic charge on pre-tanh means past
1.5 (`train/policies.py`). It rides in the entropy term, so `train/entropy_loss` includes it;
`train/mean_penalty` and `train/pre_tanh_mean_abs` log it alone.

`configs/env_waypoint.yaml` drives to chained goals instead of exploring (`envs/goals.py`).
Its reward pays for path distance closed, on a 0.075 m raster (`world/navigation.py`). It
never pays for straight-line distance, which would reward pressing against the wall between
the car and its goal; `tests/test_goals.py` pins the U-trap case. Path distance, spawns and
goals all use the floor the car body can drive to (`world/reachability.py`): lattice poses
over cells and 16 headings, joined by turning-radius arcs and straight runs, forward or
reversing. No fixed inflation gets narrow gaps right. At 0.8 m it sealed a pocket the car
drives out of in a straight line (big-arena seed 2000001, where the policy just missed the
gap), and no width alone tells a doorway the car meets straight on from a corner it cannot
fit round. The observation gains three floats
(`docs/goal-block.md`, a change the Android app must mirror), and best models are kept by
`clean_goals`.

`goal.arrival_from_odometry` (`env_waypoint_pay5_pin.yaml`) judges goal arrival from the
odometry estimate. That is not the reward reading a sensor. The goal is a pin in the phone's
drifting frame, and the true distance from the car to where that pin now sits is exactly the
estimate's distance to the goal, since drift moves the frame rigidly. Progress still pays true
path distance to the goal. Without it, arrival waits for the true goal: in pay5's drift probe
the goal the policy saw sat a median 0.37 m from the real one, and beyond the 0.5 m arrival
radius on 30 % of steps.

The Stage 1 exam (100 held-out seeds from 3,000,000, full drift) kept that rule and the halved
heading drift range of `env_waypoint_pay5_yaw25.yaml`: each beat pay5 by 0.8-1.0 clean goals
at 3M. `env_waypoint_s2.yaml` carries both. The narrower drift is a guess until ARCore has
been measured; yaw25 models lose most on the widest drift draws. A floor on the gSDE noise
(`train_ppo_v8.yaml`) did not help. The 20-episode training eval disagreed with the exam by
0.35 clean goals, so pick checkpoints by the exam, not by `best_model`.

Two Stage 2 additions are off by default. `reward.w_brake` (`envs/braking.py`) charges the
share of the stopping distance, laid along the current arc, that the body cannot drive; on
pay5's driving it fired only in the 0.2-0.4 s before crashes. `obs.goal_patience` adds a
fourth goal float, seconds since the range to the goal last fell 0.25 m below its best, so a
car dithering in a dead end sees something change; the phone must compute it too
(`docs/goal-block.md`).

## Throughput ceiling - do not chase it

`python -m selfdrive.train.bench` reports ~900 steps/s single-process and ~4,800 across
20 workers on this machine. The ~26% per-worker efficiency is per-vec-step pipe latency
in the parent under WSL2, not CPU: two workers already sit at 56%, 28 workers beat 20 by
nothing, BLAS thread pinning changes nothing, and fork/forkserver/spawn agree within 3%.
Even a zero-cost env step would cap out near 5,700 steps/s. Optimizing geometry further
buys almost nothing; the only real fix would be batching many steps per IPC round trip.

That ceiling is per run. Trainers running side by side must pin torch threads
(`torch_threads` in the training YAML, default 4). Unpinned, each parent's torch took most
of the 28 threads for PPO updates and policy inference, starving the env workers, and
three parallel runs fell to 853 steps/s in total. Pinned, the same three ran at 3,894.

## Known placeholders

`configs/env_phase1.yaml` car geometry is a generic ~1/10 RC platform. Replace with
measured values once the chassis exists — `docs/sim2real.md` lists exactly what to
measure and how to narrow the randomization ranges afterwards.
