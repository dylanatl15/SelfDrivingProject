# Self-Driving Toy Car — Simulator and RL Training Stack

Senior project. A 1/10-scale Ackermann car that drives itself: a phone on the roof runs
ARCore depth and a neural network policy, an ESP32 turns the policy's commands into PWM,
and the car finds its way to dropped waypoints without hitting anything.

**This repository is the simulation and training half.** It is a custom Gymnasium
environment that models the real car's sensors and actuators closely enough that a policy
trained here transfers to hardware, plus the PPO training, evaluation and export around
it. The other half — Android app, ESP32 firmware, chassis — is built by teammates against
[`docs/protocol.md`](docs/protocol.md), which is frozen so that both halves can be written
at the same time.

Python 3.12 · PPO (Stable-Baselines3) · Gymnasium · vectorized NumPy geometry · 400 tests

---

## Where the project is

| Phase | Goal | Status |
|---|---|---|
| **1** | Drive, never crash, never stay stuck. No goals. | **done** — 19.1 m² of clean new ground per 50 s episode |
| **2** | Drive to chained waypoints dropped as ARCore pins. | **done in sim** — 2.3–2.7 goals per episode, 15–17 % crash |
| 3 | Pan-tilt camera servo → recurrent policy. | scaffolded, not built |

Everything above runs in simulation. The chassis does not exist yet, so the car geometry
in `configs/env_phase1.yaml` is a placeholder for a generic 1/10 RC platform;
[`docs/sim2real.md`](docs/sim2real.md) lists exactly what to measure once it does.

Trained weights are **not** in this repository — `runs/` is gitignored, and checkpoints run
to hundreds of megabytes. The code that produces them is all here.

---

## Results

### Phase 1 — explore without crashing

The reward pays per square metre of *new* floor swept. Best policy (`phase1_mem_v5`,
13.5M steps) covers **19.1 m² of clean new ground** in a 50 s episode. Coverage is capped
by speed, not by arena size: at 1 m/s a 50 s episode can sweep about 25 m² at the absolute
most, so 19.1 is close to the ceiling the speed limit allows.

### Phase 2 — drive to waypoints

Best policy (`waypoint_s2_patience_seed1`, 9.5M steps), scored on 100 held-out arena
seeds it never trained on, driving to chained goals:

| Setup | Goals reached cleanly | Crash | Stuck | Speed |
|---|---|---|---|---|
| Policy alone | 2.00 | 36 % | 0 % | 0.82 m/s |
| Policy + speed shield | 2.10 | 14 % | 15 % | 0.76 m/s |
| **Policy + shield + back-out** | **2.31** | **17 %** | **3 %** | 0.78 m/s |

The shield ([`docs/shield.md`](docs/shield.md)) is a small non-learned safety layer between
the policy and the car: it caps speed by the stopping distance the sensors can actually see.
It more than halves crashes on its own, but the cap can also pin the car against a wall
forever — hence the 15 % stuck. The back-out rule reverses when the cap has held the car
still for a second, which recovers almost all of that.

### The phone's own loop, checked against the simulator

The Android app has to rebuild all 120 policy inputs itself, from ARCore depth and pose
plus serial telemetry. Get one normalization constant or one sign wrong and the car drives
confidently into a wall, with nothing to catch it at compile time.

So [`link/phone.py`](src/selfdrive/link/phone.py) is a reference implementation of that
whole loop, and it drives the simulator *through the real serial codec* — the same `$C` and
`$T` frames the ESP32 will parse. Over 100 seeds, across four variants (with and without a
wheel encoder, measured and nominal steering limits), it scored between **2.61 and 2.89
clean goals against the simulator's own 2.70** — every difference inside one standard
error. The phone-side pipeline is not measurably worse than building the inputs inside the
environment. [`docs/phone-loop.md`](docs/phone-loop.md) is the
step-by-step port guide that came out of it, including the five places where the phone and
the simulator legitimately disagree.

This is the cheapest sim-to-real insurance in the project: it catches unit, sign and
field-order bugs on a desk, not on a moving car.

---

## Quickstart

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

Train. 20 worker processes, TensorBoard on, checkpoints every 500k steps:

```bash
uv run python -m selfdrive.train.train_ppo --config configs/train_ppo_v7.yaml
```

Watch the newest run drive, live, while it is still training. Training workers never
render — this is a separate process that hot-swaps each new checkpoint between episodes:

```bash
uv run python -m selfdrive.eval.watch
```

Score a saved policy on held-out seeds:

```bash
uv run python -m selfdrive.eval.evaluate --model runs/<run>/best_model.zip --config configs/env_waypoint_s2_patience_shield.yaml --episodes 100 --seed 3000000
```

Watch one drive the six hand-built showcase arenas instead of random ones:

```bash
uv run python -m selfdrive.eval.evaluate --model runs/<run>/best_model.zip --render human --arena all
```

Run the adversarial scorecard — dead-end corridor, U-trap, tight doorway, box canyon,
moving obstacle, sensor blackout:

```bash
uv run python -m selfdrive.eval.evaluate --model runs/<run>/best_model.zip --scenarios
```

Export for the phone, as one self-contained ONNX file with no external weights:

```bash
uv run python -m selfdrive.export.to_onnx --model runs/<run>/best_model.zip
```

A display is needed only for `--render human`; WSLg works.

---

## How the car sees and acts

**Observation** — every value analytically normalized to `[-1, 1]` inside the environment,
so the exported ONNX is self-contained and there is no running-statistics file for the
Android app to keep in sync.

| Model family | Floats | Blocks |
|---|---|---|
| Phase 1 | 68 | 4 stacked frames × 17 |
| + obstacle memory | 116 | above + 48-sector memory ring |
| + goal block | 119 | above + goal range, sin and cos of bearing |
| + patience clock | 120 | above + seconds since the goal last got closer |

The 17 floats in a frame are 8 depth-arc buckets, 1 depth-confidence value, 4 ultrasonic
ranges (front/left/right/back) and 4 ego-state values (speed, steering angle, last throttle
command, last steering command). Frame-stacking gives the policy approach *rate* without an
LSTM. The exact layout is in [`envs/obs.py`](src/selfdrive/envs/obs.py) and **must be
mirrored by the Android app** — [`docs/goal-block.md`](docs/goal-block.md) and
[`docs/memory-ring.md`](docs/memory-ring.md) document the two extra blocks for that team.

**Action** — `Box(-1, 1, shape=(2,))`: `[steering, throttle]`. Steering positive is left,
throttle spans −1 (full reverse) to +1 (full forward), so reversing out of a trap is
available at every single step. Same signs in the simulator, the protocol and the firmware.

**Dynamics** — kinematic bicycle, because the chassis is Ackermann. There is no tire
friction model on purpose; the realism that decides whether transfer works lives in
[`dynamics/actuators.py`](src/selfdrive/dynamics/actuators.py) — servo slew rate, motor
lag, throttle deadband, steering trim. A servo that cannot teleport and a motor that
cannot stop instantly matter far more than slip angles at 0.8 m/s.

**Domain randomization** — every `reset(seed=)` redraws actuator constants, sensor noise,
dropout, latency and control-period jitter, all from the environment's own RNG so seeded
runs stay byte-reproducible (`tests/test_determinism.py` guards exactly this).

---

## Three hardware facts that shape the whole design

1. **ARCore will not use an ultra-wide camera.** It binds the main lens, which on the
   Galaxy S21 FE is 26 mm equivalent: about **69° horizontally in landscape, ~50° in
   portrait**. The phone's advertised 123° belongs to a lens ARCore ignores. **Mount the
   phone landscape** — 20° of peripheral vision for the cost of rotating a bracket.
2. **The S21 FE has no ToF sensor**, so ARCore depth is motion stereo: it degrades badly
   when the car is stopped or turning in place — exactly when it is trying to unstick
   itself. The simulator models speed-dependent depth dropout so the unstick behaviour is
   learned against that handicap rather than against depth that never fails.
3. **HC-SR04 ultrasonics cross-talk**, so they must fire one at a time. Four sensors
   round-robin at roughly 15–25 Hz total against a 30 Hz policy, so most telemetry frames
   repeat a stale reading. The simulator models the staleness instead of hiding it.

---

## What training actually taught us

Most of the engineering in this project went into reward functions that were technically
satisfied and behaviourally useless. Each failure is now pinned by a test so it cannot
come back. The full log is in [`CLAUDE.md`](CLAUDE.md); the short version:

- **Paying for net displacement** bought wide fast circles. A 1–2 m orbit paid nearly as
  well as a straight line over a 2 s window.
- **Paying for new ground covered** fixed that, and bought *laps* instead: every lap after
  the first is free, and a collision cost about as much as 100 m of new ground, so the car
  happily orbited open floor at full steering lock.
- **The action distribution circled too.** The environment clips Gaussian actions to
  [-1, 1], so extra standard deviation is free — the policy's steering mean ran past |5|
  and sat at full lock. Fixed by bounding actions with tanh (squashed gSDE).
- **Then it saturated by degrees.** Squashed policies held a pre-tanh throttle mean of +2
  to +3.6 with a wall a metre ahead, so *every* noise sample squashed to near-full
  throttle: exploration never braked, PPO never saw an advantage signal for braking, and
  crashes came at full lock and full throttle.
- **Charging for jittery commands** made the car hide from the charge by pinning both
  actuators at a limit, where tanh flattens the noise — and orbiting there instead. One run
  did it in reverse, where the metric that counts full-lock driving could not see it.
- **Straight-line distance to a goal** rewards pressing against the wall between the car
  and the goal. Progress is paid on *path* distance over a 0.075 m raster instead.

The reward tests pin an *ordering*, not a number: forward driving must beat parking, tight
circles, wide fast loops, a slow weave, shuffling in place, and reversing. If a reward
tweak breaks one of those, the tweak is wrong.

---

## For the firmware and app teams

[`docs/protocol.md`](docs/protocol.md) is the contract and it is **frozen**. Both example
frames in it are pinned byte-for-byte in `tests/test_protocol.py` so the document cannot
quietly rot away from the code. If v1 is genuinely wrong, propose a v2 — do not edit v1.

```
Phone → ESP32:  $C,<seq>,<steer_deg>,<throttle>*<crc>
ESP32 → Phone:  $T,<seq>,<v_mps>,<steer_deg>,<us_f>,<us_l>,<us_r>,<us_b>,<flags>*<crc>
```

**The 200 ms failsafe is not optional.** If no valid `$C` arrives within 200 ms the ESP32
must cut motor output. A crashed phone app or an unplugged USB cable must not leave a
LiPo-powered car driving at a wall, or at a person.

| Document | For |
|---|---|
| [`protocol.md`](docs/protocol.md) | the serial contract, both directions, CRC and failsafe |
| [`phone-loop.md`](docs/phone-loop.md) | the Android loop step by step, with a reference implementation |
| [`goal-block.md`](docs/goal-block.md) | computing waypoint range and bearing from ARCore |
| [`memory-ring.md`](docs/memory-ring.md) | the 3 s obstacle memory the app must maintain |
| [`shield.md`](docs/shield.md) | the speed cap and back-out rule that wrap the policy |
| [`sim2real.md`](docs/sim2real.md) | what to measure on the real chassis, and in what order |
| [`architecture.md`](docs/architecture.md) | how the simulator fits together |

`link/sim_link.py` lets a real phone drive the *simulator* over the *real* protocol, so a
hardware-in-the-loop demo — real phone, real ARCore, real policy, real serial frames,
simulated car — works even if the chassis slips.

---

## Layout

```
configs/          30 YAMLs: car, sensors, arenas, domain randomization, PPO hyperparameters
docs/             the frozen protocol and the port guides above
src/selfdrive/
  world/          vectorized ray casting, collision, arena generators, navigation raster,
                  reachability by whether the car body can actually fit and turn
  dynamics/       kinematic bicycle + the actuator model that decides transfer
  sensors/        depth arc, ultrasonics, odometry drift, noise/latency/dropout
  envs/           the Gymnasium env, observation builder, rewards, goals, memory ring,
                  speed shield, domain randomization
  render/         pygame viewport (never imported by a training worker)
  train/          PPO entry point, callbacks, vec env factory, benchmark
  eval/           rollouts, adversarial scenarios, six hand-built showcase arenas
  export/         self-contained ONNX export for the phone
  link/           serial codec, simulated ESP32, and the phone-side reference loop
tests/            400 tests across 22 files
```

---

## This is CPU work

`pyproject.toml` pins the **CPU** build of torch on a machine with an RTX 3080, on purpose.
The policy is a small MLP; throughput is bound by stepping environments across 20 worker
processes, not by the network.

`python -m selfdrive.train.bench` reports ~900 env-steps/s single-process and ~4,800 across
20 workers here. The ~26 % per-worker efficiency is per-step pipe latency in the parent
process under WSL2, not CPU starvation — two workers already sit at 56 %, 28 workers beat
20 by nothing, and thread pinning changes nothing. Even a zero-cost environment step would
cap out near 5,700 steps/s, so optimizing the geometry further buys almost nothing. Don't
chase it.

Trainers running side by side **must** pin torch threads (`torch_threads`, default 4).
Unpinned, three parallel runs fell to 853 steps/s in total; pinned, the same three ran at
3,894.

---

## Not done yet

- **Measure the real chassis.** Every car constant is a placeholder until then, and
  narrowing the randomization ranges around real measurements should beat the wide-band
  policy outright. [`docs/sim2real.md`](docs/sim2real.md) is the checklist.
- **Put it on hardware**, on a tether or with a kill switch, at reduced top speed.
- **Phase 3**: pan-tilt camera servo, which turns this into a genuine POMDP and swaps PPO
  for a recurrent policy. The observation builder is already block-structured for it.

Development notes, the full experiment log and the rules this codebase is maintained under
are in [`CLAUDE.md`](CLAUDE.md).
