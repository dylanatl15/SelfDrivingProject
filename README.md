# Self-Driving Toy Car — Simulator and RL Training Stack

Senior project. This repository is the **simulation and policy training** half: a custom
Gymnasium environment that mimics the real car's sensors and actuators closely enough
that a policy trained here works on hardware, plus the PPO training and evaluation
around it.

The other half — Android app, ESP32 firmware, chassis — is built against
[`docs/protocol.md`](docs/protocol.md), which is frozen.

## Phases

| Phase | Goal | Status |
|---|---|---|
| **1** | Drive forward, never crash, **never stay stuck**. No waypoints. | in progress |
| 2 | Navigate to a dropped GPS pin inside a perimeter. | scaffolded |
| 3 | Pan-tilt camera servo → POMDP → recurrent policy. | scaffolded |

## Quickstart

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

```bash
uv run python -m selfdrive.train.bench --n-envs 20
```

```bash
uv run python -m selfdrive.train.train_ppo --config configs/train_ppo.yaml
```

Watch training live. Workers never render, so this is a separate process that follows the
newest run's checkpoints (every 500k steps) and hot-swaps each newer policy in between
episodes. Before the first checkpoint it drives a random policy. Needs a display; WSLg works.

```bash
uv run python -m selfdrive.eval.watch
```

```bash
uv run tensorboard --logdir runs
```

Watch a specific saved policy drive:

```bash
uv run python -m selfdrive.eval.evaluate --model runs/<run>/best_model.zip --render human
```

Run the adversarial scorecard — this is the Phase 1 deliverable:

```bash
uv run python -m selfdrive.eval.evaluate --model runs/<run>/best_model.zip --scenarios
```

## How it works

**Observation** — 17 floats per frame, stacked 4 deep (68 total), all in `[-1, 1]`:
8 depth-arc buckets, 1 depth-confidence value, 4 ultrasonic ranges, and 4 ego-state
values (speed, steering angle, last throttle command, last steering command). Layout is
documented in [`src/selfdrive/envs/obs.py`](src/selfdrive/envs/obs.py) and **must** be
mirrored exactly by the Android app.

**Action** — `Box(-1, 1, shape=(2,))`: `[steering, throttle]`. Throttle spans −1 (full
reverse) to +1 (full forward), so backing out of a trap is available at every step.

**Dynamics** — kinematic bicycle (the chassis is Ackermann). No tire friction model; the
realism that matters for transfer lives in
[`dynamics/actuators.py`](src/selfdrive/dynamics/actuators.py) — servo slew rate, motor
lag, throttle deadband, steering trim.

**Reward** — progress is scored as *net displacement over a 2 s window*, not instantaneous
speed, so driving in circles earns roughly a third of what real progress does. Reversing
is a small cost, never a bonus. Rationale and the arithmetic are in
[`envs/rewards.py`](src/selfdrive/envs/rewards.py).

## Three hardware facts that shape the whole design

1. **ARCore does not support ultra-wide cameras.** It binds the main lens, which on the
   Galaxy S21 FE is 26 mm equivalent: about **69° horizontally in landscape and only
   ~50° in portrait**. The phone's advertised 123° belongs to a lens ARCore will not use.
   **Mount the phone in landscape** — 20° of peripheral vision for the price of rotating
   a bracket.
2. **The S21 FE has no ToF sensor**, so ARCore depth is motion stereo and degrades badly
   when the car is stopped or turning in place — exactly when it is trying to unstick
   itself. The simulator models speed-dependent depth dropout for this reason.
3. **HC-SR04s cross-talk**, so they must fire one at a time. Four sensors round-robin at
   roughly 15–25 Hz total against a 30 Hz policy, meaning most telemetry frames repeat a
   stale reading. The simulator models that staleness rather than hiding it.

## This is CPU work

`pyproject.toml` pins the **CPU** build of torch on a machine with an RTX 3080, on
purpose. The policy is an MLP over 68 floats; throughput is bound by env stepping across
20 worker processes, not by the network. Instructions for switching to CUDA are in the
comment above the pin.

## Layout

```
configs/     YAML: car, sensors, arena, domain randomization, PPO hyperparameters
docs/        protocol.md (frozen contract), architecture.md, sim2real.md
src/selfdrive/
  world/     vectorized ray casting, collision, arena generators
  dynamics/  kinematic bicycle + the actuator model
  sensors/   depth arc, ultrasonics, noise/latency/dropout
  envs/      the Gymnasium env, observation builder, reward, randomization
  render/    pygame viewport (never imported by a training worker)
  train/     PPO entry point, callbacks, vec env factory, benchmark
  eval/      rollouts and the adversarial scenario suite
  export/    ONNX export for the phone
  link/      serial codec + simulated ESP32
tests/
```
