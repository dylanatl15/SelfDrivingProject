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
driving must beat parking, circling, shuffling in place, and reversing. If a reward tweak
breaks one of those, the tweak is wrong, not the test.

## Throughput ceiling - do not chase it

`python -m selfdrive.train.bench` reports ~900 steps/s single-process and ~4,800 across
20 workers on this machine. The ~26% per-worker efficiency is per-vec-step pipe latency
in the parent under WSL2, not CPU: two workers already sit at 56%, 28 workers beat 20 by
nothing, BLAS thread pinning changes nothing, and fork/forkserver/spawn agree within 3%.
Even a zero-cost env step would cap out near 5,700 steps/s. Optimizing geometry further
buys almost nothing; the only real fix would be batching many steps per IPC round trip.

## Known placeholders

`configs/env_phase1.yaml` car geometry is a generic ~1/10 RC platform. Replace with
measured values once the chassis exists — `docs/sim2real.md` lists exactly what to
measure and how to narrow the randomization ranges afterwards.
