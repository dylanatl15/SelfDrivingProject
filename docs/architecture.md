# Architecture

## Hardware data flow

```
  Galaxy S21 FE                     ESP32                      chassis
 ┌────────────────┐           ┌────────────────┐         ┌────────────────┐
 │ camera         │           │ parse $C       │         │ steering servo │
 │   ↓ ARCore     │           │   ↓            │         │ drive motor    │
 │ depth map      │  USB-OTG  │ PID / limits   │   PWM   │ 4x HC-SR04     │
 │   ↓ compress   │ ────────► │   ↓            │ ──────► │                │
 │ 8-bucket arc   │  $C 30Hz  │ PWM out        │         │ 2S LiPo → 5V   │
 │   ↓            │           │   ↑            │         │   buck         │
 │ policy (ONNX)  │ ◄──────── │ read sensors   │ ◄────── │                │
 │   ↓            │  $T 30Hz  │ build $T       │         │                │
 │ [steer, throt] │           │ 200ms failsafe │         │                │
 └────────────────┘           └────────────────┘         └────────────────┘
```

The split is deliberate: the phone emits **geometric intent** (an angle and a throttle
fraction), and the ESP32 owns everything electrical. Retuning the chassis — a different
servo, a different gear ratio — does not require retraining the policy.

## Why the phone is the compute node

It already has the camera, an accelerator, ARCore's depth-from-motion, GPS for Phase 2,
and a battery. Adding a Jetson or a Pi plus a camera plus a depth pipeline would cost
more, weigh more, and do less.

## Software flow, one control step

```
world geometry ──► sensors ──► noise/dropout/latency ──► normalize ──► frame stack ──► MLP
      │                                                                                 │
      │                                                                                 ▼
      │                                                                       [steer, throttle]
      │                                                                                 │
      ▼                                                                                 ▼
  clearance, collision (GROUND TRUTH) ──────────────────► reward          actuator model ──► bicycle
```

Observations go through the corruption pipeline; rewards do not. Rewards are computed
from exact geometry because a reward derived from noisy sensors teaches the policy to
optimize sensor artifacts. Privileged information at training time costs nothing — it is
never needed at run time.

## Why a custom simulator

F1TENTH's gym is the closest existing option, but it models a racing car with a 1080-beam
lidar on a race track. This project needs a ~69° camera arc fused with four slow,
cross-talking ultrasonics, speed-dependent depth dropout, and traps designed to test an
unstick policy. That is most of a rewrite, and the environment is small — the kinematic
model is roughly forty lines.

## Observation design

A single depth frame is not Markov: it carries no velocity, no steering state, and no
approach rate. Two cheap additions fix that without a recurrent network:

- **Ego state** in every frame — speed, steering angle, last action.
- **Frame stacking**, 4 deep — supplies approach rate by difference.

Stacking supplies approach rate, not space. The camera sees 55–125° and four frames span
about 0.13 s, so a wall that leaves the field of view is gone. `phase1_v1` responded by
sweeping its nose side to side to keep things in view. The optional **obstacle memory**
(`obs.memory_sectors`, [`envs/memory.py`](../src/selfdrive/envs/memory.py)) addresses that
without a recurrent network. Every new depth or ultrasonic hit is stored as a world-frame
point for 3 s. Each step, the stored points are re-projected into 24 sectors around the
car, giving the nearest distance per sector and that point's age.

A 90-frame stack would cover the same 3 s with 1,530 inputs, each reading taken from a
different pose, and the policy would have to learn the geometry that relates them. The
ring is 48 inputs with that geometry already applied. Points are placed using a drifting
odometry estimate ([`sensors/odometry.py`](../src/selfdrive/sensors/odometry.py)), because
ARCore's pose is all the phone will have.

Phase 3 (a pan-tilt camera servo) genuinely does break the Markov property: looking left
blinds the car to the right, and no amount of stacking recovers what was never observed.
That is when `RecurrentPPO` with `MlpLstmPolicy` earns its cost. Not before.

## Phase 2 and 3 hooks

- **Phase 2 (GPS):** append a two-float `[range, bearing]` goal block to the per-frame
  observation in `envs/obs.py`, and swap the explore term in `envs/rewards.py` for
  reduction in distance-to-goal. Nothing else changes.
- **Phase 3 (pan-tilt):** add a camera-yaw action, swap `PPO` for `RecurrentPPO`. The
  environment does not change at all — `frame_stack` becomes the LSTM's job.
