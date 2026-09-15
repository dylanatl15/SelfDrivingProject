# The phone's control loop

This is for the Android app. The other phone documents each cover one piece:
[`protocol.md`](protocol.md) the serial frames, [`memory-ring.md`](memory-ring.md) the
obstacle memory, [`goal-block.md`](goal-block.md) the goal and the patience clock, and
[`shield.md`](shield.md) the throttle shield. This one puts them in order, fills in the inputs
no other document spells out, and reports what running that loop against the simulator
found.

The reference implementation is [`src/selfdrive/link/phone.py`](../src/selfdrive/link/phone.py).
It reads only what a phone has: `$T` frames, depth images, ARCore poses and the pin. It never
reads the true pose, the true speed or the randomized car. Copy it.

## Each control step, in order

1. **Read** the newest `$T` frame, the newest depth image (its buckets, which buckets are new,
   the field of view, the capture time and the camera pose at capture), ARCore's current
   camera pose and tracking state, and the current pin.
2. **Place the car** on the floor from the camera pose (`goal-block.md`, steps 2 and 3).
3. **Build this step's 17 floats**, in this order:

   | Index | Input | Value | Normalized |
   |---|---|---|---|
   | 0-7 | depth buckets | metres, bucket 0 the rightmost | `clip(d / 5.0, 0, 1) * 2 - 1` |
   | 8 | depth confidence | `clip(abs(v) / 0.30, 0, 1)`, with `v` the speed below | `c * 2 - 1` |
   | 9-12 | ultrasonics F, L, R, B | `$T` `us_*` in metres; `-1.000` reads as `4.0` | `clip(u / 4.0, 0, 1) * 2 - 1` |
   | 13 | speed | `$T` `v_mps`; with `NO_ENCODER` set, from the pose (below) | `clip(v / 1.5, -1, 1)` |
   | 14 | steering angle | `$T` `steer_deg`, in radians | `clip(rad / 0.4887, -1, 1)` |
   | 15 | last throttle | the throttle the previous `$C` sent, after the shield | as sent, in [-1, 1] |
   | 16 | last steering | the previous `$C` angle over `MAX_STEER_DEG`, after the shield | as sent, in [-1, 1] |

   Depth confidence is not a sensor reading. The simulator models ARCore's depth from motion
   failing as the car stops, and gives the policy that speed ratio. `0.30` is `depth.speed_ref`.
   When driving starts, the last commands are 0.
4. **Stack** the 17 floats: indices 0-67 hold the last four steps, oldest first, so the newest
   step is 51-67. When driving starts, fill all four with the first step.
5. **Memory ring**, indices 68-115: store this step's new points, then compute the ring
   (`memory-ring.md`).
6. **Goal block and patience clock**, indices 116-119 (`goal-block.md`).
7. **Run the model** on the 120 floats. It returns `[steer, throttle]`, each in [-1, 1].
8. **Shield** the command with the readings of step 3 (`shield.md`), backing out if
   `unstick_after` is set.
9. **Send `$C`** with `steer_deg = steer * MAX_STEER_DEG` and the shielded throttle.
10. **Keep** the steer fraction and throttle you sent; they are next step's indices 15 and 16.

Every constant comes from the model's environment config, not from the car, because the
policy was trained against those fixed values. The one number the team has to supply is
`MAX_STEER_DEG`.

### Speed without an encoder

The model reads speed (index 13), the depth confidence depends on it (index 8), and the
shield brakes on it. If the ESP32 has no encoder, it sends `v_mps` `0.0` with `NO_ENCODER`,
and the phone has to estimate speed from ARCore:

```
keep the car poses (t, x, y) of the last 0.2 s, while tracking
v = ((x - x0) cos(theta) + (y - y0) sin(theta)) / (t - t0)   // oldest kept pose; signed
while tracking is lost: keep the last v and forget the poses
```

### The steering limit

The model's steering output is a fraction of full lock, and `$C` carries degrees. The phone
multiplies by `MAX_STEER_DEG`; the ESP32 clamps to it. Both must use the same number: the
measured full-lock angle of the real car (`sim2real.md`). If they disagree, the last-steering
input no longer matches the angle the car actually got.

## What the check found

[`src/selfdrive/link/phone_check.py`](../src/selfdrive/link/phone_check.py) runs `phone.py`
against the simulator, through the real `$C`/`$T` codec and `SimEsp32`. The phone's depth
image is the simulator's depth buckets, captured the randomized number of steps late with the
pose of that moment, and its ARCore pose is the simulator's drifting odometry.

It ran the demo setup: `waypoint_s2_patience_seed1` at 9.5M steps, with the shield and
backing out, on the 100 held-out maps from 4,000,000 that the exam used. It has two modes.

- **drive**: the phone drives the car. That scores the whole loop.
- **shadow**: the simulator drives with its own inputs, and the phone builds its inputs
  beside it. Every difference is then a rule, not a drift that grew.

### Driving

| Who builds the inputs | Clean goals | Goals | Crashes | Stuck |
|---|---|---|---|---|
| the simulator (the exam) | 2.70 | 2.88 | 15 % | 4 % |
| the phone | 2.61 | 2.78 | 16 % | 1 % |
| the phone, `MAX_STEER_DEG` 28 on cars of 22-32 | 2.70 | 2.84 | 18 % | 4 % |
| the phone, no encoder | 2.89 | 3.01 | 8 % | 2 % |
| the phone, no encoder, `MAX_STEER_DEG` 28 | 2.82 | 2.95 | 11 % | 0 % |

Against the exam on the same maps, every clean-goals difference is within about one standard
error (0.16-0.20). The phone's loop drives as well as the simulator's.

Only 34-38 of the 100 maps end the same way, though, because a small difference in an input
sends the car down a different path. Averages agree; single runs do not. Even the exported
ONNX model on the simulator's own inputs matched the exam on only 55 maps.

### Inputs

In shadow mode, over 137,138 steps, with an encoder:

| Input | Largest difference (normalized) | Steps over 0.01 | Why |
|---|---|---|---|
| depth | 0 | 0 % | |
| depth confidence | 0.0033 | 0 % | `$T` rounds speed to 1 mm/s |
| ultrasonics | 0.052 | 19 % | no echo, below |
| speed, steering angle | 0.0003 | 0 % | `$T` rounding |
| memory ring | 2.0 | 3.2 % | repeated millimetres and sector edges, below |
| goal range, bearing | 0.10, 0.89 | 0.7 %, 1.0 % | tracking loss, below |
| patience clock | 2.0 | 4.1 % | tracking loss, below |

The phone's 0.5 m arrival rule agreed with the simulator on 292 of 298 goals, and never judged
a goal reached that the simulator had not. The shield decided differently on 0.07 % of steps.

Which differences change the model's answer? On 8 maps the answer moved by more than 0.05 on
1.8 % of steps. Giving the phone the simulator's value restored it for the last commands on
51 % of those, the memory ring on 28 %, the goal block on 14 %, and the ultrasonics on under
1 %. The last commands differ only because an earlier answer did, so the memory ring and the
goal block are the real sources.

Without an encoder, the pose-derived speed was more than 0.015 m/s off on 84 % of steps. In
the worst 5 % of maps it was at some moment 0.95 m/s off, and the shield decided differently
on 3.5 % of steps. The car still drove no worse. Measure the estimate against a stopwatch on
the real car before trusting it; the simulator's pose noise is not ARCore's.

## Where the phone and the simulator differ

These are known, measured above, and left as they are.

- **A new ping that repeats the last millimetre.** `$T` has no field saying a reading is new,
  so the phone skips any value equal to the previous frame's (`memory-ring.md`). The simulator
  knows and stores it. A per-sensor "new reading" flag would close this in a protocol v2.
- **Millimetre rounding** now and then moves a remembered point across a sector edge.
- **No echo.** The phone reads `-1.000` as 4.0 m. The simulator keeps the noisy value that
  ping was clipped to, a few centimetres under 4.0.
- **Tracking loss.** The phone holds the goal block while tracking is lost (`goal-block.md`).
  The simulator keeps updating it from dead reckoning.
- **3-sensor builds.** `CarEnv` puts a 3-sensor build's left, right and back readings in the
  first three ultrasonic slots and 4.0 in the fourth, although `envs/obs.py` says the front
  slot is the one dropped. `phone.py` does what `CarEnv` does, since that is what a model
  trains on. No model has been trained with 3 sensors; settle the order before one is.

## Running it

```
PYTHONPATH=src python -m selfdrive.link.phone_check \
    --onnx runs/exports/waypoint_s2_patience_seed1_9500000.onnx \
    --env <the model's env config, shield section included> \
    --mode drive --seed-start 4000000 --n 25 --out rows.jsonl
```

`--no-encoder` takes speed from the pose, and `--nominal-steer` sends degrees against the
config's 28 degrees instead of each simulated car's own limit. Each row holds the episode's
driving scores, and per input the largest difference and the steps over 0.01.
`tests/test_phone.py` holds the rules to rounding on short episodes.
