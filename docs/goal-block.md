# Goal block: range and bearing to a waypoint

This is for the Android app. Waypoint models take three more inputs than Phase 1 models,
and the phone has to compute them. Phase 1 models, with 68 or 116 inputs, are unaffected.

A model has the goal block when it was trained with `obs.goal_block: true`, as
`configs/env_waypoint.yaml` sets it. That config also has the obstacle memory, so its
models take **119 floats**:

| Indices | Block | Rules |
|---|---|---|
| 0-67 | 4 stacked frames of 17 | `envs/obs.py` |
| 68-115 | obstacle memory ring | `docs/memory-ring.md` |
| 116-118 | goal block | this document |

The goal block is always last and is not frame-stacked. Each step carries only the
current value.

Models trained with `obs.goal_patience: true` take a fourth goal float, the patience clock,
so **120 floats**, with the goal block at 116-119. See [The patience clock](#the-patience-clock).

## The three floats

| Index | Value | Formula |
|---|---|---|
| 116 | range | `clip(r / 15.0, 0, 1) * 2 - 1` |
| 117 | bearing, sine | `sin(b)` |
| 118 | bearing, cosine | `cos(b)` |

- `r` is the straight-line distance in metres from the car's centre to the goal.
- `b` is the angle from the car's heading to the goal, in radians. It is **left-positive**
  (counter-clockwise), the same convention as steering.
- A goal dead ahead reads `[.., 0, 1]`. One directly to the left reads `[.., 1, 0]`, and one
  directly behind reads `[.., 0, -1]`.
- `15.0` is `norm_goal_max`. Like every normalization constant, it is a published interface:
  changing it in training invalidates every checkpoint.
- The bearing is split into sine and cosine because a single angle jumps from +pi to -pi
  as the goal passes behind the car. The network would see that jump as a huge change in
  direction.

## Computing it from ARCore

ARCore's world frame is right-handed with +Y up. The simulator's floor plane is `(x, y)`
with angles counter-clockwise when viewed from above. Map ARCore to the simulator as
`x = X`, `y = -Z`. A positive rotation about +Y is then a left turn.

1. **Store the goal.** Store the pin in ARCore world coordinates when it is placed, and
   convert it once to `(gx, gy)`.
2. **Find the camera heading.** Every frame, take the camera pose. The camera looks along
   its local -Z. Rotate `(0, 0, -1)` by the pose to get the forward vector `(fX, fY, fZ)`.
   The car heading is `theta = atan2(-fZ, fX)`.
3. **Find the car centre.** It is not the camera. In the simulator the camera sits
   `mount_forward` = 0.18 m ahead of the centre, so `cx = camX - 0.18 cos(theta)` and
   `cy = -camZ - 0.18 sin(theta)`. Replace 0.18 with the measured mount offset once the
   chassis exists.
4. **Compute the block.** `dx = gx - cx`, `dy = gy - cy`, `r = hypot(dx, dy)`,
   `b = atan2(dy, dx) - theta`. Wrap `b` into (-pi, pi]; sine and cosine do not care, but
   logs will.

The simulator computes these floats from a drifting pose estimate, not the true pose:
about 3 cm of error per square-root metre driven, plus 0.5 s tracking outages. The policy
is therefore trained to tolerate ARCore's normal drift. It is not trained on relocalization
jumps, where the pose snaps by tens of centimetres. Expect a brief swerve after one.

While tracking is lost, keep sending the last block you computed. When no goal is set, the
simulator sends `[1, 0, 1]`, a far goal dead ahead, so the car drives forward. The app
should stop the car instead.

## The patience clock

Only for models trained with `obs.goal_patience: true`.

| Index | Value | Formula |
|---|---|---|
| 119 | seconds waited | `clip(t / 10.0, 0, 1) * 2 - 1` |

`t` is how long the car has gone without getting closer to the goal. It lets the model notice
that it is shuffling in a dead end and try something else. Keep two numbers, `best` and `t`:

1. **When a goal is set**, or the app advances to the next one: `best = r`, `t = 0`.
2. **Every policy step after that**, with `r` computed as above and `dt` the seconds since
   the previous policy step:
   - if `r <= best - 0.25`, then `best = r` and `t = 0`;
   - otherwise `t = t + dt`.

- Going back and forth never restarts the clock. Only a new low range at least 0.25 m below
  the previous one does. Backing away from the goal counts as waiting.
- `10.0` (`norm_patience_max`) and `0.25` (`patience_gain`) are published constants, like
  `15.0`.
- The clock runs while tracking is lost; keep adding `dt` against the last `r` you computed.
- The simulator computes `r` from the same drifting estimate as the rest of the block.


- **Arrival.** Advance to the next waypoint, or stop, when the range you compute is at most
  **0.5 m**. The app can only judge arrival from its own pose. Models trained with
  `goal.arrival_from_odometry: true` were trained on exactly that rule. Earlier models were
  trained to count arrival within 0.5 m of the true goal, which the app cannot know, and on
  a drifted pose they learned to hunt around the spot the app calls arrived.
- **Goal placement.** Training goals are always at least 0.5 m from any obstacle, and
  always somewhere the car body can drive to, forward and reversing, within its turning
  radius. Gaps from 0.3 m count when the car can line up with them. A goal the app places
  behind a narrower gap, or in a closed room, is outside anything the model has seen.
- **Goal distance.** Training goals are 2-12 m away along the drivable path.
- **Longer routes.** Beyond 15 m straight-line, the range input saturates and the model
  sees only "far". For a longer route, feed waypoints no more than about 10 m apart.

## What the model does not know

The model has no map. It knows which way the goal is and how far away it is in a straight
line. It also sees what the depth camera and ultrasonics show now, and what the memory ring
held over the last 3 seconds. Training rewards it for closing distance along the real route
around walls. So it learns to head toward the goal, go around obstacles, and take gaps that
lead the right way.

Without a map, it cannot know that a corridor it has not seen ends in a dead end. For a route
through a building, plan the route on the phone over ARCore's map and hand the model the next
waypoint along it.
