# Obstacle memory ring: spec for the Android app

The app needs this **only** to run a policy trained with the memory on. Such a policy
takes 116 inputs instead of 68. The serial protocol does not change.

The reference implementation is [`src/selfdrive/envs/memory.py`](../src/selfdrive/envs/memory.py)
(the ring) plus `CarEnv._remember` in
[`src/selfdrive/envs/car_env.py`](../src/selfdrive/envs/car_env.py) (what gets stored).

## What it adds

The observation is the usual 68 floats (four stacked 17-float frames), followed by 48 more:

| Index | Content |
|---|---|
| 68–91 | distance to the nearest remembered obstacle, sectors 0–23 |
| 92–115 | age of that same obstacle, sectors 0–23 |

The 24 sectors are 15° wide. Sector 0 is centred straight ahead, and indices increase
counter-clockwise, the same sign as steering: 0 is ahead, 6 left, 12 behind and 18 right.

The ring is not stacked. Compute it once per control step and append it after the stack.

## Constants

| `ObsConfig` field | Value |
|---|---|
| `memory_sectors` | 24 |
| `memory_seconds` | 3.0 s |
| `norm_memory_max` | 5.0 m |
| `norm_depth_max` | 5.0 m (already mirrored for the depth buckets) |
| `norm_ultra_max` | 4.0 m (already mirrored for the ultrasonics) |

The mounting offsets used below (camera 0.18 m ahead of the car's reference point,
ultrasonics at ±0.18 m and ±0.10 m) are the simulator's placeholder geometry. When the
chassis is measured, the real values go into `configs/env_phase1_memory.yaml` *and* the app.

## 1. Pose on the floor plane

All of this is 2D, in ARCore's world frame projected onto the floor. ARCore's world is
Y-up, so the floor plane is X and −Z, which also makes counter-clockwise positive.

```java
Pose cam = frame.getCamera().getPose();       // physical camera, not the display pose
double X = cam.tx();
double Y = -cam.tz();
float[] f = cam.rotateVector(new float[] {0f, 0f, -1f});   // lens direction
double theta = Math.atan2(-f[2], f[0]);

// The simulator's pose is the car's reference point, which sits behind the camera.
double x = X - 0.18 * Math.cos(theta);
double y = Y - 0.18 * Math.sin(theta);
```

Only use a pose while `camera.getTrackingState() == TrackingState.TRACKING`. In any other
state, **delete every stored point** and store nothing until tracking returns. Points
stored against the old pose cannot be related to the new one. The ring then reads all
+1, which is correct.

Times are seconds on one monotonic clock, for example `frame.getTimestamp() / 1e9`.

## 2. Store new points, once per control step

Only new measurements are stored. A reading repeated from an earlier moment was measured
from an earlier pose, and storing it again at the car's current pose smears the
obstacle along the car's path.

A point in the car frame `(px, py)` (forward, left) becomes a world point with the pose
`(xc, yc, θc)` named in each rule:

```
wx = xc + cos θc · px − sin θc · py
wy = yc + sin θc · px + cos θc · py
```

### Depth buckets

For each bucket `i` (0–7), with `d` the bucket's distance in metres (the value that goes
into the observation, before normalization):

- **Skip** it unless the value comes from a depth image that is new since the last
  control step *and* had at least one valid pixel in that bucket. Otherwise the value is
  being held from an earlier image.
- **Skip** it if `d ≥ 5.0`: nothing within range is not an obstacle.
- Otherwise:

  ```
  a  = −FOV/2 + (FOV/8)·(i + 0.5)     // radians, left positive: bucket 0 is the RIGHTMOST
  px = 0.18 + d·cos a
  py =        d·sin a
  ```

  Transform with the pose **at the moment that depth image was captured**, and stamp the
  point with that capture time. Depth arrives one to four control steps late, and
  at 1.5 m/s the current pose would misplace the obstacle by up to 20 cm. Keep the last
  ten or so poses with their timestamps, and pick the one matching the depth image's
  timestamp.

### Ultrasonics

For each sensor in the latest `$T` frame, with `d` its value in metres:

- **Skip** it if `d` is `-1.000` (no echo) or `d ≥ 4.0`.
- **Skip** it if `d` is identical to that sensor's value in the previous `$T` frame. A
  sensor waiting for its round-robin turn repeats its last reading. Occasionally a
  genuinely new ping repeats the same millimetre too. Skipping it loses nothing, because
  the earlier point is still stored.
- Otherwise transform with the **current** pose and stamp it `now`:

  | Sensor | Mount (forward, left) | Direction `a` |
  |---|---|---|
  | front | (0.18, 0.00) | 0° |
  | left | (0.00, 0.10) | +90° |
  | right | (0.00, −0.10) | −90° |
  | back | (−0.18, 0.00) | 180° |

  ```
  px = mount_forward + d·cos a
  py = mount_left    + d·sin a
  ```

A 3-sensor build has no front sensor; skip that row.

### Storage

Keep `(wx, wy, stamp)` in any container, and drop points once `now − stamp > 3.0`. The
worst case is 12 points a step for 3 s at 40 Hz, about 1,440, so a fixed-size array is
enough.

## 3. Compute the ring, every control step after storing

```java
float[] ring = new float[48];
Arrays.fill(ring, 1f);
double[] best = new double[24];
Arrays.fill(best, Double.POSITIVE_INFINITY);
double c = Math.cos(theta), s = Math.sin(theta);    // current pose x, y, theta

for (StoredPoint p : points) {
    if (now - p.stamp > 3.0) continue;
    double dx = p.x - x, dy = p.y - y;
    double fwd = c * dx + s * dy;
    double left = c * dy - s * dx;
    double r = Math.hypot(fwd, left);
    if (r >= 5.0) continue;
    int k = Math.floorMod((int) Math.floor(Math.atan2(left, fwd) / (Math.PI / 12) + 0.5), 24);
    if (r < best[k]) {
        best[k] = r;
        ring[k] = (float) (r / 5.0 * 2.0 - 1.0);
        ring[24 + k] = (float) (Math.min((now - p.stamp) / 3.0, 1.0) * 2.0 - 1.0);
    }
}
// observation[68 + i] = ring[i], for i = 0 .. 47
```

If two points in one sector are at exactly the same distance, either may win. With real
sensor noise this does not happen.

## Worked example

Pinned by `test_worked_example_from_the_android_spec` in
[`tests/test_memory.py`](../tests/test_memory.py). An implementation that disagrees with
this table is wrong.

The car is at x = 1.0, y = 2.0, θ = 90° (facing +Y), and now = 10.0 s. Stored points:

| Point | x | y | stamp | (fwd, left) | r | Sector | Result |
|---|---|---|---|---|---|---|---|
| A | 1.0 | 3.5 | 9.5 | (1.5, 0) | 1.5 | 0 | `ring[0] = −0.4`, `ring[24] = −0.667` |
| B | 0.0 | 2.0 | 8.0 | (0, 1.0) | 1.0 | 6 | `ring[6] = −0.6`, `ring[30] = +0.333` |
| C | 1.6 | 2.0 | 7.5 | (0, −0.6) | 0.6 | 18 | `ring[18] = −0.76`, `ring[42] = +0.667` |
| D | 2.5 | 2.0 | 9.0 | (0, −1.5) | 1.5 | 18 | none: C is nearer |
| E | 1.0 | 1.0 | 6.9 | — | — | — | none: 3.1 s old |
| F | 1.0 | −4.0 | 9.0 | (−6.0, 0) | 6.0 | — | none: beyond 5 m |

Every other entry of `ring` is +1.

## What the simulator does that the phone must not

The simulator corrupts the pose on purpose ([`sensors/odometry.py`](../src/selfdrive/sensors/odometry.py)):

- drift of a few centimetres per √metre travelled and about a degree per √radian turned;
- a small scale error;
- occasional tracking loss.

All of these are randomized per episode. On the phone, ARCore's real errors take their
place. They are listed here so that nobody adds them to the app to match the simulator.
