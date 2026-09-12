# Sim-to-real calibration

The simulator currently ships **placeholder** chassis numbers for a generic ~1/10 scale
RC platform. Domain randomization is wide enough to cover a lot of wrong guesses, but a
wide band costs sample efficiency and final performance. Measuring the real car and
narrowing the ranges is the single highest-value thing to do once the chassis exists.

Nothing here blocks training. Do it during the November calibration window.

## What to measure

| Quantity | How | Goes to |
|---|---|---|
| Wheelbase | Front axle centre to rear axle centre, with a tape measure | `car.wheelbase` |
| Body length / width | Widest points **including** the phone bracket and any overhang | `car.length`, `car.width` |
| Max steering angle | Full lock, measure the front wheel angle against the chassis centreline | `car.max_steer_deg` |
| Servo slew rate | Command full left → full right, film it at 60 fps, count frames | `car.steer_rate_deg_s` |
| Steering trim | Command zero, drive straight 5 m, measure the sideways drift | `car.steer_trim_deg` |
| Top speed | Full throttle down a measured 5 m, time it | `car.max_speed_fwd` |
| Reverse top speed | Same, backwards | `car.max_speed_rev` |
| Acceleration lag | Full throttle from rest; time to reach ~63% of top speed | `car.accel_tau` |
| Throttle deadband | Ramp throttle up slowly; note the value where it first moves | `car.throttle_deadband` |
| Stopping distance | Full speed to throttle zero; measure the coast | sanity-check `accel_tau` |
| Minimum turning circle | Full lock, drive a full circle, measure the diameter | validates `min_turn_radius` |

## Sensor checks

| Quantity | How | Goes to |
|---|---|---|
| Depth FOV | Point at a wall with markers at known angles; find where depth stops being valid | `depth.fov_deg` |
| Depth max useful range | Walk back from a wall; note where depth gets unusable | `depth.max_range` |
| Depth error vs distance | Compare reported depth to tape measure at 0.5, 1, 2, 3, 4 m | `depth.noise_frac` |
| **Depth dropout while stationary** | Hold the car still, log the fraction of invalid pixels for 10 s | `depth.stationary_dropout` |
| Depth latency | Wave a hand across the view, count frames until depth reacts | `depth.latency_steps` |
| Ultrasonic refresh rate | Log `$T` frames, count how often each reading actually changes | `ultrasonic.update_hz` |
| Ultrasonic noise | 20 readings against a flat wall at 1 m; take the standard deviation | `ultrasonic.noise_m` |
| Ultrasonic dropout | Same against carpet, foam, and a wall at 45° — the surfaces that scatter | `ultrasonic.dropout_prob` |

The stationary-dropout measurement matters more than it looks. It is the number that
decides whether the trained unstick behaviour works on hardware, because the car is
stationary exactly when it needs to decide to reverse.

## Narrowing the ranges

Once measured, set the nominal value in `configs/env_phase1.yaml` and tighten the band in
`DomainRandConfig` to roughly **±15% around the measurement** rather than the current wide
spans. Keep the band — do not collapse it to a point. Batteries sag, servos wear, and
carpet is not tile.

Retrain after narrowing. Expect better final performance than the wide-band policy.

## Order of operations

1. Measure the chassis, update `configs/env_phase1.yaml`, retrain, check the scenario suite.
2. Export to ONNX, run it on the phone against `SimEsp32` over the real protocol
   (`src/selfdrive/link/sim_link.py`) — this catches unit, sign and field-order bugs on a
   desk instead of on a moving car.
3. Only then put it on the real chassis, on a **tether or with a kill switch**, in an open
   space, at reduced `max_speed_fwd`.

## If the real car behaves worse than the sim

In rough order of likelihood:

1. **Sign conventions.** Steering positive is left, throttle positive is forward, in the
   sim, the protocol and the firmware. One flipped sign produces a car that steers
   confidently into obstacles.
2. **Observation normalization drift.** The Android app's constants must match
   `ObsConfig` exactly. Print both and compare digit by digit.
3. **Latency worse than modelled.** Measure the true phone-to-actuation delay. If it
   exceeds `depth.latency_steps` at the top of the randomized range, widen it and retrain.
4. **Ultrasonic cross-talk.** If readings look impossibly close, the ESP32 is firing
   sensors in parallel. They must be round-robin.
5. **Depth FOV narrower than configured.** Portrait mount is ~50°, not 69°.
