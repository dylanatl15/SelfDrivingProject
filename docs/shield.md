# Speed shield: capping the throttle near obstacles

This is for the Android app. A model trained with `shield.enabled: true` in its environment
config learned to drive with this shield between it and the car, so the phone has to run it
exactly as described here. It changes only the throttle in `$C`. The model's inputs and the
serial protocol are unchanged.

The rule lives in `src/selfdrive/envs/shield.py`, and `tests/test_shield.py` pins it.

## Why

The crash probe found that Stage 2 models crashed through bad decisions, not blindness.
They drove into obstacles their sensors had already reported: forward at about 1 m/s at
full lock, or backing at about 0.5 m/s, and almost never braking. The shield keeps the car
slow enough to stop short of the nearest obstacle in the direction it is driving.

## Each control step

1. Take the latest `$T` frame (`v_mps`, `us_f`, `us_b`) and the latest depth buckets, in
   metres, before normalization.
2. Build the observation from those readings, as before.
3. Run the policy. It returns `steer` and `throttle`.
4. **Shield the throttle** with the rule below, using the same readings as step 1. Never
   touch the steering.
5. Send `$C` with the shielded throttle.
6. Use the shielded throttle as the "last throttle" input of the next observation. That is
   what the model saw in training: the command the car actually received.

## The rule

```
forward = min(smallest depth bucket, us_f)   // no front ultrasonic: depth buckets alone
back    = us_b                               // no back ultrasonic: reversing is never capped

allowed(d) = 0                                 if d <= margin
           = max(floor, (d - margin) / horizon) otherwise

if throttle > 0:
    a = allowed(forward)
    if v_mps > a + slack:   throttle = -brake
    else:                   throttle = min(throttle, a / MAX_SPEED_FWD)
else if throttle < 0:
    a = allowed(back)
    if -v_mps > a + slack:  throttle = +brake
    else:                   throttle = max(throttle, -a / MAX_SPEED_REV)
// throttle == 0 passes through unchanged
```

The side ultrasonics are not read.

## Backing out (only when `unstick_after` > 0)

A model trained without the shield never learned that pushing into its cap gets nowhere. With
the shield bolted on, such a model can sit nose-in against an obstacle at full lock, asking for
forward throttle that the shield cuts to a crawl, with clear floor behind it. This optional
rule backs it out:

```
// kept across control steps, all zero when driving starts
held = 0; left = 0; backing_steer = 0

each step, after the policy answers:
if left > 0:
    left = left - 1
    throttle = shielded(-unstick_throttle)       // the rule above, reverse branch
    if throttle == 0: left = 0                   // nothing more behind
    steer = backing_steer
else:
    out = shielded(throttle)
    if unstick_after > 0:
        if throttle > 0 and the shield capped it and |v_mps| < unstick_speed: held = held + 1
        else: held = 0
        if held >= round(unstick_after / period) and allowed(back) > 0:
            held = 0; left = round(unstick_for / period); backing_steer = -steer
    throttle = out                               // steer stays the policy's
```

`period` is the control period, about 1/30 s. Reversing on the opposite lock turns the heading
the way the policy was steering, so the nose swings off the obstacle, as in a three-point turn.
While backing out the shield sets the steering as well as the throttle, and the next
observation's last-steer and last-throttle inputs are the values sent, as always.

## Constants

Read these from the `shield:` section of the model's environment config. The defaults are:

| Name | Default | Units | Meaning |
|---|---|---|---|
| `margin` | 0.15 | m | inside this reported distance, the allowed speed is 0 |
| `horizon` | 0.5 | s | beyond the margin, the allowed speed closes the gap in this long |
| `floor` | 0.0 | m/s | least allowed speed while the distance exceeds the margin |
| `brake` | 0.3 | throttle | command against the motion when the car is too fast |
| `slack` | 0.05 | m/s | how far over the allowed speed the car may be before braking |
| `unstick_after` | 0 | s | held this long while pushing forward, back out; 0 never does |
| `unstick_for` | 1.0 | s | how long to back out, cut short at the back margin |
| `unstick_throttle` | 0.5 | throttle | reverse command while backing out, capped by the rule |
| `unstick_speed` | 0.1 | m/s | slower than this while capped counts as held |
| `MAX_SPEED_FWD` | 1.5 | m/s | `car.max_speed_fwd`: the speed at throttle +1 |
| `MAX_SPEED_REV` | 0.6 | m/s | `car.max_speed_rev`: the speed at throttle −1 |

The top speeds are the config's nominal values, not measurements. Training randomizes the
car's real top speed around them, and the phone never knows the true value either.

## Details that matter

- **Same readings as the observation.** The shield must see the depth frame and ultrasonic
  values the observation was built from this step. A newer depth frame that arrives while
  the policy runs waits for the next step.
- **`-1.000` (no echo).** Use the same value the observation's ultrasonic input uses for
  that sensor. In the simulator, a sensor with nothing in range reads 4.000, and a lost echo
  repeats its previous value. Whichever mapping the app picks for the observation, the shield
  uses it too.
- **`NO_ENCODER`.** If `v_mps` is an estimate, the shield uses it as-is. At a flat `0.0` the
  shield can still cap the throttle, but it can never brake.
- **Real top speed.** The cap assumes throttle scales linearly to 1.5 m/s. If the chassis is
  faster or slower, the cap is off by that ratio. Braking on measured speed still catches a
  car that is too fast. Update both top speeds in the config once the car is measured
  (`docs/sim2real.md`).
