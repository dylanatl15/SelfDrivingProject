# Hardware BOM and sponsorship target list

**Every price here was checked against the vendor's own store on 2026-09-29** and is linked.
Prices move; recheck before quoting one in an email. Assume the team owns nothing, so that
sponsors cover the build rather than the members.

This list is for the **next** hardware generation: a depth camera and a single-board
computer in place of the phone. What that swap costs the software is in
[`architecture.md`](architecture.md#what-the-move-costs). Nothing here is built yet.

> **The 2026 memory shortage dominates this BOM.** LPDDR4/5 is being bid away by AI
> datacentre demand, so single-board computers have roughly doubled or tripled in a year:
> the Raspberry Pi 5 16 GB went from $120 at launch to **$305**. Compute is now within
> striking distance of the depth camera, which is a legitimate thing to say in a
> sponsorship email, because it is why the team cannot simply buy these parts.

## The architecture fork — two options, and both are fine

| | Camera does | Host does | Sensor + compute |
|---|---|---|---|
| **A — recommended** | depth **and runs the policy on-device** | bridge to the ESP32, logging, GPS | **~$454** |
| **B** | depth in its own silicon, over USB | bucket the depth frame, run the policy | **~$499–519** |

The two are close enough on price that whichever vendor answers first wins. A wins slightly
because the host never has to see a depth frame at all.

**Neither option needs an AI accelerator, and we should stop asking for one.** That line was
sized for a host doing monocular depth *estimation*. It is not: every camera below produces
depth in hardware, so the Pi's entire job is to reduce a depth frame to eight column minima
and evaluate a 120 → 256 → 256 → 2 MLP at 30 Hz. That is microseconds of CPU. An AI HAT+
would sit idle, and asking a sponsor for one we do not need is exactly the thing this
document is supposed to prevent.

Both options run the same reward, the same exams and the same shield. Email both.

### Ruled out, with the reason — do not re-add these

Recorded here so that nobody, us included, puts them back on the sponsor page in six months.

- **360° planar laser scanners (Slamtec RPLIDAR and anything like them).** It reads as the
  best sensor on paper — 360°, 12 m, $72 — and it is wrong for *this* car. A scanner has to
  be mounted above the chassis so the body does not sit in its own beam, and from up there
  it sees a single horizontal plane above the car. Most of what a 1/10-scale car has to
  avoid is *shorter* than the car: cones, kerbs, chair feet, the lip of a doorway. A sensor
  that cannot see a cone is not an upgrade over a camera that can, whatever its spec sheet
  says. This is not a randomization range we can widen; it is geometry.
- **Small CSI time-of-flight modules (Arducam ToF and similar), ~$50.** 240×180 at 4 m, and
  narrower than the phone. We already own a phone whose ARCore depth is better in daylight
  and at range. Its one genuine advantage over the phone — depth while stationary — is an
  advantage every camera in the table below also has, at real resolution. Taking one would
  mean a sponsor paid for a downgrade.
- **Stereolabs ZED 2i, $499, and the NVIDIA Jetson Orin Nano Super, $399, that it needs.**
  The ZED does its depth on the *host* GPU, so it drags a Jetson in behind it and nearly
  triples the compute budget. It buys 110° of field of view, which is inside our randomized
  55–125° band and therefore worth roughly nothing that the $264 Orbbec does not already
  give us. The Jetson had no other reason to be on this list and goes with it.

---

## Tier A — the car does not exist without these

### 1. Depth sensor — ask every one of these

| Option | Price | FOV | Note |
|---|---|---|---|
| [**Luxonis OAK-D S2**](https://shop.luxonis.com/collections/oak-cameras-1) | **$329** | ~69° HFOV RGB, ~80° stereo | **best fit.** 69° is exactly what the policy already trained on, and it runs the network on-device (Option A) |
| [Luxonis OAK-D](https://shop.luxonis.com/collections/oak-cameras-1) | $329 | same | the original, same price |
| [Luxonis OAK-D Lite](https://shop.luxonis.com/collections/oak-cameras-1) | $269 | ~69° | cheapest on-device option |
| [Luxonis OAK-D Pro W](https://shop.luxonis.com/collections/oak-cameras-1) | $529 | ~95–127° | active IR, works in the dark; at the wide end of our randomized 55–125° band |
| [Luxonis OAK-D Short Range](https://shop.luxonis.com/collections/oak-cameras-1) | $329 | — | tuned for < 1 m, which is where our crashes happen |
| [**Intel RealSense D435i**](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435i.html) | **$334** | 87° × 58° | **has a built-in IMU**, which plugs the pose hole the phone leaves behind |
| [Intel RealSense D435](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435.html) | $314 | 87° | same without the IMU |
| [Orbbec Gemini 335](https://store.orbbec.com/products/gemini-335) | $264 | ~90° | cheapest name-brand active stereo |
| [Orbbec Gemini 335L](https://store.orbbec.com/products/gemini-335l) | $359 | ~90° | longer range |

Every row above does its depth work in its own silicon, which is why the host stays small.
See *Ruled out* for the three that used to be in this table.

Contacts: Luxonis, RealSense Inc., Orbbec, RobotShop, DFRobot, Seeed, Waveshare.

### 2. Host computer

| Option | Price | Note |
|---|---|---|
| [Raspberry Pi 5 4 GB](https://www.pishop.us/product/raspberry-pi-5-4gb/) | **$110** | genuinely enough: the camera does the depth and the policy is a two-layer MLP |
| [**Raspberry Pi 5 8 GB**](https://www.pishop.us/product/raspberry-pi-5-8gb/) | **$175** | **ask for this one.** The headroom is for the logging, not the network |
| [Raspberry Pi 5 16 GB](https://www.raspberrypi.com/products/raspberry-pi-5/) | **$305** | only if a sponsor offers it outright |
| Active cooler + 27 W USB-C PSU + NVMe HAT + SSD | ~$75 total, not individually verified | the Pi 5 throttles without cooling, and an SD card will not survive the logging |

Contacts: Raspberry Pi, PiShop, SparkFun, Seeed, Waveshare, CanaKit.

### 3. Microcontroller — 2 units, one spare

| Option | Price | Note |
|---|---|---|
| [**Espressif ESP32-S3-DevKitC-1-N8R8**](https://www.mouser.com/en/ProductDetail/Espressif-Systems/ESP32-S3-DevKitC-1-N8R8) | **$15** | **pick this.** Native USB, so it presents as a clean USB-CDC device to the host with no bridge chip |
| [ESP32-S3-DevKitC-1-N32R16V](https://www.mouser.com/ProductDetail/Espressif-Systems/ESP32-S3-DevKitC-1-N32R16V) | $17 | more flash and PSRAM |
| [Adafruit ESP32-S3 DevKitC](https://www.adafruit.com/product/5364) | $19.95 | 32 MB flash, easy to mount |
| Seeed XIAO ESP32-S3 | ~$8, not verified | tiny, native USB |

Contacts: Espressif, Mouser, DigiKey, Adafruit, SparkFun, Seeed, DFRobot.

### 4. Chassis — 1/10 scale Ackermann

| Option | Price | Note |
|---|---|---|
| [**Hiwonder Ackermann metal chassis**](https://www.hiwonder.com/products/ackermann-steering-chassis) | **$49.99** | **best value by a wide margin** — includes **dual encoder motors and the front steering servo**, so it covers items 5, 6 and most of 8 in one line. **Verify its dimensions against wheelbase 0.25 m, length 0.40 m, width 0.20 m before asking**; it may be smaller than 1/10, which would mean re-measuring the sim's car block (that is a config change and a retrain, not a redesign) |
| [Tamiya TT-02R 4×4 chassis kit](https://www.towerhobbies.com/product/110-tt-02r-4x4-chassis-kit/TAM47326.html) | **$157.50** | cheapest true 1/10 Ackermann kit; assembling it is a feature for a senior project |
| [Tamiya TT-02 Type S](https://www.tamiyausa.com/shop/110-4wd-shaft-drive-road-tt/rc-tt-02-type-s-chassis-kit/) | $290 | adjustable suspension |
| [Traxxas Slash 4X4 VXL](https://traxxas.com/slash-4x4-vxl-68386-4) | **$429.95** | the **F1TENTH / MIT RACECAR standard platform** — worth naming in the email for credibility. Note its VXL system is *sensorless* brushless, which handles badly at our 1.5 m/s ceiling; ask for a brushed or sensored variant, or plan to replace the ESC |
| [Traxxas Slash 4X4 Ultimate](https://traxxas.com/68077-4-110-slash-4x4-ultimate-short-course-truck) | $499.95 | — |

Contacts: Hiwonder, Tamiya USA, Tower Hobbies, Horizon Hobby, Traxxas, RobotShop, ThinkRobotics.

### 5. Steering servo

Requirement from the simulator: **≥ 28° each way, ≥ 200 °/s**. Everything below clears the
rate requirement by 3× or more, so torque and price are the only real variables.

| Option | Price | Spec |
|---|---|---|
| [Savox SC-1251MG](https://www.whiterosehobbies.com/products/savox-1251mg-low-profile-servo-09-125-at-6-0v) | **$69–81** | 125 oz-in (~9 kg·cm), 0.09 s/60° ≈ **667 °/s** |
| Power HD 1501MG | ~$20, not verified | ~15 kg·cm, 0.14 s/60° ≈ 430 °/s — budget, still plenty |
| Included with the Hiwonder chassis | $0 | if item 4 lands, skip this line |

Contacts: Savox, Hitec, Power HD, ServoCity, AMain Hobbies.

### 6. Drive motor and ESC — **the one non-obvious spec on this page**

The policy tops out at **1.5 m/s forward, 0.6 m/s reverse**, which is a crawl by RC
standards. **Sensorless brushless cogs and stutters down there.** Ask for **brushed, or
*sensored* brushless.** Sensored also gives hall-effect speed, which is a free wheel
encoder for the safety shield — and the shield reads `v_mps` on every step.

| Option | Price | Note |
|---|---|---|
| [**Hobbywing QuicRun 1060 brushed ESC**](https://www.hobbywingdirect.com/products/quicrun-10-brushed-esc) | **$23.99** | smooth at low speed, drag brake, built-in BEC. Pair with any 540/550 27T brushed motor (~$20, not verified) |
| [Hobbywing QuicRun 10BL120 G2 sensored combo + 3650 SL G2 motor](https://www.amainhobbies.com/hobbywing-quicrun-10bl120-g2-sensored-brushless-esc-combo-w-3650-sl-g2-motor-hwa38020381/p1564886) | **$99.99** | sensored, so the encoder comes free |
| [QuicRun 3650 sensored motor alone](https://www.hobbywingdirect.com/products/quicrun-3650-sensored-2-pole-brushless-motor) | $49.99–54.99 | pick a high turn count (17.5T or 21.5T) for low-speed control |
| Included with the Hiwonder chassis | $0 | encoder motors already |

Contacts: Hobbywing, Castle Creations, Traxxas, AMain Hobbies, Tower Hobbies.

### 7. Rangefinders — 4 units, and keep the count at 4

Four is not arbitrary: `$T` in the **frozen** protocol v1 has exactly four ultrasonic
fields. Keep four rangefinders of *any* technology and the firmware teammate's codec work
survives untouched.

| Option | Price each | Note |
|---|---|---|
| [**Adafruit VL53L1X ToF breakout**](https://www.adafruit.com/product/3967) | **$14.95** | **upgrade over HC-SR04.** Laser ToF, 4 m, ~50 Hz, **no cross-talk at all**. Strictly better than what the simulator models, so it can only help |
| [Pololu VL53L1X carrier](https://www.pololu.com/product/3415) | $22.95 | onboard regulator, more rugged |
| [SparkFun VL53L1X (Qwiic)](https://www.sparkfun.com/sparkfun-distance-sensor-breakout-4-meter-vl53l1x-qwiic.html) | ~$25 | daisy-chainable I²C |
| HC-SR04 | ~$2 | what the simulator models today; entirely adequate |

Contacts: ST Microelectronics, Adafruit, Pololu, SparkFun.

### 8. Pose sensors — the piece the phone used to provide for free

Dropping the phone drops ARCore's pose, and both the obstacle-memory ring and the goal
block are built on a pose estimate. This is the **one genuine hole** in the swap.

| Option | Price | Note |
|---|---|---|
| [**Adafruit BNO085 9-DOF IMU**](https://www.adafruit.com/product/4754) | **$29.50** | fuses on-chip and outputs orientation directly, so the host does no filtering |
| [SparkFun u-blox NEO-M9N GPS (U.FL)](https://www.sparkfun.com/sparkfun-gps-breakout-neo-m9n-u-fl-qwiic.html) | **$70.95** | Phase 2 waypoints |
| [SparkFun ZED-F9P RTK](https://www.sparkfun.com/sparkfun-gps-rtk-sma-breakout-zed-f9p-qwiic.html) | $259.95 | centimetre waypoints. A stretch ask, but a real one |
| Wheel encoder | included with a sensored motor or the Hiwonder chassis | otherwise an AS5600 magnetic encoder, ~$10, not verified |
| A RealSense D435i (item 1) | — | its IMU covers this line by itself |

Contacts: Adafruit, SparkFun, u-blox, ArduSimple, CEVA.

### 9. Power

| Item | Option | Price |
|---|---|---|
| Drive packs ×3 | [Gens Ace 2S 5000 mAh hardcase](https://www.buddyrc.com/collections/gens-ace-batteries) | **$40–53 each** |
| Charger | [SkyRC B6Neo, 1S–6S balance](https://www.amazon.com/clp/B0CCT2WGNS) | **$35** |
| **5 V buck, 5 A** | [Pololu D24V50F5](https://www.pololu.com/product/2851) | **$32.95** |
| LiPo safe bag | any | ~$15, not verified |

A Pi 5 can pull 25 W on its own, plus 2–5 W for the camera and servo peaks on top. **Do not
run the host off the ESC's BEC** — it will brown out under a steering peak and reboot the
computer mid-drive. Size the buck for 5 A and keep the rails separate.

Contacts: Gens Ace, Ovonic, Pololu, SkyRC, ISDT, BuddyRC.

### 10. Safety — include this in the email, it reads as maturity

| Item | Option | Price |
|---|---|---|
| Manual override and kill switch | [FlySky FS-i6X + iA6B receiver](https://www.hobbytown.com/flysky-i6x-6channel-afhds-2a-air-radio-system-w-ia6b-receiver-fsy-i6xia6b/p1674181) | **$53–70** |
| Hard cut-off | Pololu big MOSFET slide switch | ~$8, not verified |

[`sim2real.md`](sim2real.md) already requires a tether or a kill switch before the first
real drive. Saying so in the email is worth more than the part costs.

---

## Tier B — Phase 3, ask as a stretch

Phase 3 adds a pan-tilt camera head, and the camera's yaw becomes part of the observation.
That means the servo must **report** its angle, which a hobby servo cannot do.

| Option | Price | Note |
|---|---|---|
| [**ROBOTIS Dynamixel XL330-M288-T**](https://www.robotis.us/dynamixel-xl330-m288-t/) | **$23.90–27.49** | position, velocity, current and temperature feedback over a serial bus. Exactly what the Phase 3 observation needs |
| [Feetech STS3215 serial bus servo](https://www.amazon.com/RCmall-Continuous-Programmable-SO-ARM100-Controller/dp/B0F87XY6F2) | **$29–39** | magnetic encoder, same idea, 19–30 kg·cm |
| Hobby micro servo + pan-tilt bracket | ~$20, not verified | open-loop fallback |
| Wiring, XT60/JST, servo leads, standoffs | ~$50, not verified | ask a distributor to bundle it |

Contacts: ROBOTIS, Feetech, Waveshare, ServoCity, Pololu, Adafruit.

---

## Ballpark totals, verified lines only

| Build | Camera | Host | Chassis path | Total |
|---|---|---|---|---|
| **A — cheapest** | OAK-D S2 $329 | Pi 5 4 GB $110 + ESP32 $15 | Hiwonder $50 | **~$504** |
| **B — Orbbec** | Gemini 335 $264 | Pi 5 8 GB $175 + ESP32 $15 | Hiwonder $50 | **~$504** |
| **B — RealSense** | D435i $334 | Pi 5 8 GB $175 + ESP32 $15 | Hiwonder $50 | **~$574** |
| **A — kit chassis** | OAK-D S2 $329 | Pi 5 8 GB $175 + ESP32 $15 | TT-02R $158 | **~$677** |

Add roughly **$300** to any row for the common lines: rangefinders $60, IMU $30, GPS $71,
batteries and charger $155, buck $33, radio kill switch $60, cooling and storage $75,
wiring $50. A fully sponsored build lands between **$800 and $980** depending on which
camera and which chassis say yes first. The IMU line disappears if the camera that
arrives is the RealSense, which has one built in.

---

## Do not ask for

- **3D printing or filament.** A team member has a printer; this is covered in-house.
- **A phone.** Either the current S21 FE stays as the pose and GPS source, or items 8 and 1
  replace it. Neither case needs a donated phone.
- **A 360° laser scanner, a small CSI ToF module, or a ZED.** See *Ruled out* above for the
  reason in each case. Turning down a donated part is worse for everyone than never asking.

## Notes for whoever writes the emails

- **Lead with the repository and the video.** `github.com/dylanatl15/SelfDrivingProject` is
  public, has 414 tests and a demo reel at `media/demo.mp4`. A working artifact opens doors
  that a description of intent does not.
- **One item per email**, with the option list attached as "any of these would work." It
  costs the vendor nothing to pick the cheapest thing on their shelf, and it raises the
  yield.
- **Name the F1TENTH / MIT RACECAR lineage** when writing to Traxxas.
- **Mention the memory shortage.** "A Raspberry Pi 5 16 GB has gone from $120 to $305 in
  eighteen months" explains the ask better than any appeal to student budgets.
- **Do not claim the car never crashes.** It crashes on 17 % of 100 held-out seeds and is
  improving. Understating that would be the one thing that damages the team's credibility
  if a sponsor reads the repository, which is exactly what we are inviting them to do.
- **No vendor here has a verified student-donation program.** These are cold asks. Do not
  write "we saw you sponsor student teams" unless someone has actually confirmed it.
