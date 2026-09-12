"""Pygame viewport.

Only ever constructed from `CarEnv.render()`, which is only reached when `render_mode`
is set. Training runs leave it at `None`, so the 20 worker processes never import pygame
at all - it would cost a core and throttle itself to the display refresh rate.

The view deliberately draws the true sensor sweep *and* the corrupted readings the policy
actually receives. Watching the two diverge is the fastest way to confirm that dropout,
latency and staleness are behaving, and those are the parts of the model most likely to
be silently wrong.

Under everything sits the ground the explore reward has already paid for, bright where the
car has just been and dimmer with age. A car that orbits shows up at once as a ring that
stops growing. Press C to toggle it.
"""

from __future__ import annotations

import math

import numpy as np

from ..world.geometry import obb_corners

BG = (18, 20, 24)
WALL = (208, 214, 224)
CONE = (232, 160, 80)
CAR = (90, 190, 255)
CAR_EDGE = (220, 240, 255)
RAY_TRUE = (60, 82, 104)
RAY_SEEN = (120, 220, 150)
ULTRA = (235, 110, 130)
TEXT = (210, 216, 226)
WARN = (250, 200, 90)
PAINT_NEW = (64, 168, 128)
PAINT_OLD = (36, 66, 60)
PAINT_FADE_S = 8.0  # seconds for covered ground to fade from PAINT_NEW to PAINT_OLD


class PygameView:
    def __init__(self, mode: str, fps: int, size: tuple[int, int] = (960, 720)):
        import pygame  # noqa: PLC0415 - deliberately lazy, see module docstring

        self.pygame = pygame
        self.mode = mode
        self.fps = fps
        self.size = size

        pygame.init()
        pygame.font.init()
        if mode == "human":
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("selfdrive - phase 1")
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface(size)
            self.clock = None
        self.font = pygame.font.SysFont("monospace", 14)
        self.show_coverage = True

    # --- world <-> screen ----------------------------------------------------

    def _fit(self, bounds) -> tuple[float, float, float]:
        x0, y0, x1, y1 = bounds
        margin = 40
        sx = (self.size[0] - 2 * margin) / max(x1 - x0, 1e-6)
        sy = (self.size[1] - 2 * margin) / max(y1 - y0, 1e-6)
        scale = min(sx, sy)
        ox = self.size[0] / 2.0 - scale * (x0 + x1) / 2.0
        oy = self.size[1] / 2.0 + scale * (y0 + y1) / 2.0
        return scale, ox, oy

    def _to_screen(self, pts: np.ndarray, t) -> list[tuple[int, int]]:
        scale, ox, oy = t
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        xs = pts[:, 0] * scale + ox
        ys = -pts[:, 1] * scale + oy  # screen y grows downward
        return [(int(round(a)), int(round(b))) for a, b in zip(xs, ys, strict=True)]

    # --- drawing -------------------------------------------------------------

    def draw(self, env):
        pg = self.pygame
        if self.mode == "human":
            # Draining the queue keeps the window responsive. Closing the window ends the
            # program rather than leaving a frozen one behind.
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    raise SystemExit(0)
                if event.type == pg.KEYDOWN and event.key == pg.K_c:
                    self.show_coverage = not self.show_coverage

        world, car = env.world, env.car
        s = car.state
        t = self._fit(world.bounds)
        self.screen.fill(BG)
        if self.show_coverage:
            self._draw_coverage(env, t)

        for seg in world.segments:
            a, b = self._to_screen(np.array([[seg[0], seg[1]], [seg[2], seg[3]]]), t)
            pg.draw.line(self.screen, WALL, a, b, 3)

        scale = t[0]
        for cx, cy, r in world.circles:
            centre = self._to_screen(np.array([[cx, cy]]), t)[0]
            pg.draw.circle(self.screen, CONE, centre, max(int(r * scale), 2))

        self._draw_depth(env, t)
        self._draw_ultrasonic(env, t)

        corners = obb_corners(s.x, s.y, s.theta, car.p.length, car.p.width)
        poly = self._to_screen(corners, t)
        pg.draw.polygon(self.screen, CAR, poly)
        pg.draw.polygon(self.screen, CAR_EDGE, poly, 2)
        # A nose marker, so heading is readable when the car is reversing.
        nose = self._to_screen(
            np.array([[s.x + math.cos(s.theta) * car.p.length * 0.5,
                       s.y + math.sin(s.theta) * car.p.length * 0.5]]), t
        )[0]
        pg.draw.circle(self.screen, CAR_EDGE, nose, 4)

        self._draw_hud(env)

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(pg.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def _draw_coverage(self, env, t):
        raster = getattr(env.reward_fn, "coverage_raster", None)
        if raster is None:
            # A training process that loaded an older reward module can still import this
            # view for its eval videos. Draw without the layer rather than crash the run.
            return
        age, (ox, oy), res = raster()
        covered = age < env.reward_fn.c.revisit_s
        cols = np.flatnonzero(covered.any(axis=1))
        rows = np.flatnonzero(covered.any(axis=0))
        if cols.size == 0:
            return

        # Only the bounding box of covered ground, so the cost tracks what has been driven.
        x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        age, covered = age[x0:x1, y0:y1], covered[x0:x1, y0:y1]
        fade = np.clip(age / PAINT_FADE_S, 0.0, 1.0)[..., None]
        new, old = np.array(PAINT_NEW, np.float32), np.array(PAINT_OLD, np.float32)
        rgb = (1.0 - fade) * new + fade * old
        rgb = np.where(covered[..., None], rgb, np.array(BG, np.float32)).astype(np.uint8)

        pg = self.pygame
        surf = pg.surfarray.make_surface(np.ascontiguousarray(rgb[:, ::-1]))  # screen y is down
        w, h = covered.shape
        scale = t[0]
        size = (max(1, round(w * res * scale)), max(1, round(h * res * scale)))
        # Pixel [i, j] is centred on origin + (i, j) * res, so the top-left corner of the
        # crop is half a pixel left of its first column and above its last row.
        corner = np.array([[ox + (x0 - 0.5) * res, oy + (y0 + h - 0.5) * res]])
        self.screen.blit(pg.transform.scale(surf, size), self._to_screen(corner, t)[0])

    def _draw_depth(self, env, t):
        pg = self.pygame
        s = env.car.state
        p = env.depth.p
        fov = math.radians(p.fov_deg)
        width = fov / p.n_buckets
        centres = -fov / 2.0 + width * (np.arange(p.n_buckets) + 0.5)

        truth = env.depth.true_ranges(env.world, s)
        latest = env.obs_builder.latest_frame()
        seen = env.obs_builder.depth_slice(latest) if latest is not None else None
        origin = np.array([s.x + math.cos(s.theta) * p.mount_forward,
                           s.y + math.sin(s.theta) * p.mount_forward])
        o_scr = self._to_screen(origin[None, :], t)[0]

        for i, c in enumerate(centres):
            ang = s.theta + c
            d = float(truth[i])
            end = origin + np.array([math.cos(ang), math.sin(ang)]) * d
            pg.draw.line(self.screen, RAY_TRUE, o_scr, self._to_screen(end[None, :], t)[0], 1)

            if seen is not None:
                # Un-normalise the value the policy actually got, so dropout and latency
                # show up as a marker sitting away from the true hit.
                d_seen = (float(seen[i]) + 1.0) / 2.0 * env.cfg.obs.norm_depth_max
                pt = origin + np.array([math.cos(ang), math.sin(ang)]) * d_seen
                pg.draw.circle(self.screen, RAY_SEEN, self._to_screen(pt[None, :], t)[0], 3)

    def _draw_ultrasonic(self, env, t):
        pg = self.pygame
        s = env.car.state
        ranges = env.ultra.true_ranges(env.world, s)
        for i, d in enumerate(ranges):
            ang = s.theta + float(env.ultra.angles[i])
            mount = env.ultra.mount(i)
            c, sn = math.cos(s.theta), math.sin(s.theta)
            origin = np.array([s.x + mount[0] * c - mount[1] * sn,
                               s.y + mount[0] * sn + mount[1] * c])
            end = origin + np.array([math.cos(ang), math.sin(ang)]) * float(d)
            pg.draw.line(
                self.screen, ULTRA,
                self._to_screen(origin[None, :], t)[0],
                self._to_screen(end[None, :], t)[0], 2,
            )

    def _draw_hud(self, env):
        s = env.car.state
        lines = [
            f"step {env.steps:5d}/{env.cfg.max_steps}   dt {env.dt * 1000:5.1f} ms",
            f"speed {s.speed:+5.2f} m/s   steer {math.degrees(s.steer):+6.1f} deg",
            f"depth conf {env.depth.confidence:4.2f}   fov {env.depth.p.fov_deg:5.1f} deg",
            f"stalled {env.reward_fn.stalled_steps:3d}/{env.cfg.reward.stall_limit}   "
            f"coverage {env.reward_fn.coverage_m2:5.1f} m2",
        ]
        for i, text in enumerate(lines):
            colour = WARN if "stalled" in text and env.reward_fn.stalled_steps > 0 else TEXT
            self.screen.blit(self.font.render(text, True, colour), (12, 10 + i * 18))

        overlay = env.hud_overlay
        for i, text in enumerate(overlay):
            y = self.size[1] - 10 - (len(overlay) - i) * 18
            self.screen.blit(self.font.render(text, True, TEXT), (12, y))

    def close(self):
        self.pygame.display.quit()
        self.pygame.quit()
