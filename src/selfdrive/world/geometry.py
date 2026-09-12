"""Vectorized 2D geometry: ray casting, collision, and clearance queries.

A world is two flat float arrays - `segments` (walls) and `circles` (cones, posts).
Every sensor ray is intersected against every primitive in a single numpy operation, so
a full 12-ray sweep costs tens of microseconds. That matters here: the policy is a small
MLP over ~17 floats, so training throughput is bound by env stepping on the CPU, not by
the GPU, and geometry is the hot path.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D scalar cross product, broadcasting over any leading axes."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def rot(theta: float) -> np.ndarray:
    """Rotation matrix for `theta` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def obb_corners(x: float, y: float, theta: float, length: float, width: float) -> np.ndarray:
    """The four corners of the car body: front-left, rear-left, rear-right, front-right."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return local @ rot(theta).T + np.array([x, y])


def point_seg_distance(pts: np.ndarray, a: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Distance from every point to every segment. pts (P,2), a (N,2), e (N,2) -> (P,N)."""
    ap = pts[:, None, :] - a[None, :, :]
    ee = np.sum(e * e, axis=-1)[None, :]
    t = np.clip(np.sum(ap * e[None, :, :], axis=-1) / np.maximum(ee, EPS), 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * e[None, :, :]
    return np.linalg.norm(pts[:, None, :] - closest, axis=-1)


def segments_cross(p: np.ndarray, pe: np.ndarray, q: np.ndarray, qe: np.ndarray) -> np.ndarray:
    """Proper crossing test between two segment sets. p,pe (P,2); q,qe (N,2) -> (P,N) bool.

    Collinear overlap is deliberately not reported. It has measure zero, and the callers
    that care (collision) also run a containment test that covers the degenerate case.
    """
    w = q[None, :, :] - p[:, None, :]
    pe_b, qe_b = pe[:, None, :], qe[None, :, :]
    denom = cross2(pe_b, qe_b)
    parallel = np.abs(denom) < EPS
    safe = np.where(parallel, 1.0, denom)
    t = cross2(w, qe_b) / safe
    u = cross2(w, pe_b) / safe
    return (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def obb_sdf(pts: np.ndarray, x: float, y: float, theta: float, length: float,
            width: float) -> np.ndarray:
    """Signed distance from points to the car box. Negative inside, zero on the surface."""
    local = (pts - np.array([x, y])) @ rot(theta)  # v @ R == R.T @ v
    d = np.abs(local) - np.array([length / 2.0, width / 2.0])
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


class World:
    """The static obstacle set for one episode.

    Segments are `(N, 4)` rows of `x1, y1, x2, y2`; circles are `(M, 3)` rows of
    `cx, cy, r`. Both may be empty. `bounds` is an optional `(xmin, ymin, xmax, ymax)`
    used for spawning and for framing the renderer.
    """

    def __init__(self, segments=None, circles=None, bounds=None):
        seg = (np.zeros((0, 4)) if segments is None
               else np.asarray(segments, dtype=float).reshape(-1, 4))
        cir = (np.zeros((0, 3)) if circles is None
               else np.asarray(circles, dtype=float).reshape(-1, 3))
        self.segments = seg
        self.circles = cir
        self.seg_a = seg[:, :2].copy()
        self.seg_e = seg[:, 2:] - seg[:, :2]
        self.circ_c = cir[:, :2].copy()
        self.circ_r = cir[:, 2].copy()
        self.bounds = bounds

    @property
    def n_primitives(self) -> int:
        return self.seg_a.shape[0] + self.circ_c.shape[0]

    def raycast(self, origins: np.ndarray, directions: np.ndarray, max_range: float,
                normalize: bool = True) -> np.ndarray:
        """Cast R rays and return the distance to the first hit, clipped to `max_range`.

        Pass `normalize=False` when the caller already supplies unit directions - the
        sensors build theirs from cos/sin, and this runs on every step of every worker.
        """
        origins = np.asarray(origins, dtype=float).reshape(-1, 2)
        directions = np.asarray(directions, dtype=float).reshape(-1, 2)
        if normalize:
            directions = directions / np.maximum(
                np.linalg.norm(directions, axis=1, keepdims=True), EPS
            )
        best = np.full(origins.shape[0], np.inf)

        if self.seg_a.shape[0]:
            o = origins[:, None, :]
            d = directions[:, None, :]
            ao = self.seg_a[None, :, :] - o
            e = self.seg_e[None, :, :]
            denom = cross2(d, e)
            parallel = np.abs(denom) < EPS
            safe = np.where(parallel, 1.0, denom)
            t = cross2(ao, e) / safe
            u = cross2(ao, d) / safe
            ok = (~parallel) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
            best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))

        if self.circ_c.shape[0]:
            oc = origins[:, None, :] - self.circ_c[None, :, :]
            b = np.sum(directions[:, None, :] * oc, axis=-1)
            c = np.sum(oc * oc, axis=-1) - self.circ_r[None, :] ** 2
            disc = b * b - c
            sq = np.sqrt(np.maximum(disc, 0.0))
            near, far = -b - sq, -b + sq
            # Falling back to the far root keeps a ray that starts inside a circle sane.
            t = np.where(near >= 0.0, near, far)
            ok = (disc >= 0.0) & (t >= 0.0)
            best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))

        return np.minimum(best, max_range)

    def collides(self, x: float, y: float, theta: float, length: float, width: float) -> bool:
        """True if the car body overlaps any obstacle."""
        corners = obb_corners(x, y, theta, length, width)
        edges_e = np.roll(corners, -1, axis=0) - corners

        if self.seg_a.shape[0]:
            if segments_cross(corners, edges_e, self.seg_a, self.seg_e).any():
                return True
            # A wall short enough to sit entirely inside the body crosses no edge.
            ends = np.vstack([self.seg_a, self.seg_a + self.seg_e])
            if (obb_sdf(ends, x, y, theta, length, width) <= 0.0).any():
                return True

        if self.circ_c.shape[0]:
            if (obb_sdf(self.circ_c, x, y, theta, length, width) <= self.circ_r).any():
                return True

        return False

    def clearance(self, x: float, y: float, theta: float, length: float, width: float) -> float:
        """Distance from the car body to the nearest obstacle, `inf` in an empty world.

        This is ground truth, deliberately free of sensor noise: it feeds the proximity
        reward, and a reward computed from noisy observations would teach the policy to
        chase sensor artifacts. It clamps at 0 rather than going negative on overlap -
        the overlap case is `collides()`, which terminates the episode anyway.
        """
        best = np.inf
        corners = obb_corners(x, y, theta, length, width)
        edges_e = np.roll(corners, -1, axis=0) - corners

        if self.seg_a.shape[0]:
            # In 2D the gap between two non-crossing segments is always realised at an
            # endpoint, so both directions have to be checked.
            corner_to_wall = point_seg_distance(corners, self.seg_a, self.seg_e).min()
            ends = np.vstack([self.seg_a, self.seg_a + self.seg_e])
            wall_to_body = point_seg_distance(ends, corners, edges_e).min()
            best = min(best, float(corner_to_wall), float(wall_to_body))

        if self.circ_c.shape[0]:
            d = obb_sdf(self.circ_c, x, y, theta, length, width) - self.circ_r
            best = min(best, float(d.min()))

        return max(best, 0.0)
