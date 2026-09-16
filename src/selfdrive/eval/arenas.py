"""Hand-built arenas for showing the policy off, not for training it.

The random generators draw what the policy trains on, and that is deliberately harsher than
any real test space: cone densities down to 0.02 per square metre, which a car can drive
straight through, and interior walls whose doorways can be 0.30 m wide. A demo run wants the
floor the car will actually be driven on. These six are built to that instead:

*   no opening between obstacles narrower than `MIN_GAP` 0.60 m, walls included;
*   cones from 0.06 m across (a small marker) to 1.00 m (a drum);
*   a spawn the car can drive out of, on floor connected to the rest of the arena.

Goals are still drawn by the environment, from the floor this car can reach
(`world/reachability.py`), so a showcase goal is never sealed off either.

    python -m selfdrive.eval.evaluate --model <ckpt> --config <env.yaml> \\
        --render human --arena cone_slalom          # or --arena all to cycle through them

Two families take their names from what the car meets: `cone_*` are open ground with dense
cones, `office_loop` and `warehouse_aisles` are walls only, and the last two mix them.
`tests/test_arenas.py` pins the three rules above.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..world.generators import rect_walls
from ..world.geometry import World

MIN_GAP = 0.60  # m: the narrowest opening the real test space will have
MARKER_R = 0.04  # a small floor marker, 0.08 m across
BOLLARD_R = 0.06
CONE_R = 0.15  # a traffic cone, 0.30 m across
DRUM_R = 0.45  # a barrel, 0.90 m across


@dataclass(frozen=True)
class Arena:
    """One hand-built floor: its obstacles and where the car starts on it."""

    name: str
    description: str
    world: World
    pose: tuple[float, float, float]

    def options(self) -> dict:
        """What `CarEnv.reset` needs to place this arena and this start pose."""
        return {"world": self.world, "pose": self.pose}


def _outer(width: float, height: float) -> list[list[float]]:
    return rect_walls(-width / 2.0, -height / 2.0, width / 2.0, height / 2.0)


def _wall(x0: float, y0: float, x1: float, y1: float,
          *doors: tuple[float, float]) -> list[list[float]]:
    """A straight wall broken by openings, each `(centre along the wall, width)` in metres."""
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    out, at = [], 0.0
    for centre, width in sorted(doors):
        start, end = centre - width / 2.0, centre + width / 2.0
        if start - at > 1e-6:
            out.append([x0 + ux * at, y0 + uy * at, x0 + ux * start, y0 + uy * start])
        at = end
    if length - at > 1e-6:
        out.append([x0 + ux * at, y0 + uy * at, x1, y1])
    return out


def _ellipse(a: float, b: float, n: int, r: float) -> list[list[float]]:
    """`n` cones of radius `r` evenly spaced in angle around an ellipse."""
    return [[a * math.cos(2 * math.pi * k / n), b * math.sin(2 * math.pi * k / n), r]
            for k in range(n)]


def cone_slalom() -> Arena:
    """An 18 x 10 m lot: eight gates to weave through, markers down both flanks."""
    cones = []
    for k in range(8):
        x, centre = -7.0 + 2.0 * k, (0.9 if k % 2 == 0 else -0.9)
        cones += [[x, centre + 0.75, CONE_R], [x, centre - 0.75, CONE_R]]  # 1.2 m between
    for side in (1.0, -1.0):
        for x in (-7.5, -6.0, -4.5, -3.0, -1.5, 1.5, 3.0, 4.5, 6.0, 7.5):
            cones.append([x, side * 3.5, MARKER_R])
    cones += [[0.0, 3.6, DRUM_R], [0.0, -3.6, DRUM_R]]
    return Arena(
        name="cone_slalom",
        description="Open lot, 38 cones: a line of 1.2 m gates with clutter down the flanks.",
        world=World(_outer(18.0, 10.0), cones, bounds=(-9.0, -5.0, 9.0, 5.0)),
        pose=(-8.0, 0.0, 0.0),
    )


def cone_yard() -> Arena:
    """A 16 x 12 m yard coned off as a course: an outer edge, an island, four drums."""
    cones = _ellipse(6.5, 4.5, 28, MARKER_R) + _ellipse(3.0, 1.8, 16, MARKER_R)
    cones += [[x, y, DRUM_R] for x, y in ((3.4, 2.4), (-3.4, 2.4), (3.4, -2.4), (-3.4, -2.4))]
    return Arena(
        name="cone_yard",
        description="A coned course: a 3.5 m lane round an island, with drums parked in it.",
        world=World(_outer(16.0, 12.0), cones, bounds=(-8.0, -6.0, 8.0, 6.0)),
        pose=(4.75, 0.0, math.pi / 2),
    )


def office_loop() -> Arena:
    """An 18 x 12 m floor plan: three rooms off a hall, an island to drive round, one
    dead end wide enough to reverse out of."""
    walls = _outer(18.0, 12.0)
    walls += _wall(-9.0, 2.0, 9.0, 2.0, (3.0, 1.0), (9.0, 1.0), (15.0, 1.0))  # doors at -6, 0, 6
    walls += [[-3.0, 2.0, -3.0, 6.0], [3.0, 2.0, 3.0, 6.0]]  # the partitions between the rooms
    walls += rect_walls(-2.0, -3.5, 2.0, -0.5)  # the island: 2.5 m of hall all round it
    walls += [[5.0, -6.0, 5.0, -2.5], [7.2, -6.0, 7.2, -2.5]]  # a 2.2 m dead end
    return Arena(
        name="office_loop",
        description="Rooms off a hall through 1.0 m doors, a loop round an island, one dead end.",
        world=World(walls, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(-7.5, -4.0, 0.0),
    )


def warehouse_aisles() -> Arena:
    """A 20 x 12 m warehouse: four racks with 2.4 m aisles, one pass-through, a store room."""
    walls = _outer(20.0, 12.0)
    for y in (-3.6, -1.2, 3.6):
        walls.append([-6.5, y, 6.5, y])
    walls += _wall(-6.5, 1.2, 6.5, 1.2, (6.5, 1.0))  # one rack has a 1.0 m pass-through
    walls += [[8.0, -6.0, 8.0, -1.5]] + _wall(8.0, -1.5, 10.0, -1.5, (1.0, 1.0))
    return Arena(
        name="warehouse_aisles",
        description="Racking with 2.4 m aisles, a 1.0 m pass-through and a store room door.",
        world=World(walls, bounds=(-10.0, -6.0, 10.0, 6.0)),
        pose=(-8.25, 0.0, math.pi / 2),
    )


def loading_bay() -> Arena:
    """An 18 x 12 m yard against a building: a roller door, a parked trailer, a coned lane."""
    walls = _outer(18.0, 12.0)
    walls += _wall(-9.0, 3.0, 9.0, 3.0, (5.0, 1.4), (13.5, 0.9))  # roller door, personnel door
    walls += [[0.0, 3.0, 0.0, 6.0]]  # the wall between the two store rooms
    walls += rect_walls(-7.0, -2.5, -2.0, -1.0)  # a parked trailer
    cones = [[x, y, CONE_R] for y in (1.2, -0.6) for x in (0.5, 2.0, 3.5, 5.0, 6.5)]
    cones += [[7.5, -4.0, DRUM_R], [6.0, -4.8, DRUM_R]]
    cones += [[x, y, MARKER_R] for x, y in ((-1.0, -0.2), (-1.0, -3.2), (-8.0, -4.5), (0.0, -5.0))]
    return Arena(
        name="loading_bay",
        description="A yard with a 1.4 m roller door, a parked trailer and a 1.5 m coned lane.",
        world=World(walls, cones, bounds=(-9.0, -6.0, 9.0, 6.0)),
        pose=(0.0, -4.0, 0.0),
    )


def campus_path() -> Arena:
    """A 16 x 12 m outdoor space: two buildings with a passage between, benches, bollards."""
    walls = _outer(16.0, 12.0)
    walls += rect_walls(-7.0, 1.5, -2.0, 5.0) + rect_walls(2.0, 1.5, 7.0, 5.0)
    walls += [[-5.0, -1.0, -3.5, -1.0], [3.5, -1.0, 5.0, -1.0]]  # benches
    cones = [[0.0, y, BOLLARD_R] for y in (-3.0, -1.8, -0.6, 0.6)]  # a line of bollards
    cones += [[x, y, DRUM_R] for x, y in ((-6.0, -4.0), (6.0, -4.0), (0.0, -4.5))]  # planters
    return Arena(
        name="campus_path",
        description="Two buildings with a 4 m passage, benches, and bollards 1.1 m apart.",
        world=World(walls, cones, bounds=(-8.0, -6.0, 8.0, 6.0)),
        pose=(-6.0, -2.5, 0.0),
    )


ARENAS: dict[str, Arena] = {a.name: a for a in (
    cone_slalom(), cone_yard(), office_loop(), warehouse_aisles(), loading_bay(), campus_path(),
)}


def arena_cycle(name: str) -> Callable[[int], dict]:
    """Reset options per episode: one named arena every time, or `all` in turn."""
    if name == "all":
        chosen = list(ARENAS.values())
    elif name in ARENAS:
        chosen = [ARENAS[name]]
    else:
        raise SystemExit(f"unknown arena {name!r}; have: {', '.join(ARENAS)}, or all")
    return lambda episode: chosen[episode % len(chosen)].options()
