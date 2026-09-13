"""YAML configuration loading.

Explicit section mapping rather than reflection over type annotations. With
`from __future__ import annotations` in force everywhere, dataclass field types are
strings at runtime, and the generic version of this would be more magic than the eight
sections it saves.

Angles are stored in radians but written in degrees, because nobody wants to review a
config file full of 0.4887.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .dynamics.base import CarParams
from .envs.car_env import EnvConfig
from .envs.goals import GoalConfig
from .envs.obs import ObsConfig
from .envs.randomize import DomainRandConfig
from .envs.rewards import RewardConfig
from .sensors.depth_arc import DepthArcParams
from .sensors.odometry import OdometryParams
from .sensors.ultrasonic import UltrasonicParams
from .world.generators import ArenaParams

SECTIONS: dict[str, type] = {
    "car": CarParams,
    "depth": DepthArcParams,
    "ultrasonic": UltrasonicParams,
    "odometry": OdometryParams,
    "obs": ObsConfig,
    "reward": RewardConfig,
    "arena": ArenaParams,
    "domain_rand": DomainRandConfig,
    "goal": GoalConfig,
}

# Degree-valued aliases accepted in YAML, mapped to the radian field they set.
DEGREE_ALIASES: dict[str, str] = {
    "max_steer_deg": "max_steer_rad",
    "steer_rate_deg_s": "steer_rate_rad_s",
    "steer_trim_deg": "steer_trim_rad",
}


def _coerce(cls: type, data: dict[str, Any] | None):
    if not data:
        return cls()

    import math

    valid = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        name = key
        # DomainRandConfig has its own *_deg fields that are genuinely in degrees and
        # must not be converted, so only translate an alias the class does not define.
        if key in DEGREE_ALIASES and key not in valid:
            name = DEGREE_ALIASES[key]
            value = math.radians(float(value))
        if name not in valid:
            raise ValueError(
                f"{cls.__name__} has no field {key!r}; valid fields: {sorted(valid)}"
            )
        # YAML gives lists where the dataclasses declare ranges as tuples.
        kwargs[name] = tuple(value) if isinstance(value, list) else value

    return cls(**kwargs)


def env_config_from_dict(data: dict[str, Any] | None) -> EnvConfig:
    data = dict(data or {})
    unknown = set(data) - set(SECTIONS) - {"dt", "max_steps"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")

    kwargs: dict[str, Any] = {name: _coerce(cls, data.get(name)) for name, cls in SECTIONS.items()}
    if "dt" in data:
        kwargs["dt"] = float(data["dt"])
    if "max_steps" in data:
        kwargs["max_steps"] = int(data["max_steps"])
    return EnvConfig(**kwargs)


def load_env_config(path: str | Path | None) -> EnvConfig:
    """Load an environment config, or return defaults when `path` is None."""
    if path is None:
        return EnvConfig()
    with open(path) as fh:
        return env_config_from_dict(yaml.safe_load(fh) or {})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def describe(obj: Any, indent: int = 0) -> str:
    """Flatten a nested config to readable lines, for run logs and the written report."""
    out = []
    pad = "  " * indent
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            out.append(f"{pad}{f.name}:")
            out.append(describe(value, indent + 1))
        else:
            out.append(f"{pad}{f.name}: {value}")
    return "\n".join(out)
