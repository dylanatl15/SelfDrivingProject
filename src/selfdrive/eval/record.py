"""Record a demo reel: one clip per showcase arena, streamed into a single mp4.

`evaluate.py --video` buffers every frame in a list before it encodes anything. A
960x720 RGB frame is about 2 MB, so a 50 s clip is ~3 GB and a six-arena reel would ask
for ~19 GB before writing a byte. `StreamSink` hands each frame straight to the encoder
instead, which is what makes a reel of all six arenas practical.

Each arena gets `--seconds` of driving, applied by lowering `max_steps` rather than by
dropping frames, so the clip ends at an episode boundary and the printed score line
describes exactly what is in the video.

    python -m selfdrive.eval.record --model <ckpt> --config <env.yaml> --out media/demo.mp4
"""

from __future__ import annotations

import argparse

from ..config import load_env_config
from .arenas import ARENAS
from .evaluate import _load_policy, run_episodes


class StreamSink:
    """A `frame_sink` for `run_episodes` that encodes as it goes instead of accumulating.

    `run_episodes` only ever calls `.append(frame)`, so this stands in for the list it
    expects without holding more than one frame at a time.
    """

    def __init__(self, writer):
        self.writer = writer
        self.n = 0

    def append(self, frame) -> None:
        self.writer.append_data(frame)
        self.n += 1


def main(argv: list[str] | None = None) -> None:
    from ..envs.car_env import CarEnv

    p = argparse.ArgumentParser(description="Record a showcase-arena demo reel.")
    p.add_argument("--model", default=None, help="saved PPO model; omit for a random policy")
    p.add_argument("--config", default="configs/env_waypoint_s2_patience_shield.yaml")
    p.add_argument("--out", default="media/demo.mp4")
    p.add_argument("--seconds", type=float, default=20.0, help="driving time per arena")
    p.add_argument("--arena", default="all", help="one arena name, or 'all'")
    p.add_argument("--seed", type=int, default=4_000_000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=18,
                   help="x264 quality, lower is sharper. The viewport is flat colour and "
                        "thin text, which the encoder default smears into mush")
    args = p.parse_args(argv)

    names = list(ARENAS) if args.arena == "all" else [args.arena]
    unknown = [n for n in names if n not in ARENAS]
    if unknown:
        raise SystemExit(f"unknown arena {unknown[0]!r}; have: {', '.join(ARENAS)}, or all")

    cfg = load_env_config(args.config)
    cfg.max_steps = max(1, round(args.seconds / cfg.dt))
    policy = _load_policy(args.model, args.seed)
    env = CarEnv(cfg, render_mode="rgb_array")

    import imageio.v2 as imageio

    # ffmpeg_params lands after the quality-derived flags, so this -crf is the one that
    # takes effect. The HUD is 11 px text; at the imageio default it is unreadable.
    writer = imageio.get_writer(
        args.out, fps=args.fps, macro_block_size=1,
        ffmpeg_params=["-crf", str(args.crf), "-preset", "slow"],
    )
    sink = StreamSink(writer)
    try:
        for i, name in enumerate(names):
            arena = ARENAS[name]
            before = sink.n
            result = run_episodes(
                policy, env, n_episodes=1, seed=args.seed + i, deterministic=True,
                render=True, frame_sink=sink, options=lambda _i, a=arena: a.options(),
            )
            print(f"{name:18s} {sink.n - before:5d} frames  {result}")
    finally:
        writer.close()
        env.close()
    print(f"wrote {args.out} ({sink.n} frames, {sink.n / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
