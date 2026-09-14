"""Viewport colour ramps and offscreen frames. A window needs a display; neither of these does."""

import numpy as np

from selfdrive.envs.car_env import CarEnv, EnvConfig
from selfdrive.render.pygame_view import (
    PAINT_FULL,
    PAINT_HALF,
    PAINT_STOPPED,
    PAINT_UNTIMED,
    _speed_colour,
)


def test_trail_runs_red_stopped_to_yellow_at_half_speed_to_green_at_full():
    rgb = _speed_colour(np.array([0.0, 0.5, 1.0, 1.4, np.nan]))
    np.testing.assert_allclose(
        rgb, [PAINT_STOPPED, PAINT_HALF, PAINT_FULL, PAINT_FULL, PAINT_UNTIMED], atol=1e-4
    )


def test_video_frames_leave_audio_and_the_rest_of_pygame_alone():
    # Trainers record eval videos in-process. pygame.init() opened audio there, and its
    # PulseAudio thread hung pygame.quit() when the eval swapped the env back.
    import pygame  # noqa: PLC0415 - the module under test imports it lazily too

    env = CarEnv(EnvConfig(), render_mode="rgb_array")
    env.reset(seed=0)
    try:
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert not pygame.get_init()
        assert pygame.mixer.get_init() is None
    finally:
        env.close()
