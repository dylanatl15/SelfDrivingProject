"""Viewport colour ramps. Drawing needs a display; the arithmetic behind it does not."""

import numpy as np

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
