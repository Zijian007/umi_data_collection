"""The one frame shared by the trajectory and the calibration dataset.

An Insta360 X3 360 4K60 recording is one 3840x1920 picture: the front lens is the
right half, the rear lens the left half. The r1440 calibration belongs to a 1440
centre crop of that half. ``profiles.default_crop_size`` for this mode is 960;
this tool does not use it. Every second frame is kept, so 59.94 fps becomes 29.97.
"""

from __future__ import annotations

from software.camera.x3.profiles import plan_lens_crop

PROFILE = "x3_360_4k_60"
SRC_W, SRC_H = 3840, 1920
CROP = 1440
STEP = 2
T0_S = 1.0                                  # EuRoC files shift the clock by one second
FPS_SRC = 60000.0 / 1001.0
FPS_USED = 30000.0 / 1001.0
LENS_HALF = {"front": "right", "rear": "left"}

# The crop the rest of the tool was checked against. plan_lens_crop must keep producing it.
_LOCKED = {"right": (2160, 240, 1440, 1440), "left": (240, 240, 1440, 1440)}


def crop_xywh(half: str) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` of one lens in the 3840x1920 frame. ``half`` is ``left`` or ``right``."""
    if half not in _LOCKED:
        raise ValueError(f"lens half must be 'left' or 'right', got {half!r}")
    box = plan_lens_crop(SRC_W, SRC_H, half, CROP)["crop_xywh_abs"]
    got = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if got != _LOCKED[half]:
        raise RuntimeError(f"{half} lens crop is {got}, the r1440 frame is {_LOCKED[half]}")
    return got


crop_xywh("right")
crop_xywh("left")
