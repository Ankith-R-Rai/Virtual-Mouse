"""
utils/math_utils.py — Pure mathematical helpers used across all modules.

No side effects, no imports from other project files.
All functions operate on plain Python scalars or NumPy arrays.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """
    Euclidean distance between two 2-D points.

    Args:
        p1, p2: (x, y) tuples in any consistent unit (pixels, normalized, etc.)

    Returns:
        float distance
    """
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def map_to_screen(norm_x: float, norm_y: float,
                  screen_w: int, screen_h: int) -> tuple[int, int]:
    """
    Map MediaPipe normalized [0, 1] coordinates to screen pixel coordinates.

    np.interp handles values outside [0, 1] by clamping to the output range.

    Args:
        norm_x, norm_y : normalized landmark coordinates
        screen_w, screen_h : target screen resolution

    Returns:
        (screen_x, screen_y) as integers
    """
    sx = int(np.interp(norm_x, [0.0, 1.0], [0, screen_w]))
    sy = int(np.interp(norm_y, [0.0, 1.0], [0, screen_h]))
    return sx, sy


def landmark_to_pixel(landmark, frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Convert a single MediaPipe NormalizedLandmark to frame pixel coordinates.

    Args:
        landmark  : mediapipe landmark object with .x and .y in [0, 1]
        frame_w, frame_h : webcam frame dimensions

    Returns:
        (px, py) as integers
    """
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_value(current: float, previous: float, factor: float) -> float:
    """
    Exponential moving-average (EMA) for a single scalar.

    Formula: output = previous + factor * (current - previous)

    factor = 0.0 → output never changes (frozen at previous)
    factor = 1.0 → output = current (no smoothing)
    factor = 0.5 → equal blend of previous and current (default in M1)

    Args:
        current  : new raw measurement
        previous : last smoothed value
        factor   : EMA blend factor in [0, 1]

    Returns:
        Smoothed float value
    """
    return previous + factor * (current - previous)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Rolling buffer statistics
# ---------------------------------------------------------------------------

def rolling_delta(buffer) -> float:
    """
    Compute the signed difference between the last and first value in a buffer.

    Used by M3 velocity-scroll to compute dy over a rolling wrist-y window.

    Args:
        buffer : sequence (list / deque) of numeric values

    Returns:
        float delta, or 0.0 if buffer has fewer than 2 elements
    """
    if len(buffer) < 2:
        return 0.0
    return float(buffer[-1] - buffer[0])


def frame_delta(buffer) -> float:
    """
    Compute the signed difference between the last two values in a buffer.

    Used for per-frame dy computation in velocity scroll.
    """
    if len(buffer) < 2:
        return 0.0
    return float(buffer[-1] - buffer[-2])
