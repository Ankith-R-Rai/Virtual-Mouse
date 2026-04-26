"""
utils/gesture_utils.py — Finger-state classifiers built on raw MediaPipe landmarks.

All functions accept a list of 21 NormalizedLandmark objects (one hand) and
return simple booleans or lists.  No pyautogui / cv2 imports here.
"""

import sys
import os

# Allow importing config from the parent package regardless of how the script
# is invoked (standalone test vs. main application).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Single-finger state
# ---------------------------------------------------------------------------

def is_finger_up(landmarks: list, tip_idx: int, pip_idx: int) -> bool:
    """
    Check if a finger is extended (tip above PIP joint).

    In MediaPipe's coordinate system y increases downward, so "tip above PIP"
    means tip.y < pip.y.

    Args:
        landmarks : list of 21 NormalizedLandmark objects
        tip_idx   : landmark index of the finger tip
        pip_idx   : landmark index of the PIP (proximal interphalangeal) joint

    Returns:
        True if the finger appears extended
    """
    return landmarks[tip_idx].y < landmarks[pip_idx].y


# ---------------------------------------------------------------------------
# Full-hand finger state
# ---------------------------------------------------------------------------

def count_fingers_up(landmarks: list) -> list[bool]:
    """
    Return a list of five booleans: [thumb, index, middle, ring, pinky].

    Thumb uses a y-axis heuristic (tip above THUMB_MCP) which works reliably
    when the hand is held upright facing the camera.  For the four fingers the
    standard tip-vs-PIP comparison is used.

    Returns:
        [thumb_up, index_up, middle_up, ring_up, pinky_up]
    """
    fingers = []

    # Thumb: tip y < THUMB_MCP y  (landmark 2 = THUMB_MCP)
    thumb_up = landmarks[config.LM_THUMB_TIP].y < landmarks[config.LM_THUMB_MCP].y
    fingers.append(thumb_up)

    # Four fingers: tip y < PIP y
    for tip_idx, pip_idx in [
        (config.LM_INDEX_TIP,  config.LM_INDEX_PIP),
        (config.LM_MIDDLE_TIP, config.LM_MIDDLE_PIP),
        (config.LM_RING_TIP,   config.LM_RING_PIP),
        (config.LM_PINKY_TIP,  config.LM_PINKY_PIP),
    ]:
        fingers.append(is_finger_up(landmarks, tip_idx, pip_idx))

    return fingers  # [thumb, index, middle, ring, pinky]


# ---------------------------------------------------------------------------
# Composite gesture predicates
# ---------------------------------------------------------------------------

def is_fist(landmarks: list) -> bool:
    """
    All four fingers curled (thumb state is intentionally ignored so that
    natural fist and thumbs-up both register as a fist for erase purposes).
    """
    for tip_idx, pip_idx in [
        (config.LM_INDEX_TIP,  config.LM_INDEX_PIP),
        (config.LM_MIDDLE_TIP, config.LM_MIDDLE_PIP),
        (config.LM_RING_TIP,   config.LM_RING_PIP),
        (config.LM_PINKY_TIP,  config.LM_PINKY_PIP),
    ]:
        if is_finger_up(landmarks, tip_idx, pip_idx):
            return False
    return True


def is_index_only(landmarks: list) -> bool:
    """Only the index finger is extended — used for whiteboard draw mode."""
    fingers = count_fingers_up(landmarks)
    # index=True, middle=False, ring=False, pinky=False
    return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]


def is_index_middle_up(landmarks: list) -> bool:
    """
    Index and middle fingers extended, ring and pinky curled.

    Used as the entry condition for scroll detection in M1 and M3.
    """
    fingers = count_fingers_up(landmarks)
    return fingers[1] and fingers[2] and not fingers[3] and not fingers[4]


def is_pinch(landmarks: list, tip_a_idx: int, tip_b_idx: int,
             frame_w: int, frame_h: int, threshold_px: float) -> bool:
    """
    Generic pinch detector: returns True if two landmark tips are within
    threshold_px of each other (measured in frame pixel space).

    Args:
        landmarks    : list of 21 NormalizedLandmark objects
        tip_a_idx    : first landmark index
        tip_b_idx    : second landmark index
        frame_w/h    : frame dimensions for denormalization
        threshold_px : maximum pixel distance to qualify as pinch

    Returns:
        True if the two tips are within threshold_px
    """
    from utils.math_utils import euclidean_distance, landmark_to_pixel

    a = landmark_to_pixel(landmarks[tip_a_idx], frame_w, frame_h)
    b = landmark_to_pixel(landmarks[tip_b_idx], frame_w, frame_h)
    return euclidean_distance(a, b) < threshold_px


def get_finger_count(landmarks: list) -> int:
    """Return the total number of extended fingers (0–5)."""
    return sum(count_fingers_up(landmarks))
