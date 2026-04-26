"""
modules/gestures.py — M3: Novel Gesture Features

Three features built on top of Kalman-smoothed coordinates from M2:

  A. Velocity-Proportional Scroll  — maps wrist delta-y → scroll amount
  B. Pinch-to-Zoom                 — two-hand thumb distance → Ctrl+/−
  C. Air Whiteboard                — index finger draws on transparent canvas

Design
------
GestureModule is a stateful dispatcher.  Each frame the integration loop
(M5) calls process(state) where state is the dict returned by M1's
process_frame().  process() routes to the correct feature handler based on
self.mode, then returns the enriched state dict.

M4 calls set_scroll_sensitivity() to apply context-aware scroll multipliers.
M5 calls set_mode() when the user switches modes via Tab or voice command.
"""

import sys
import os
import time

import cv2
import numpy as np
import pyautogui

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.math_utils    import euclidean_distance, landmark_to_pixel
from utils.gesture_utils import (
    is_fist, is_index_only, is_pinky_only_up,
    is_finger_up, count_fingers_up,
)


class GestureModule:
    """
    M3 — Novel Gesture Features.

    Public interface
    ----------------
    process(state)                 -> dict   Main per-frame dispatcher.
    set_mode(mode)                           Switch active mode.
    set_scroll_sensitivity(value)            Called by M4 context profiles.
    clear_canvas()                           Called by voice command.
    """

    def __init__(self, frame_shape: tuple = (480, 640)) -> None:
        # Active interaction mode — shared reference with M5 mode switcher.
        self.mode: str = "MOUSE"

        # ---- Feature A: Velocity Scroll --------------------------------
        # Sensitivity is overridden by M4 context profile each 500 ms.
        self._scroll_sensitivity: float = config.SCROLL_SENSITIVITY

        # ---- Feature B: Pinch-to-Zoom ----------------------------------
        self._prev_zoom_dist: float | None = None
        self._last_zoom_time: float        = 0.0

        # ---- Feature C: Air Whiteboard ---------------------------------
        h, w = frame_shape
        self._canvas            = np.zeros((h, w, 3), dtype=np.uint8)
        self._prev_draw_pt: tuple | None = None
        self._color_idx: int             = 0
        self._draw_color: tuple          = config.WHITEBOARD_COLORS[0]
        self._last_color_switch: float   = 0.0

        # ---- State exposed to HUD (M5) ---------------------------------
        self.last_gesture: str = "NONE"

    # ------------------------------------------------------------------
    # Main dispatcher
    # ------------------------------------------------------------------

    def process(self, state: dict) -> dict:
        """
        Dispatch to the correct feature handler for the current mode.

        Reads:
            state["landmarks"]      — list of landmark lists (one per hand)
            state["handedness"]     — ["Left"/"Right", ...]
            state["wrist_y_buffer"] — rolling deque of wrist y-pixels (M1)
            state["frame"]          — BGR numpy array (mutated in-place)
            state["frame_w/h"]      — frame dimensions
            state["mode"]           — optional override from M4/M5

        Writes:
            state["m3_gesture"]     — gesture string for HUD
            state["frame"]          — frame with whiteboard overlay blended
        """
        # Allow M4/M5 to push a mode change through the state dict.
        if "mode" in state:
            self.mode = state["mode"]

        landmarks = state.get("landmarks", [])
        frame     = state.get("frame")
        w         = state.get("frame_w", 640)
        h         = state.get("frame_h", 480)

        # Resize canvas if the frame dimensions changed (e.g. camera restart).
        if self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

        gesture = "NONE"

        primary   = state.get("primary_landmarks")
        secondary = state.get("secondary_landmarks")

        if primary or secondary:
            if self.mode == "MOUSE" and secondary:
                gesture = self._velocity_scroll(secondary, state)

            elif self.mode == "WHITEBOARD" and primary:
                gesture = self._whiteboard(primary, state, w, h)

            elif self.mode == "ZOOM" and primary and secondary:
                gesture = self._pinch_zoom(landmarks, state)
        else:
            # No hand — reset continuity state.
            self._prev_draw_pt   = None
            self._prev_zoom_dist = None
            self._prev_wrist_y   = None

        # Blend the drawing canvas over the frame in every mode so drawings
        # persist when the user temporarily switches away from WHITEBOARD.
        if frame is not None:
            state["frame"] = self._apply_canvas(frame)

        self.last_gesture    = gesture
        state["m3_gesture"]  = gesture
        return state

    # ------------------------------------------------------------------
    # Feature A: Velocity-Proportional Scroll
    # ------------------------------------------------------------------

    def _velocity_scroll(self, landmarks, state: dict) -> str:
        """
        Map wrist vertical velocity to a scroll amount.

        Algorithm
        ---------
        1. Require index + middle fingers raised (scroll posture).
        2. Consume the rolling wrist-y buffer populated by M1 each frame.
        3. Compute per-frame dy = buffer[-1] − buffer[-2].
        4. Apply sensitivity factor (overridden by M4 context profile).
        5. Ignore deltas below the dead-zone to suppress idle jitter.
        6. pyautogui.scroll(−amount)  (positive = up, negative = down).
        """
        if landmarks is None:
            return "NONE"

        buf = state.get("wrist_y_buffer")
        if buf is None or len(buf) < 2:
            return "NONE"

        # Per-frame delta-y in frame pixels.
        dy = float(buf[-1] - buf[-2])

        if abs(dy) < config.VELOCITY_SCROLL_DEADZONE:
            return "NONE"

        # Map dy to scroll units; context profile sensitivity overrides default.
        scroll_amount = int(dy * self._scroll_sensitivity)
        if scroll_amount == 0:
            return "NONE"

        # In pyautogui: positive = scroll up, negative = scroll down.
        # dy > 0 means wrist moved down (y increases downward) → scroll down.
        try:
            pyautogui.scroll(-scroll_amount)
        except pyautogui.FailSafeException:
            pass

        return "SCROLL_UP" if scroll_amount < 0 else "SCROLL_DOWN"

    def set_scroll_sensitivity(self, value: float) -> None:
        """M4 context-aware profiles call this to adjust scroll speed."""
        self._scroll_sensitivity = value

    # ------------------------------------------------------------------
    # Feature B: Pinch-to-Zoom
    # ------------------------------------------------------------------

    def _pinch_zoom(self, all_landmarks: list, state: dict) -> str:
        """
        Two-hand pinch-to-zoom via Ctrl+Plus / Ctrl+Minus.

        Algorithm
        ---------
        1. Require both hands detected (len(all_landmarks) >= 2).
        2. Identify left-hand and right-hand thumb tips from handedness labels.
        3. Compute Euclidean distance D between the two thumb tips.
        4. Compare D with D from the previous frame.
           D increased by ZOOM_THRESHOLD → Ctrl+Plus  (zoom in)
           D decreased by ZOOM_THRESHOLD → Ctrl+Minus (zoom out)
        5. Enforce a per-action cooldown to prevent rapid repeated triggers.
        """
        if len(all_landmarks) < 2:
            self._prev_zoom_dist = None
            return "NONE"

        handedness = state.get("handedness", [])
        w          = state.get("frame_w", 640)
        h          = state.get("frame_h", 480)

        left_landmarks = None
        right_landmarks = None

        for i, lm_list in enumerate(all_landmarks):
            label = handedness[i] if i < len(handedness) else "Right"
            if label == "Left":
                left_landmarks = lm_list
            else:
                right_landmarks = lm_list

        if left_landmarks is None or right_landmarks is None:
            self._prev_zoom_dist = None
            return "NONE"

        # Zoom trigger: Left hand has Index + Middle up
        from utils.gesture_utils import count_fingers_up
        f_left = count_fingers_up(left_landmarks)
        if not (f_left[1] and f_left[2]):
            self._prev_zoom_dist = None
            return "NONE"

        left_thumb  = landmark_to_pixel(left_landmarks[config.LM_THUMB_TIP], w, h)
        right_thumb = landmark_to_pixel(right_landmarks[config.LM_THUMB_TIP], w, h)

        current_dist = euclidean_distance(left_thumb, right_thumb)

        if self._prev_zoom_dist is None:
            self._prev_zoom_dist = current_dist
            return "NONE"

        delta                = current_dist - self._prev_zoom_dist
        self._prev_zoom_dist = current_dist

        now = time.time()
        if now - self._last_zoom_time < config.ZOOM_COOLDOWN:
            return "NONE"

        if delta > config.ZOOM_THRESHOLD:
            try:
                pyautogui.hotkey("ctrl", "+")
            except pyautogui.FailSafeException:
                pass
            self._last_zoom_time = now
            return "ZOOM_IN"

        if delta < -config.ZOOM_THRESHOLD:
            try:
                pyautogui.hotkey("ctrl", "-")
            except pyautogui.FailSafeException:
                pass
            self._last_zoom_time = now
            return "ZOOM_OUT"

        return "NONE"

    # ------------------------------------------------------------------
    # Feature C: Air Whiteboard
    # ------------------------------------------------------------------

    def _whiteboard(self, landmarks, state: dict, frame_w: int, frame_h: int) -> str:
        """
        Freehand drawing on a persistent transparent canvas overlay.

        Gesture → action mapping
        ------------------------
        Index finger only raised  → draw line (fingertip path)
        Fist                      → erase circle at wrist position
        Thumb tip near ring tip   → cycle through color palette
        Any other posture         → pen-up (break line continuity)
        """
        now = time.time()

        # ---- Color switch: Thumb tip ↔ Pinky tip pinch (Generous gap) ---
        thumb_px = landmark_to_pixel(landmarks[config.LM_THUMB_TIP], frame_w, frame_h)
        pinky_px = landmark_to_pixel(landmarks[config.LM_PINKY_TIP], frame_w, frame_h)

        if (euclidean_distance(thumb_px, pinky_px) < 40 and
                now - self._last_color_switch > 0.8):
            self._color_idx      = (self._color_idx + 1) % len(config.WHITEBOARD_COLORS)
            self._draw_color     = config.WHITEBOARD_COLORS[self._color_idx]
            self._last_color_switch = now
            return "COLOR_SWITCH"

        # ---- Erase: closed fist --------------------------------------
        if is_fist(landmarks):
            # Anchor erase to knuckles (Index MCP) instead of wrist
            cx = int(landmarks[config.LM_INDEX_MCP].x * frame_w)
            cy = int(landmarks[config.LM_INDEX_MCP].y * frame_h)
            cv2.circle(
                self._canvas, (cx, cy),
                config.WHITEBOARD_ERASE_RADIUS, (0, 0, 0), -1,
            )
            self._prev_draw_pt = None
            return "ERASE"

        # ---- Draw: index finger extended & NOT a fist ----------------
        # We allow other fingers to slightly flutter as long as it's not a fist.
        is_drawing = is_finger_up(landmarks, config.LM_INDEX_TIP, config.LM_INDEX_PIP) and not is_fist(landmarks)
        if is_drawing:
            screen_pt = state.get("cursor_pos")
            if screen_pt is not None:
                curr_pt = (
                    int(screen_pt[0] * frame_w / config.SCREEN_WIDTH),
                    int(screen_pt[1] * frame_h / config.SCREEN_HEIGHT)
                )
            else:
                curr_pt = landmark_to_pixel(
                    landmarks[config.LM_INDEX_TIP], frame_w, frame_h,
                )
            if self._prev_draw_pt is not None:
                cv2.line(
                    self._canvas,
                    self._prev_draw_pt, curr_pt,
                    self._draw_color,
                    config.WHITEBOARD_THICKNESS,
                )
            self._prev_draw_pt = curr_pt
            return "DRAW"

        # Any other posture lifts the pen.
        self._prev_draw_pt = None
        return "NONE"

    def clear_canvas(self) -> None:
        """Wipe the entire drawing canvas (voice command: 'clear whiteboard')."""
        self._canvas[:] = 0
        self._prev_draw_pt = None

    def _apply_canvas(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the drawing canvas over the webcam frame.

        Only non-zero canvas pixels contribute so the webcam feed shows
        through everywhere that has not been drawn on.

        cv2.addWeighted blends the entire canvas, which means the webcam
        is slightly darkened even in empty areas when WHITEBOARD_ALPHA > 0.
        A mask-based blend is used to avoid this:
            output = frame         where canvas == 0
            output = weighted mix  where canvas != 0
        """
        if self._canvas.shape != frame.shape:
            self._canvas = np.zeros_like(frame)

        # Build a boolean mask: True wherever there is any drawing.
        mask = np.any(self._canvas != 0, axis=2)   # (H, W) bool

        output = frame.copy()
        # Blend only the drawn pixels; untouched areas stay as the webcam frame.
        output[mask] = cv2.addWeighted(
            frame, 1.0 - config.WHITEBOARD_ALPHA,
            self._canvas, config.WHITEBOARD_ALPHA, 0,
        )[mask]
        return output

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """
        Switch the active gesture mode.

        Clears transient per-mode state so there is no carryover between
        modes (e.g. stale zoom distance, dangling draw point).
        """
        if mode not in config.MODES:
            return
        self.mode            = mode
        self._prev_draw_pt   = None
        self._prev_zoom_dist = None
        # The canvas is intentionally NOT cleared on mode switch so
        # the user can return to their whiteboard drawing.
