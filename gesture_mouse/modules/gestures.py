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

        # ---- Feature A: Scroll (finger-count on left hand) --------
        self._last_scroll_time: float  = 0.0
        self._scroll_direction: str | None = None   # "up" or "down", latched

        # ---- Feature B: Zoom (finger-count on right hand) ----------
        self._last_zoom_time: float   = 0.0

        # ---- Feature C: Air Whiteboard ---------------------------------
        h, w = frame_shape
        self._canvas            = np.zeros((h, w, 3), dtype=np.uint8)
        self._prev_draw_pt: tuple | None = None
        self._color_idx: int             = 0
        self._draw_color: tuple          = config.WHITEBOARD_COLORS[0]
        self._last_color_switch: float   = 0.0

        # ---- Feature D: Volume / Brightness (open-palm) ----------------
        self._prev_media_wrist: tuple | None = None
        self._last_media_time: float         = 0.0

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
            if self.mode == "MOUSE":
                # Scroll uses LEFT hand finger count (independent of right hand)
                if secondary is not None:
                    scroll_g = self._finger_scroll(secondary)
                    if scroll_g != "NONE":
                        gesture = scroll_g

            elif self.mode == "WHITEBOARD" and primary:
                gesture = self._whiteboard(primary, state, w, h)

            elif self.mode == "ZOOM" and primary:
                gesture = self._finger_zoom(primary)

            elif self.mode == "MEDIA" and primary:
                gesture = self._media_control(primary, state, w, h)
        else:
            # No hand — reset continuity state.
            self._prev_draw_pt     = None
            self._prev_media_wrist = None

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

    def _finger_scroll(self, secondary_landmarks) -> str:
        """
        Scroll using finger count on the LEFT hand.

        1 finger up (index)          → scroll UP
        2+ fingers up (index+middle) → scroll DOWN
        0 fingers up or hand removed → reset direction

        Uses STICKY direction: once scroll starts in a direction, it
        stays that way even if the middle finger flickers. Direction
        only changes when all fingers are put down (reset).
        """
        fingers = count_fingers_up(secondary_landmarks)
        n_up = sum(fingers[1:])   # non-thumb fingers

        # No fingers up → reset direction latch
        if n_up == 0:
            self._scroll_direction = None
            return "NONE"

        # At least one non-thumb finger is up — latch direction
        if self._scroll_direction is None:
            if n_up == 1:
                self._scroll_direction = "up"
            else:
                self._scroll_direction = "down"

        now = time.time()
        if now - self._last_scroll_time < config.SCROLL_COOLDOWN_INTERVAL:
            return "NONE"

        if self._scroll_direction == "up":
            try:
                pyautogui.scroll(config.SCROLL_AMOUNT)
            except pyautogui.FailSafeException:
                pass
            self._last_scroll_time = now
            return "SCROLL_UP"
        else:
            try:
                pyautogui.scroll(-config.SCROLL_AMOUNT)
            except pyautogui.FailSafeException:
                pass
            self._last_scroll_time = now
            return "SCROLL_DOWN"

    def set_scroll_sensitivity(self, value: float) -> None:
        """M4 context-aware profiles call this to adjust scroll speed."""
        self._scroll_sensitivity = value

    # ------------------------------------------------------------------
    # Feature B: Zoom (finger-count on right hand)
    # ------------------------------------------------------------------

    def _finger_zoom(self, landmarks) -> str:
        """
        Zoom using hand pose on the RIGHT hand (ZOOM mode only).

        Open palm (all 5 fingers)  → zoom IN  (Ctrl++)
        Fist (0 non-thumb fingers) → zoom OUT (Ctrl+-)
        Any other pose             → no zoom

        Open palm and fist are the most reliable MediaPipe detections,
        avoiding the middle-finger flicker issue that broke index-only.
        """
        fingers = count_fingers_up(landmarks)
        n_up = sum(fingers[1:])   # non-thumb fingers

        now = time.time()
        if now - self._last_zoom_time < config.ZOOM_COOLDOWN:
            return "NONE"

        # Open palm (3+ non-thumb fingers up) → zoom in
        if n_up >= 3:
            try:
                pyautogui.hotkey("ctrl", "=")
            except pyautogui.FailSafeException:
                pass
            self._last_zoom_time = now
            return "ZOOM_IN"

        # Fist (0 non-thumb fingers up) → zoom out
        if n_up == 0:
            try:
                pyautogui.hotkey("ctrl", "-")
            except pyautogui.FailSafeException:
                pass
            self._last_zoom_time = now
            return "ZOOM_OUT"

        return "NONE"

    # ------------------------------------------------------------------
    # Feature D: Volume / Brightness (open-palm wrist tracking)
    # ------------------------------------------------------------------

    def _media_control(self, landmarks, state: dict,
                       frame_w: int, frame_h: int) -> str:
        """
        Volume and brightness control in MEDIA mode.

        Gesture mapping (distinct gestures to avoid conflict):
            Index finger only up  → VOLUME control (wrist X movement)
                Move hand right → volume up
                Move hand left  → volume down
            Index + Middle up     → BRIGHTNESS control (wrist Y movement)
                Move hand up    → brightness up
                Move hand down  → brightness down
        """
        if landmarks is None:
            self._prev_media_wrist = None
            return "NONE"

        fingers = count_fingers_up(landmarks)
        # Index only up  → volume
        is_volume_pose     = (fingers[1] and not fingers[2]
                              and not fingers[3] and not fingers[4])
        # Index + Middle up → brightness
        is_brightness_pose = (fingers[1] and fingers[2]
                              and not fingers[3] and not fingers[4])

        if not is_volume_pose and not is_brightness_pose:
            self._prev_media_wrist = None
            return "NONE"

        wrist_x = int(landmarks[config.LM_WRIST].x * frame_w)
        wrist_y = int(landmarks[config.LM_WRIST].y * frame_h)

        if self._prev_media_wrist is None:
            self._prev_media_wrist = (wrist_x, wrist_y)
            return "NONE"

        dx = wrist_x - self._prev_media_wrist[0]
        dy = wrist_y - self._prev_media_wrist[1]
        self._prev_media_wrist = (wrist_x, wrist_y)

        now = time.time()
        if now - self._last_media_time < config.MEDIA_COOLDOWN:
            return "NONE"

        gesture = "NONE"

        # --- Volume: index-only + wrist horizontal --------------------
        if is_volume_pose and abs(dx) > config.VOLUME_DEADZONE:
            try:
                if dx > 0:
                    for _ in range(config.VOLUME_STEP):
                        pyautogui.press("volumeup")
                    gesture = "VOL_UP"
                else:
                    for _ in range(config.VOLUME_STEP):
                        pyautogui.press("volumedown")
                    gesture = "VOL_DOWN"
                self._last_media_time = now
            except Exception:
                pass

        # --- Brightness: peace-sign + wrist vertical ------------------
        if is_brightness_pose and abs(dy) > config.BRIGHTNESS_DEADZONE:
            try:
                import screen_brightness_control as sbc
                current = sbc.get_brightness(display=0)
                if isinstance(current, list):
                    current = current[0]
                if dy < 0:   # wrist moved up → brightness up
                    new_val = min(100, current + config.BRIGHTNESS_STEP)
                    gesture = "BRIGHT_UP"
                else:        # wrist moved down → brightness down
                    new_val = max(0, current - config.BRIGHTNESS_STEP)
                    gesture = "BRIGHT_DOWN"
                sbc.set_brightness(new_val, display=0)
                self._last_media_time = now
            except ImportError:
                pass
            except Exception:
                pass

        return gesture

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
        self.mode              = mode
        self._prev_draw_pt     = None
        self._prev_media_wrist = None
        # The canvas is intentionally NOT cleared on mode switch so
        # the user can return to their whiteboard drawing.
