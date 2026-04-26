"""
modules/ui.py — M5: Playground UI + System Integration

Responsibilities
----------------
1. Render a live HUD overlay on the webcam frame:
     • Mode badge          (top-left)
     • Context profile     (top-left, below mode)
     • Kalman filter status
     • FPS counter         (top-right)
     • Gesture guide       (bottom bar — mode-specific)
     • Voice status        (bottom bar)
     • Landmark skeleton   (optional, drawn by M1)
     • Whiteboard color swatch (WHITEBOARD mode only)

2. Mode switcher — Tab key cycles MOUSE → WHITEBOARD → ZOOM → VOICE.
   Voice commands ("switch to whiteboard") are handled by M4 and reflected
   via state["mode"] each frame.

3. The UIModule is passed references to M2, M3, M4 so it can query their
   status properties directly for the HUD.
"""

import sys
import os
import time
from collections import deque

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Gesture guide text shown in the bottom bar per mode
# ---------------------------------------------------------------------------
_GESTURE_GUIDE: dict[str, str] = {
    "MOUSE":      "L/R-Click: Right hand  |  Scroll: Move Left Hand Up/Down",
    "WHITEBOARD": "Draw: Right Index Up | Erase: Right Fist | Color: Right Thumb+Pinky",
    "ZOOM":       "Hold 2 fingers UP on Left Hand + move hands apart/together",
    "VOICE":      "Say: 'open browser' | 'take screenshot' | 'whiteboard on/off'",
}


class UIModule:
    """
    M5 — Playground UI + System Integration.

    Public interface
    ----------------
    render_hud(state) -> np.ndarray   Draw all HUD elements; return final frame.
    handle_key(key, state)            Respond to keyboard input from the main loop.
    mode                              str property — current active mode.
    fps                               float property — rolling-average FPS.
    """

    def __init__(self, kalman_module=None, gesture_module=None,
                 voice_module=None) -> None:
        """
        Args:
            kalman_module  : KalmanFilterModule (M2) — for status text / color.
            gesture_module : GestureModule      (M3) — for draw color swatch.
            voice_module   : VoiceContextModule (M4) — for profile / voice text.
        """
        self.kalman  = kalman_module
        self.gesture = gesture_module
        self.voice   = voice_module

        # Active mode (kept in sync with M3 / M4 via state dict each frame).
        self._mode: str      = "MOUSE"
        self._mode_idx: int  = 0

        # Rolling FPS computation (last 10 frame timestamps).
        self._ts_buf: deque = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value in config.MODES:
            self._mode     = value
            self._mode_idx = config.MODES.index(value)

    @property
    def fps(self) -> float:
        if len(self._ts_buf) < 2:
            return 0.0
        elapsed = self._ts_buf[-1] - self._ts_buf[0]
        return (len(self._ts_buf) - 1) / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # Key handler (called from the main loop)
    # ------------------------------------------------------------------

    def handle_key(self, key: int, state: dict) -> None:
        """
        Process a single keypress.

        Tab      — cycle through MOUSE → WHITEBOARD → ZOOM → VOICE
        k        — toggle Kalman filter (M2)
        c        — clear whiteboard canvas (M3)
        """
        if key == ord("\t"):
            self._mode_idx = (self._mode_idx + 1) % len(config.MODES)
            self._mode     = config.MODES[self._mode_idx]
            state["mode"]  = self._mode
            if self.gesture:
                self.gesture.set_mode(self._mode)
            print(f"[M5] Mode → {self._mode}")

        elif key == ord("k"):
            if self.kalman:
                enabled = self.kalman.toggle()
                print(f"[M5] Kalman filter {'ON' if enabled else 'OFF'}")

        elif key == ord("c"):
            if self.gesture:
                self.gesture.clear_canvas()
                print("[M5] Whiteboard canvas cleared.")

    # ------------------------------------------------------------------
    # HUD rendering — main entry point
    # ------------------------------------------------------------------

    def render_hud(self, state: dict) -> np.ndarray:
        """
        Composite all HUD elements onto state["frame"] and return it.

        Must be called after M1, M2, M3, M4 have processed the frame so
        that state contains up-to-date gesture, profile, and voice data.
        """
        frame = state.get("frame")
        if frame is None:
            return frame

        # Sync mode from state (M4 voice commands may have changed it).
        incoming_mode = state.get("mode")
        if incoming_mode and incoming_mode != self._mode:
            self.mode = incoming_mode

        # Track frame timestamp for FPS calculation.
        self._ts_buf.append(time.perf_counter())

        h, w = frame.shape[:2]

        # --- Draw each HUD layer in order ----------------------------
        self._draw_top_bar(frame, state, w)
        self._draw_bottom_bar(frame, state, w, h)

        # Whiteboard color swatch (top-right corner in WHITEBOARD mode)
        if self._mode == "WHITEBOARD" and self.gesture:
            self._draw_color_swatch(frame, w)

        return frame

    # ------------------------------------------------------------------
    # Top bar
    # ------------------------------------------------------------------

    def _draw_top_bar(self, frame: np.ndarray, state: dict, w: int) -> None:
        """
        Semi-transparent dark panel at the top of the frame.

        Layout (left → right):
          [MODE badge]  [CONTEXT badge]  [KALMAN status]    [FPS]
        """
        BAR_H = 95

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, BAR_H), config.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (0, 0), (w, BAR_H), config.COLOR_GRAY, 1)

        # ---- Mode badge (pill shape) ---------------------------------
        mode       = self._mode
        mode_color = config.MODE_COLORS.get(mode, config.COLOR_WHITE)
        badge_text = f"  {mode}  "
        (btw, bth), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx, by = 10, 8
        cv2.rectangle(frame, (bx - 4, by - 2), (bx + btw + 4, by + bth + 6),
                      mode_color, -1)
        cv2.rectangle(frame, (bx - 4, by - 2), (bx + btw + 4, by + bth + 6),
                      config.COLOR_WHITE, 1)
        cv2.putText(frame, badge_text, (bx, by + bth),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_BLACK, 2)

        # ---- Context profile badge ----------------------------------
        if self.voice:
            profile      = self.voice.profile_display
            voice_status = self.voice.voice_status
        else:
            profile      = state.get("context_profile", "DEFAULT").upper()
            voice_status = state.get("voice_status", "—")

        cx = bx + btw + 20
        cv2.putText(frame, f"CTX: {profile}",
                    (cx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_CYAN, 1)

        # ---- Kalman status ------------------------------------------
        if self.kalman:
            kal_text  = self.kalman.status_text
            kal_color = self.kalman.status_color
        else:
            kal_text  = state.get("kalman_status", "SMOOTH ?")
            kal_color = config.COLOR_GRAY

        cv2.putText(frame, kal_text,
                    (cx, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, kal_color, 1)

        # ---- Gesture from M1 / M3 -----------------------------------
        gesture = state.get("m3_gesture") or state.get("gesture", "NONE")
        cv2.putText(frame, f"Gesture: {gesture}",
                    (cx, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_YELLOW, 1)

        # ---- FPS (top-right) ----------------------------------------
        fps_text = f"FPS {self.fps:4.1f}"
        (ftw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame, fps_text,
                    (w - ftw - 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2)

        # ---- Voice status (top-right, below FPS) --------------------
        vs = voice_status[:28]   # truncate to fit
        cv2.putText(frame, f"Voice: {vs}",
                    (w - 210, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_YELLOW, 1)

        # ---- Keyboard hint (top-right) ------------------------------
        cv2.putText(frame, "TAB=mode  k=kalman  c=clear  q=quit",
                    (w - 300, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_GRAY, 1)

    # ------------------------------------------------------------------
    # Bottom bar — gesture guide
    # ------------------------------------------------------------------

    def _draw_bottom_bar(self, frame: np.ndarray, state: dict,
                         w: int, h: int) -> None:
        """
        Thin translucent bar at the bottom showing the gesture guide
        for the current mode.
        """
        BAR_H   = 30
        guide   = _GESTURE_GUIDE.get(self._mode, "")
        y_top   = h - BAR_H

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (w, h), config.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0, y_top), (w, h), config.COLOR_GRAY, 1)

        cv2.putText(frame, guide,
                    (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_WHITE, 1)

    # ------------------------------------------------------------------
    # Whiteboard color swatch
    # ------------------------------------------------------------------

    def _draw_color_swatch(self, frame: np.ndarray, w: int) -> None:
        """Show the active drawing color in the top-right corner."""
        color = self.gesture._draw_color   # direct access — same process
        cv2.rectangle(frame, (w - 45, 100), (w - 5, 140), color, -1)
        cv2.rectangle(frame, (w - 45, 100), (w - 5, 140), config.COLOR_WHITE, 1)
        cv2.putText(frame, "CLR", (w - 43, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)
