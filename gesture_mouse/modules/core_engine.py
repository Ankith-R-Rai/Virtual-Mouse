"""
modules/core_engine.py — M1: Core Gesture Engine

Responsibilities
----------------
1. Open the webcam and capture frames at ~30 FPS.
2. Mirror each frame (horizontal flip) for a natural mirror experience.
3. Run MediaPipe Hands to extract 21 3-D landmarks per detected hand.
4. Map index-finger-tip coordinates to screen space with EMA smoothing.
5. Detect left-click  (thumb ↔ index  pinch < LEFT_CLICK_THRESHOLD px).
6. Detect right-click (index ↔ middle pinch < RIGHT_CLICK_THRESHOLD px).
7. Detect basic scroll (index + middle up, wrist moves vertically).
8. Expose process_frame() so M2–M5 can consume state without running the loop.
9. Expose set_kalman_coords() so M2 can inject Kalman-filtered positions.

Build order: M1 is the base — build and test FULLY before adding M2.
"""

import sys
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---------------------------------------------------------------------------
# Path setup — works whether run as __main__ or imported by main.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from utils.math_utils import (
    euclidean_distance,
    map_to_screen,
    landmark_to_pixel,
    smooth_value,
    clamp,
)
from utils.gesture_utils import is_pinky_only_up


class CoreGestureEngine:
    """
    M1 — Core Gesture Engine.

    Public interface
    ----------------
    process_frame(frame)          -> dict   Process one BGR webcam frame.
    set_kalman_coords(x, y)                 M2 injects Kalman-filtered cursor.
    draw_debug_info(frame, state) -> frame  Lightweight debug HUD (M5 replaces).
    run_standalone()                        Full blocking loop for M1 testing.
    release()                               Close MediaPipe resources.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # ---- PyAutoGUI safety ----------------------------------------
        # FAILSAFE=True means moving the real mouse to the top-left corner
        # raises FailSafeException, giving the user an emergency abort.
        pyautogui.FAILSAFE = True
        # Remove the default 0.1-second inter-call pause so cursor movement
        # is smooth at 30 FPS.
        pyautogui.PAUSE = 0.0

        # ---- Screen dimensions ---------------------------------------
        self.screen_w, self.screen_h = pyautogui.size()

        # ---- MediaPipe setup -----------------------------------------
        self._mp_hands      = mp.solutions.hands
        self._mp_draw       = mp.solutions.drawing_utils
        self._mp_draw_style = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
        )

        # ---- Cursor state --------------------------------------------
        # Start cursor at screen centre so first movement is not jarring.
        self.cursor_x: int = self.screen_w  // 2
        self.cursor_y: int = self.screen_h // 2

        # ---- Click cooldown ------------------------------------------
        self._last_left_click:  float = 0.0
        self._last_right_click: float = 0.0

        # ---- Scroll state (M1 basic) ---------------------------------
        self._prev_wrist_y: int | None = None

        # Rolling wrist-y buffer — consumed by M3 velocity scroll.
        self.wrist_y_buffer: deque = deque(maxlen=config.VELOCITY_SCROLL_BUFFER_SIZE)

        # ---- Coordinate preprocessor hook (assigned by M2) ----------
        # M2 sets:  engine.coord_preprocessor = kalman_module.filter_coords
        # Signature: (raw_x: int, raw_y: int) -> (filtered_x: int, filtered_y: int)
        # When None, raw screen coordinates are used directly.
        self.coord_preprocessor = None

        # ---- State for HUD -------------------------------------------
        self.current_gesture: str = "NONE"
        self.hand_count:      int = 0

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, mode: str = "MOUSE") -> dict:
        """
        Process one BGR webcam frame (already flipped by caller).

        Returns a state dictionary consumed by M3, M4, M5:

        {
            "landmarks"    : list[list[NormalizedLandmark]]  # one list per hand
            "handedness"   : list[str]                       # "Left" / "Right"
            "hand_count"   : int
            "cursor_pos"   : (int, int)                      # screen pixels
            "raw_cursor"   : (int, int) | None               # before smoothing (for M2)
            "gesture"      : str                             # "NONE" / "LEFT_CLICK" / ...
            "wrist_y_buffer": deque                          # for M3 velocity scroll
            "frame"        : np.ndarray                      # annotated frame
            "frame_w"      : int
            "frame_h"      : int
        }
        """
        h, w = frame.shape[:2]

        # Build the base state returned even when no hand is detected.
        state: dict = {
            "landmarks":      [],
            "handedness":     [],
            "hand_count":     0,
            "cursor_pos":     (self.cursor_x, self.cursor_y),
            "raw_cursor":     None,
            "gesture":        "NONE",
            "wrist_y_buffer": self.wrist_y_buffer,
            "frame":          frame,
            "frame_w":        w,
            "frame_h":        h,
        }

        # ---- BGR → RGB (MediaPipe expects RGB) -----------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # avoid unnecessary copy in MediaPipe
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            # No hand in frame — reset transient state.
            self.current_gesture = "NONE"
            self._prev_wrist_y   = None
            self.hand_count      = 0
            return state

        # ---- Draw skeleton on frame ----------------------------------
        for hand_lm in results.multi_hand_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                hand_lm,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_draw_style.get_default_hand_landmarks_style(),
                self._mp_draw_style.get_default_hand_connections_style(),
            )

        # ---- Populate state with landmark data -----------------------
        self.hand_count       = len(results.multi_hand_landmarks)
        state["hand_count"]   = self.hand_count
        state["landmarks"]    = [hl.landmark for hl in results.multi_hand_landmarks]
        if results.multi_handedness:
            state["handedness"] = [
                h.classification[0].label for h in results.multi_handedness
            ]

        # Identify hands by label
        primary   = state["landmarks"][0] # fallback
        secondary = None
        
        if "handedness" in state:
            for i, label in enumerate(state["handedness"]):
                if label == "Right":
                    primary = state["landmarks"][i]
                elif label == "Left":
                    secondary = state["landmarks"][i]
        
        state["primary_landmarks"]   = primary
        state["secondary_landmarks"] = secondary

        # ---- Cursor movement -----------------------------------------
        # 1. Map index-finger-tip normalized → screen pixels.
        raw_x, raw_y = map_to_screen(
            primary[config.LM_INDEX_TIP].x,
            primary[config.LM_INDEX_TIP].y,
            self.screen_w,
            self.screen_h,
        )
        state["raw_cursor"] = (raw_x, raw_y)

        # 2. Apply coordinate preprocessor (M2 Kalman filter) if registered.
        #    M2 sets engine.coord_preprocessor = kalman_module.filter_coords
        #    so filtering happens in-frame with zero lag.
        if self.coord_preprocessor is not None:
            target_x, target_y = self.coord_preprocessor(raw_x, raw_y)
        else:
            target_x, target_y = raw_x, raw_y

        # 3. Apply EMA smoothing (M1-level; M2 already applied its own filter).
        self.cursor_x = int(smooth_value(target_x, self.cursor_x, config.CURSOR_SMOOTHING))
        self.cursor_y = int(smooth_value(target_y, self.cursor_y, config.CURSOR_SMOOTHING))

        # 4. Clamp to screen bounds.
        self.cursor_x = int(clamp(self.cursor_x, 0, self.screen_w - 1))
        self.cursor_y = int(clamp(self.cursor_y, 0, self.screen_h - 1))

        state["cursor_pos"] = (self.cursor_x, self.cursor_y)

        # 5. Move the system cursor.
        if mode == "MOUSE":
            try:
                pyautogui.moveTo(self.cursor_x, self.cursor_y)
            except pyautogui.FailSafeException:
                # User triggered emergency abort — do not crash, just skip move.
                pass

        # ---- Gesture detection ---------------------------------------
        # Scroll is now triggered by the Left hand (secondary)
        gesture = self._detect_scroll(secondary, w, h, mode)
        
        # IF the left hand is visible, we LOCK the mouse from clicking 
        # to ensure the scroller never accidentally clicks things.
        if secondary is not None:
            if gesture == "NONE":
                gesture = "SCROLL_WAIT" # dummy state to block click detection
        else:
            if gesture == "NONE":
                gesture = self._detect_click(primary, w, h, mode)

        self.current_gesture = gesture
        state["gesture"]     = gesture

        # ---- Update rolling wrist-y buffer (for M3) ------------------
        # If Left hand exists, track its wrist for velocity scrolling.
        if secondary:
            wrist_y_px = int(secondary[config.LM_WRIST].y * h)
            self.wrist_y_buffer.append(wrist_y_px)
        else:
            self.wrist_y_buffer.clear() # No scroller hand = no scroll velocity
            
        state["frame"] = frame
        return state

    # ------------------------------------------------------------------
    # Click detection
    # ------------------------------------------------------------------

    def _detect_click(self, landmarks, frame_w: int, frame_h: int, mode: str = "MOUSE") -> str:
        """
        Detect left-click and right-click via pinch distances.

        Left click  : thumb tip (LM4) ↔ index tip (LM8)  < LEFT_CLICK_THRESHOLD px
        Right click : index tip (LM8) ↔ middle tip (LM12) < RIGHT_CLICK_THRESHOLD px

        A per-gesture cooldown of CLICK_COOLDOWN seconds prevents repeated firing
        while fingers remain close together.
        """
        now = time.time()


        # Denormalize landmarks to frame pixel space for distance measurement.
        thumb_px  = landmark_to_pixel(landmarks[config.LM_THUMB_TIP],  frame_w, frame_h)
        index_px  = landmark_to_pixel(landmarks[config.LM_INDEX_TIP],  frame_w, frame_h)
        middle_px = landmark_to_pixel(landmarks[config.LM_MIDDLE_TIP], frame_w, frame_h)

        left_dist  = euclidean_distance(thumb_px,  index_px)
        # Right click: thumb tip to middle tip
        right_dist = euclidean_distance(thumb_px, middle_px)

        # Left click
        if (left_dist < config.LEFT_CLICK_THRESHOLD and
                now - self._last_left_click > config.CLICK_COOLDOWN):
            if mode == "MOUSE":
                try:
                    pyautogui.click()
                except pyautogui.FailSafeException:
                    pass
            self._last_left_click = now
            return "LEFT_CLICK"

        # Right click (only if left click did not fire — avoids conflict)
        if (right_dist < config.RIGHT_CLICK_THRESHOLD and
                now - self._last_right_click > config.CLICK_COOLDOWN):
            if mode == "MOUSE":
                try:
                    pyautogui.rightClick()
                except pyautogui.FailSafeException:
                    pass
            self._last_right_click = now
            return "RIGHT_CLICK"

        return "NONE"

    # ------------------------------------------------------------------
    # Basic scroll detection (M3 velocity scroll will override this)
    # ------------------------------------------------------------------

    def _detect_scroll(self, landmarks, frame_w: int, frame_h: int, mode: str = "MOUSE") -> str:
        """
        Left hand presence triggers scroll mode.
        """
        if landmarks is None:
            self._prev_wrist_y = None
            return "NONE"

        wrist_y_px = int(landmarks[config.LM_WRIST].y * frame_h)

        if self._prev_wrist_y is None:
            self._prev_wrist_y = wrist_y_px
            return "NONE"

        dy = wrist_y_px - self._prev_wrist_y
        self._prev_wrist_y = wrist_y_px

        if abs(dy) < config.SCROLL_DEADZONE:
            return "NONE"

        # dy > 0 means wrist moved down  → scroll down (negative in pyautogui)
        # dy < 0 means wrist moved up    → scroll up   (positive in pyautogui)
        scroll_amount = -config.SCROLL_FIXED if dy > 0 else config.SCROLL_FIXED

        if mode == "MOUSE":
            try:
                pyautogui.scroll(scroll_amount)
            except pyautogui.FailSafeException:
                pass

        return "SCROLL_UP" if scroll_amount > 0 else "SCROLL_DOWN"

    # ------------------------------------------------------------------
    # Debug / minimal HUD  (M5 renders the full HUD over this)
    # ------------------------------------------------------------------

    def draw_debug_info(self, frame: np.ndarray, state: dict) -> np.ndarray:
        """
        Render a lightweight status panel on the frame for M1 standalone testing.

        M5 replaces this with the full HUD overlay; this method is only used
        when running test_m1.py directly.
        """
        gesture    = state.get("gesture",    "NONE")
        hand_count = state.get("hand_count", 0)
        cursor_pos = state.get("cursor_pos", (0, 0))

        # Semi-transparent dark background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 90), config.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.rectangle(frame, (0, 0), (340, 90), config.COLOR_GRAY, 1)

        cv2.putText(frame, f"[M1]  Hands: {hand_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN,  2)
        cv2.putText(frame, f"      Gesture: {gesture}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 2)
        cv2.putText(frame, f"      Cursor : {cursor_pos[0]}, {cursor_pos[1]}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE,  1)

        return frame

    # ------------------------------------------------------------------
    # Standalone run loop (M1 testing only)
    # ------------------------------------------------------------------

    def run_standalone(self) -> None:
        """
        Full blocking capture-and-display loop.

        Controls
        --------
        q  — quit
        Any key — continue
        """
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not cap.isOpened():
            raise RuntimeError(
                f"[M1] Cannot open camera index {config.CAMERA_INDEX}.\n"
                "  • Check that a webcam is connected.\n"
                "  • On Windows, ensure camera privacy permissions are enabled.\n"
                "  • Try a different CAMERA_INDEX in config.py."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

        print("[M1] Core Gesture Engine — standalone mode")
        print(f"[M1] Screen: {self.screen_w} × {self.screen_h}")
        print("[M1] Controls: 'q' to quit  |  Move mouse to top-left corner = emergency stop")

        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Transient read failure — log and retry.
                    print("[M1] Warning: frame read failed; retrying …")
                    continue

                # Mirror horizontally for natural interaction.
                frame = cv2.flip(frame, 1)

                # Process gestures.
                state = self.process_frame(frame)

                # Lightweight debug overlay.
                annotated = self.draw_debug_info(state["frame"], state)

                # FPS counter (top-right corner).
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-9)
                prev_time = curr_time
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (annotated.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GREEN, 2,
                )

                cv2.imshow("Virtual Mouse — M1 Core Engine", annotated)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        except KeyboardInterrupt:
            print("[M1] Interrupted by user (Ctrl-C).")
        finally:
            self.release()
            cap.release()
            cv2.destroyAllWindows()
            print("[M1] Resources released cleanly.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release MediaPipe graph resources."""
        self.hands.close()
