"""
test_m3.py — Test runner for M3: Novel Gesture Features

Run from the gesture_mouse/ directory:
    python test_m3.py

Tests
-----
1. Unit test  — verify canvas operations and scroll math (no camera).
2. Live test  — webcam demo with Tab-key mode switching:
                MOUSE mode   → velocity scroll (raise index + middle, move wrist)
                WHITEBOARD   → draw with index finger, fist to erase
                ZOOM mode    → bring both hands into frame, pinch/expand thumbs
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import config
from modules.core_engine   import CoreGestureEngine
from modules.kalman_filter import KalmanFilterModule
from modules.gestures      import GestureModule


# ---------------------------------------------------------------------------
# Unit test — no camera required
# ---------------------------------------------------------------------------

def unit_test() -> None:
    print("--- M3 Unit test ---")

    m3 = GestureModule(frame_shape=(480, 640))

    # 1. Scroll sensitivity setter
    m3.set_scroll_sensitivity(1.5)
    assert m3._scroll_sensitivity == 1.5, "set_scroll_sensitivity failed"
    print("  set_scroll_sensitivity : OK")

    # 2. Canvas starts empty
    assert np.all(m3._canvas == 0), "Canvas should start empty"
    print("  canvas initialised     : OK")

    # 3. clear_canvas wipes non-zero pixels
    m3._canvas[100, 100] = (255, 0, 0)
    m3.clear_canvas()
    assert np.all(m3._canvas == 0), "clear_canvas did not wipe all pixels"
    print("  clear_canvas           : OK")

    # 4. Mode switching
    for mode in config.MODES:
        m3.set_mode(mode)
        assert m3.mode == mode, f"set_mode({mode}) failed"
    print("  set_mode (all modes)   : OK")

    # 5. Canvas overlay blend — non-zero canvas pixels alter output
    frame  = np.full((480, 640, 3), 100, dtype=np.uint8)
    m3._canvas[200, 200] = (0, 255, 0)   # one green pixel
    result = m3._apply_canvas(frame)
    # The drawn pixel should differ from the raw frame pixel
    assert not np.array_equal(result[200, 200], frame[200, 200]), \
        "Canvas overlay did not blend the drawn pixel"
    print("  canvas overlay blend   : OK")

    print("--- M3 Unit test PASSED ---\n")


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------

def live_test() -> None:
    print("--- M3 Live test ---")
    print("  TAB         : cycle modes (MOUSE → WHITEBOARD → ZOOM → VOICE)")
    print("  MOUSE mode  : raise index + middle, move wrist up/down to scroll")
    print("  WHITEBOARD  : index finger draws | fist erases | thumb+ring cycles color")
    print("  ZOOM mode   : show both hands, expand/contract thumbs to zoom")
    print("  q           : quit\n")

    engine = CoreGestureEngine()
    m2     = KalmanFilterModule(engine)
    m3     = GestureModule(frame_shape=(config.FRAME_HEIGHT, config.FRAME_WIDTH))

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"[M3] Cannot open camera index {config.CAMERA_INDEX}."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    mode_idx  = 0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # M1 + M2 processing
            state = engine.process_frame(frame)
            state["mode"] = config.MODES[mode_idx]

            # M3 processing
            state = m3.process(state)
            frame = state["frame"]

            h, w = frame.shape[:2]

            # ---- HUD panel ------------------------------------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, 110), config.COLOR_GRAY, 1)

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time

            mode      = config.MODES[mode_idx]
            m3_gesture = state.get("m3_gesture", "NONE")
            color     = config.MODE_COLORS.get(mode, config.COLOR_WHITE)

            cv2.putText(frame, f"[M3]  Mode: {mode}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"      Gesture: {m3_gesture}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 2)
            cv2.putText(frame, f"      Kalman: {m2.status_text}",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m2.status_color, 1)
            cv2.putText(frame,
                        f"FPS: {fps:.1f}   TAB=cycle mode   q=quit",
                        (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1)

            # Draw color swatch in WHITEBOARD mode
            if mode == "WHITEBOARD":
                swatch_color = m3._draw_color
                cv2.rectangle(frame, (w - 40, 5), (w - 5, 40), swatch_color, -1)
                cv2.rectangle(frame, (w - 40, 5), (w - 5, 40), config.COLOR_WHITE, 1)

            cv2.imshow("Virtual Mouse — M3 Novel Gestures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("\t"):
                mode_idx = (mode_idx + 1) % len(config.MODES)
                m3.set_mode(config.MODES[mode_idx])
                print(f"[M3] Mode → {config.MODES[mode_idx]}")

    except KeyboardInterrupt:
        print("[M3] Interrupted.")
    finally:
        engine.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[M3] Resources released.")


if __name__ == "__main__":
    unit_test()
    live_test()
