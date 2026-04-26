"""
test_m2.py — Test runner for M2: Kalman Filter

Run from the gesture_mouse/ directory:
    python test_m2.py

Tests
-----
1. Unit test  — verify filter output is smoother than noisy input (no camera).
2. Live test  — webcam feed with side-by-side RAW vs FILTERED cursor display
                and Ctrl+K to toggle the filter on/off.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

import config
from modules.kalman_filter import KalmanFilter2D, KalmanFilterModule
from modules.core_engine   import CoreGestureEngine


# ---------------------------------------------------------------------------
# Unit test — no camera required
# ---------------------------------------------------------------------------

def unit_test() -> None:
    """
    Simulate a trembling signal (true position + sine-wave noise) and verify
    that the Kalman filter's output is closer to the true position.
    """
    print("--- Unit test ---")

    kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)

    true_x, true_y = 400.0, 300.0
    tremor_amp  = 20.0   # pixels — represents ±20 px Parkinson's-like tremor
    tremor_freq = 8.0    # Hz     — within the 5–12 Hz action-tremor range
    dt          = 1 / 30 # 30 FPS

    raw_errors:      list[float] = []
    filtered_errors: list[float] = []

    rng = np.random.default_rng(seed=42)

    for i in range(120):  # 4 seconds at 30 FPS
        t = i * dt

        # Noisy observation: true position + deterministic tremor + random noise
        noisy_x = true_x + tremor_amp * math.sin(2 * math.pi * tremor_freq * t) \
                         + rng.normal(0, 3)
        noisy_y = true_y + tremor_amp * math.cos(2 * math.pi * tremor_freq * t) \
                         + rng.normal(0, 3)

        fx, fy = kf.update(noisy_x, noisy_y)

        raw_errors.append(math.hypot(noisy_x - true_x, noisy_y - true_y))
        filtered_errors.append(math.hypot(fx - true_x, fy - true_y))

        # Simulate frame timing
        time.sleep(dt * 0.01)   # 1% of real time so the test is fast

    mean_raw      = float(np.mean(raw_errors))
    mean_filtered = float(np.mean(filtered_errors))
    improvement   = (1 - mean_filtered / mean_raw) * 100

    print(f"  Mean raw error      : {mean_raw:.2f} px")
    print(f"  Mean filtered error : {mean_filtered:.2f} px")
    print(f"  Noise reduction     : {improvement:.1f} %")

    assert mean_filtered < mean_raw, \
        "Kalman filter must reduce mean error vs raw signal"
    print("--- Unit test PASSED ---\n")


# ---------------------------------------------------------------------------
# Live test — webcam + side-by-side visual comparison
# ---------------------------------------------------------------------------

def live_test() -> None:
    """
    Webcam test showing the Kalman filter effect in real time.

    Controls
    --------
    k / Ctrl+K   toggle Kalman filter on/off
    q            quit
    t            switch to tremor-mode preset  (strong smoothing)
    n            switch to normal-mode preset  (minimal smoothing)
    """
    print("--- Live test ---")
    print("  Controls: 'k' toggle filter | 't' tremor preset | 'n' normal preset | 'q' quit")
    print("  The webcam window shows: RAW trail (red) vs FILTERED trail (green)\n")

    engine = CoreGestureEngine()
    m2     = KalmanFilterModule(engine)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"[M2] Cannot open camera index {config.CAMERA_INDEX}.\n"
            "  • Check webcam is connected and accessible.\n"
            "  • Change CAMERA_INDEX in config.py."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    # Trail buffers for visual comparison
    raw_trail:      list[tuple[int, int]] = []
    filtered_trail: list[tuple[int, int]] = []
    TRAIL_LEN = 30

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # --- Temporarily disable filter to capture raw position -------
            m2.enabled = False
            raw_state = engine.process_frame(frame.copy())
            raw_pos   = raw_state["raw_cursor"]

            # --- Re-enable filter and get filtered position ---------------
            m2.enabled = True
            filt_state  = engine.process_frame(frame)
            filt_pos    = filt_state["cursor_pos"]

            # --- Update trails --------------------------------------------
            if raw_pos:
                raw_trail.append(raw_pos)
            if filt_pos:
                filtered_trail.append(filt_pos)
            if len(raw_trail)      > TRAIL_LEN: raw_trail.pop(0)
            if len(filtered_trail) > TRAIL_LEN: filtered_trail.pop(0)

            # --- Draw trails on frame (scaled to frame size) --------------
            sw, sh = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
            def to_frame(sp):
                return (int(sp[0] * w / sw), int(sp[1] * h / sh))

            for i in range(1, len(raw_trail)):
                cv2.line(frame, to_frame(raw_trail[i-1]),
                         to_frame(raw_trail[i]), config.COLOR_RED, 2)
            for i in range(1, len(filtered_trail)):
                cv2.line(frame, to_frame(filtered_trail[i-1]),
                         to_frame(filtered_trail[i]), config.COLOR_GREEN, 2)

            # --- HUD panel -----------------------------------------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (360, 120), config.COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.rectangle(frame, (0, 0), (360, 120), config.COLOR_GRAY, 1)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time

            cv2.putText(frame, f"[M2]  {m2.status_text}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, m2.status_color, 2)
            cv2.putText(frame, "RED trail  = RAW cursor",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_RED, 1)
            cv2.putText(frame, "GREEN trail = FILTERED cursor",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_GREEN, 1)
            cv2.putText(frame, f"FPS: {fps:.1f}   |  k=toggle  t=tremor  n=normal  q=quit",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1)

            cv2.imshow("Virtual Mouse — M2 Kalman Filter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("k"):
                state = m2.toggle()
                print(f"[M2] Kalman filter {'ON' if state else 'OFF'}")
            elif key == ord("t"):
                m2.set_tremor_mode()
                print("[M2] Switched to TREMOR preset (strong smoothing)")
            elif key == ord("n"):
                m2.set_normal_mode()
                print("[M2] Switched to NORMAL preset (minimal smoothing)")

    except KeyboardInterrupt:
        print("[M2] Interrupted.")
    finally:
        engine.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[M2] Resources released.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unit_test()
    live_test()
