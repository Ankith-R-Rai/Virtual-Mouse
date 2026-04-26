"""
test_m4.py — Test runner for M4: Voice Control + Context-Aware Profiles

Run from the gesture_mouse/ directory:
    python test_m4.py

Tests
-----
1. Unit test  — verify profile detection logic and command queue (no mic/camera).
2. Live test  — webcam + voice + context HUD. Speak commands listed below.
                Context profile changes automatically as you switch windows.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

import config
from modules.core_engine    import CoreGestureEngine
from modules.kalman_filter  import KalmanFilterModule
from modules.gestures       import GestureModule
from modules.voice_context  import VoiceContextModule, _detect_profile


# ---------------------------------------------------------------------------
# Unit test — no camera / microphone required
# ---------------------------------------------------------------------------

def unit_test() -> None:
    print("--- M4 Unit test ---")

    # 1. Profile detection from window titles
    cases = [
        ("google chrome — new tab",           "browser"),
        ("visual studio code",                 "ide"),
        ("vlc media player",                   "media"),
        ("microsoft powerpoint — slide 1",     "present"),
        ("windows explorer",                   "default"),
        ("",                                   "default"),
    ]
    for title, expected in cases:
        got = _detect_profile(title)
        assert got == expected, f"_detect_profile('{title}') = '{got}', want '{expected}'"
    print("  _detect_profile (6 cases) : OK")

    # 2. Command queue — inject actions directly and drain them
    engine = CoreGestureEngine()
    m4     = VoiceContextModule(engine)

    m4._cmd_queue.put("zoom_in")
    m4._cmd_queue.put("zoom_out")
    assert m4._cmd_queue.qsize() == 2, "Queue should have 2 items"

    state = {"mode": "MOUSE"}
    m4.process_commands(state)   # drains the queue (executes zoom_in / zoom_out)
    assert m4._cmd_queue.empty(), "Queue should be empty after process_commands"
    print("  command queue drain       : OK")

    # 3. Scroll boost flag
    m4._cmd_queue.put("scroll_fast_up")
    state = {"mode": "MOUSE"}
    m4.process_commands(state)
    assert m4._scroll_boost_until > time.time(), "Scroll boost should be active"
    print("  scroll boost activation   : OK")

    # 4. Mode switch via voice
    m4._cmd_queue.put("mode_whiteboard")
    state = {"mode": "MOUSE"}
    m4.process_commands(state)
    assert state.get("mode") == "WHITEBOARD", "mode_whiteboard should set state['mode']"
    print("  mode_whiteboard command   : OK")

    engine.release()
    print("--- M4 Unit test PASSED ---\n")


# ---------------------------------------------------------------------------
# Live test
# ---------------------------------------------------------------------------

VOICE_COMMANDS_HELP = """
  Voice commands to try
  ---------------------
  "open browser"        → Ctrl+T (new tab)
  "take screenshot"     → saves PNG to gesture_mouse/
  "enable smooth mode"  → toggles Kalman filter
  "whiteboard on"       → switches to WHITEBOARD mode
  "whiteboard off"      → returns to MOUSE mode
  "zoom in"             → Ctrl++
  "zoom out"            → Ctrl+-
  "scroll up fast"      → 2× scroll speed for 3 s
  "switch to zoom"      → ZOOM mode

  Context profiles change automatically — switch windows to test.
"""


def live_test() -> None:
    print("--- M4 Live test ---")
    print(VOICE_COMMANDS_HELP)
    print("  q = quit\n")

    engine = CoreGestureEngine()
    m2     = KalmanFilterModule(engine)
    m3     = GestureModule(frame_shape=(config.FRAME_HEIGHT, config.FRAME_WIDTH))
    m4     = VoiceContextModule(engine, gesture_module=m3, kalman_module=m2)

    m4.start_voice()   # launch background listener thread

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"[M4] Cannot open camera index {config.CAMERA_INDEX}."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # M1 + M2
            state = engine.process_frame(frame)

            # M3
            state = m3.process(state)

            # M4 — drain voice queue + poll context profile
            state = m4.process_commands(state)

            frame = state["frame"]
            h, w  = frame.shape[:2]

            # ---- HUD panel ------------------------------------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 135), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.rectangle(frame, (0, 0), (w, 135), config.COLOR_GRAY, 1)

            curr_time = time.time()
            fps       = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time

            mode    = m4.mode
            profile = m4.profile_display
            voice   = m4.voice_status
            color   = config.MODE_COLORS.get(mode, config.COLOR_WHITE)

            cv2.putText(frame, f"[M4]  Mode: {mode}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"      Context: {profile}",
                        (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_CYAN, 2)
            cv2.putText(frame, f"      Kalman : {m2.status_text}",
                        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, m2.status_color, 1)
            cv2.putText(frame, f"      Voice  : {voice[:50]}",
                        (10, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_YELLOW, 1)
            cv2.putText(frame, f"FPS: {fps:.1f}   q=quit",
                        (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_WHITE, 1)

            cv2.imshow("Virtual Mouse — M4 Voice + Context", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        print("[M4] Interrupted.")
    finally:
        m4.stop_voice()
        engine.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[M4] Resources released.")


if __name__ == "__main__":
    unit_test()
    live_test()
