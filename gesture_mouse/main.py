"""
main.py — Gesture and Voice Controlled Virtual Mouse
Entry point — wires all five modules into the production loop.

Pipeline each frame
-------------------
  Webcam frame
       ↓
  [M1] CoreGestureEngine.process_frame()
       • MediaPipe hand detection
       • Index-tip → screen coordinates
       • EMA smoothing
       ↓ (coord_preprocessor hook intercepts raw coords)
  [M2] KalmanFilter2D.update()
       • Predict + Update Kalman cycle
       • Returns smoothed (x, y) → back to M1 action dispatcher
       ↓
  [M3] GestureModule.process()
       • Velocity-proportional scroll   (MOUSE mode)
       • Pinch-to-zoom                  (ZOOM mode)
       • Air whiteboard draw/erase/color (WHITEBOARD mode)
       • Blends canvas onto frame
       ↓
  [M4] VoiceContextModule.process_commands()
       • Drains voice command queue (daemon thread fills it)
       • Polls active window → updates context profile
       • Executes pending actions (mode switch, kalman toggle, …)
       ↓
  [M5] UIModule.render_hud()
       • Mode badge, context profile, Kalman status
       • FPS counter, gesture label, voice status
       • Bottom gesture guide
       ↓
  cv2.imshow()  →  key handler  →  loop

Controls
--------
  TAB     cycle modes: MOUSE → WHITEBOARD → ZOOM → VOICE
  k       toggle Kalman filter
  c       clear whiteboard canvas
  q       quit
  Move mouse to top-left corner → emergency stop (PyAutoGUI failsafe)

Run
---
  cd gesture_mouse
  python main.py
"""

import sys
import os
import time

import cv2

# Ensure the project root is on the path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.core_engine    import CoreGestureEngine
from modules.kalman_filter  import KalmanFilterModule
from modules.gestures       import GestureModule
from modules.voice_context  import VoiceContextModule
from modules.ui             import UIModule


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║       Gesture and Voice Controlled Virtual Mouse             ║
║       Course: 24CSE48  |  New Horizon College of Engineering ║
╠══════════════════════════════════════════════════════════════╣
║  Controls                                                    ║
║    TAB  — cycle mode (MOUSE / WHITEBOARD / ZOOM / VOICE)     ║
║    k    — toggle Kalman tremor filter                        ║
║    c    — clear whiteboard canvas                            ║
║    q    — quit                                               ║
║  Emergency stop: move mouse to top-left corner               ║
╚══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _open_camera() -> cv2.VideoCapture:
    """Open the primary webcam with configured resolution and frame rate."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {config.CAMERA_INDEX}.\n"
            "  • Ensure a webcam is connected.\n"
            "  • On Windows: Settings → Privacy → Camera → allow access.\n"
            "  • Try changing CAMERA_INDEX in config.py."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)
    return cap


def _build_modules():
    """
    Construct all five modules and wire their cross-references.

    Returns (engine, m2, m3, m4, m5) in build order.
    M2 is wired into M1 via engine.coord_preprocessor at construction time.
    M3, M4 receive references to each other for sensitivity updates.
    M5 receives M2/M3/M4 for HUD status queries.
    """
    # M1 — Core Gesture Engine
    engine = CoreGestureEngine()

    # M2 — Kalman Filter (registers itself as engine.coord_preprocessor)
    m2 = KalmanFilterModule(engine)

    # M3 — Novel Gestures
    m3 = GestureModule(
        frame_shape=(config.FRAME_HEIGHT, config.FRAME_WIDTH)
    )

    # M4 — Voice + Context (receives M2 and M3 references)
    m4 = VoiceContextModule(engine, gesture_module=m3, kalman_module=m2)

    # M5 — UI + Integration (receives M2, M3, M4 for HUD status queries)
    m5 = UIModule(kalman_module=m2, gesture_module=m3, voice_module=m4)

    return engine, m2, m3, m4, m5


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    print(BANNER)

    cap = _open_camera()
    engine, m2, m3, m4, m5 = _build_modules()

    # Start voice listener daemon thread (M4).
    m4.start_voice()

    print(f"[main] Screen: {config.SCREEN_WIDTH} × {config.SCREEN_HEIGHT}")
    print(f"[main] Camera: {config.FRAME_WIDTH}×{config.FRAME_HEIGHT} @ {config.TARGET_FPS} FPS target")
    print("[main] Entering main loop — press 'q' in the window to quit.\n")

    read_failures = 0

    try:
        while True:
            # ----------------------------------------------------------
            # 1. Capture frame
            # ----------------------------------------------------------
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures > 10:
                    print("[main] ERROR: Camera stopped sending frames. Exiting.")
                    break
                continue
            read_failures = 0

            # Mirror horizontally — natural interaction
            frame = cv2.flip(frame, 1)

            # ----------------------------------------------------------
            # 2. Build shared state dict
            #    Modules read and write fields on this dict each frame.
            # ----------------------------------------------------------
            state: dict = {"mode": m5.mode}

            # ----------------------------------------------------------
            # 3. M1 + M2 — hand detection and Kalman-filtered cursor
            #    (M2 is applied transparently via coord_preprocessor)
            # ----------------------------------------------------------
            state = engine.process_frame(frame)
            state["mode"] = m5.mode   # ensure mode is present after M1 resets it

            # ----------------------------------------------------------
            # 4. M3 — novel gesture features
            # ----------------------------------------------------------
            state = m3.process(state)

            # ----------------------------------------------------------
            # 5. M4 — drain voice queue + apply context profile
            # ----------------------------------------------------------
            state = m4.process_commands(state)

            # Sync mode back to M5 in case M4 changed it via voice command.
            m5.mode = state.get("mode", m5.mode)

            # ----------------------------------------------------------
            # 6. M5 — render HUD overlay onto frame
            # ----------------------------------------------------------
            annotated = m5.render_hud(state)

            # ----------------------------------------------------------
            # 7. Display
            # ----------------------------------------------------------
            cv2.imshow("Gesture & Voice Virtual Mouse", annotated)

            # ----------------------------------------------------------
            # 8. Keyboard input
            # ----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[main] Quit key pressed.")
                break
            m5.handle_key(key, state)

            # Propagate any mode change from keyboard back to M3 / M4.
            m3.mode = m5.mode
            m4.mode = m5.mode

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user (Ctrl-C).")
    finally:
        m4.stop_voice()
        engine.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[main] All resources released. Goodbye.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
