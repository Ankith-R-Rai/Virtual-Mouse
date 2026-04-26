"""
test_m1.py — Standalone test runner for M1: Core Gesture Engine.

Run from the gesture_mouse/ directory:
    python test_m1.py

What this tests
---------------
1. Camera opens successfully.
2. MediaPipe initialises without errors.
3. Cursor moves with your index finger.
4. Left click  fires when you pinch thumb + index.
5. Right click fires when you pinch index + middle.
6. Scroll fires when index + middle are raised and wrist moves up/down.
7. Debug HUD renders on the webcam window.
8. 'q' key exits cleanly and releases all resources.
"""

import sys
import os

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.core_engine import CoreGestureEngine


def smoke_test() -> None:
    """Verify imports and basic object construction without opening the camera."""
    print("--- Smoke test ---")

    # 1. Config loads correctly
    assert config.SCREEN_WIDTH  > 0, "Screen width must be positive"
    assert config.SCREEN_HEIGHT > 0, "Screen height must be positive"
    print(f"  Screen resolution : {config.SCREEN_WIDTH} × {config.SCREEN_HEIGHT}")

    # 2. Engine constructs without error
    engine = CoreGestureEngine()
    assert engine.screen_w == config.SCREEN_WIDTH
    assert engine.screen_h == config.SCREEN_HEIGHT
    print(f"  CoreGestureEngine : OK  (cursor starts at {engine.cursor_x}, {engine.cursor_y})")

    # 3. Kalman override setter
    engine.set_kalman_coords(100, 200)
    assert engine._kalman_coords == (100, 200), "Kalman override not stored"
    print("  set_kalman_coords : OK")

    engine.release()
    print("  release()         : OK")
    print("--- Smoke test PASSED ---\n")


def run_live() -> None:
    """Open the webcam and run M1 interactively."""
    print("--- Live test ---")
    print("  Point your index finger at the screen to move the cursor.")
    print("  Pinch thumb + index  → LEFT CLICK")
    print("  Pinch index + middle → RIGHT CLICK")
    print("  Raise index + middle, move wrist up/down → SCROLL")
    print("  Press 'q' in the webcam window to quit.\n")

    engine = CoreGestureEngine()
    engine.run_standalone()


if __name__ == "__main__":
    smoke_test()
    run_live()
