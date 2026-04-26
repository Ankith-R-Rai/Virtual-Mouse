"""
modules/kalman_filter.py — M2: Kalman Filter for Tremor Smoothing

Academic context
----------------
Parkinson's disease causes resting tremors at 4–6 Hz and action tremors
at 5–12 Hz.  A standard cursor system amplifies these directly.

This module implements a discrete-time Kalman filter that:
  • Models the cursor as a point mass with position and velocity (constant-
    velocity motion model).
  • In the PREDICT step, extrapolates position forward using velocity.
  • In the UPDATE step, blends the prediction with the noisy observation using
    the Kalman gain — a noise-optimal weighting derived from two matrices:
      Q (process noise): how uncertain the motion model is.
      R (measurement noise): how noisy the sensor (camera/MediaPipe) is.

Tuning
------
  KALMAN_PROCESS_NOISE   = 0.01 (low  Q) → strong smoothing (Parkinson's mode)
  KALMAN_PROCESS_NOISE   = 1.0  (high Q) → minimal smoothing (normal mode)

Implementation is ~35 lines of pure NumPy; no external Kalman library is
required.  This is intentional — it shows understanding of the algorithm
and keeps the dependency footprint small.

Architecture
------------
KalmanFilter2D      — the filter itself (pure math, no engine coupling)
KalmanFilterModule  — thin wrapper that registers the filter as M1's
                      coord_preprocessor callback so filtering is in-frame.
"""

import sys
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# KalmanFilter2D — ~35 lines of pure NumPy
# ---------------------------------------------------------------------------

class KalmanFilter2D:
    """
    Discrete-time Kalman filter for a 2-D cursor moving with constant velocity.

    State vector  x = [px, py, vx, vy]   (4 × 1 column vector)
    Observation   z = [px, py]            (2 × 1 — we can only see position)

    Matrices
    --------
    F : state transition  (4×4)  — propagates [x, y, vx, vy] by dt
    H : observation       (2×4)  — selects [x, y] from the state
    Q : process noise     (4×4)  — uncertainty in the motion model
    R : measurement noise (2×2)  — uncertainty in the camera/MediaPipe reading

    Predict step:
        x_pred = F @ x
        P_pred = F @ P @ Fᵀ + Q

    Update step:
        S = H @ P_pred @ Hᵀ + R          (innovation covariance, 2×2)
        K = P_pred @ Hᵀ @ S⁻¹            (Kalman gain, 4×2)
        x = x_pred + K @ (z − H @ x_pred)
        P = (I − K @ H) @ P_pred
    """

    def __init__(self, process_noise: float = config.KALMAN_PROCESS_NOISE,
                 measurement_noise: float = config.KALMAN_MEASUREMENT_NOISE) -> None:

        # 4-D state column vector [px, py, vx, vy]
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._initialized = False

        # State covariance — start with large uncertainty so the filter
        # converges quickly to the first real observation.
        self._P = np.eye(4, dtype=np.float64) * 1000.0

        # Observation matrix: z = H @ x  (only position is observed)
        self._H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]], dtype=np.float64)

        # Process noise Q — scaled identity; low value = trust the motion
        # model more = strong smoothing.
        self._Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise R — scaled identity; high value = trust the
        # sensor less = stronger prediction dominance.
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

        self._I4 = np.eye(4, dtype=np.float64)   # cached identity
        self._last_t: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, raw_x: float, raw_y: float) -> tuple[int, int]:
        """
        Run one predict-update cycle.

        Args:
            raw_x, raw_y : observed screen-space coordinates (pixels)

        Returns:
            (filtered_x, filtered_y) as integers, clamped to screen bounds.
        """
        now = time.perf_counter()

        # First call — initialise state and skip update.
        if not self._initialized:
            self._state = np.array([[raw_x], [raw_y], [0.0], [0.0]],
                                   dtype=np.float64)
            self._initialized = True
            self._last_t = now
            return int(raw_x), int(raw_y)

        # dt: actual elapsed time since last frame (clamped to sensible range).
        dt = float(np.clip(now - self._last_t, 0.001, 0.1))
        self._last_t = now

        # ---- PREDICT ------------------------------------------------
        F = self._make_F(dt)
        x_pred = F @ self._state                        # (4, 1)
        P_pred = F @ self._P @ F.T + self._Q           # (4, 4)

        # ---- UPDATE -------------------------------------------------
        z = np.array([[raw_x], [raw_y]], dtype=np.float64)   # (2, 1)

        S = self._H @ P_pred @ self._H.T + self._R     # (2, 2) innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)      # (4, 2) Kalman gain

        innovation = z - self._H @ x_pred              # (2, 1)
        self._state = x_pred + K @ innovation          # (4, 1)
        self._P = (self._I4 - K @ self._H) @ P_pred    # (4, 4)

        # Extract filtered position and clamp to screen bounds.
        fx = int(np.clip(self._state[0, 0], 0, config.SCREEN_WIDTH  - 1))
        fy = int(np.clip(self._state[1, 0], 0, config.SCREEN_HEIGHT - 1))
        return fx, fy

    def reset(self) -> None:
        """Reset filter state (call when the hand re-enters the frame)."""
        self._state[:] = 0.0
        self._P = np.eye(4, dtype=np.float64) * 1000.0
        self._initialized = False
        self._last_t = None

    def set_process_noise(self, q: float) -> None:
        """Live-tune process noise. Low q → strong smoothing."""
        self._Q = np.eye(4, dtype=np.float64) * q

    def set_measurement_noise(self, r: float) -> None:
        """Live-tune measurement noise. High r → trust prediction more."""
        self._R = np.eye(2, dtype=np.float64) * r

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_F(dt: float) -> np.ndarray:
        """
        Constant-velocity state transition matrix for time step dt.

        [ 1  0  dt  0 ]   px' = px + vx·dt
        [ 0  1   0 dt ]   py' = py + vy·dt
        [ 0  0   1  0 ]   vx' = vx
        [ 0  0   0  1 ]   vy' = vy
        """
        return np.array([[1, 0, dt,  0],
                          [0, 1,  0, dt],
                          [0, 0,  1,  0],
                          [0, 0,  0,  1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# KalmanFilterModule — M2 integration wrapper
# ---------------------------------------------------------------------------

class KalmanFilterModule:
    """
    M2 wrapper that plugs KalmanFilter2D into the CoreGestureEngine pipeline.

    Usage
    -----
        engine = CoreGestureEngine()
        m2     = KalmanFilterModule(engine)
        # Now engine.coord_preprocessor is set; process_frame() will
        # automatically route raw coords through the Kalman filter.

    Toggle at runtime via:
        m2.toggle()                       # keyboard shortcut Ctrl+K
        m2.enabled = False                # direct assignment
    """

    def __init__(self, engine) -> None:
        """
        Args:
            engine : CoreGestureEngine instance (M1).
                     Its coord_preprocessor slot is set here.
        """
        self.engine  = engine
        self.filter  = KalmanFilter2D(
            process_noise=config.KALMAN_PROCESS_NOISE,
            measurement_noise=config.KALMAN_MEASUREMENT_NOISE,
        )
        self.enabled: bool = config.KALMAN_ENABLED_DEFAULT

        # Register as M1's in-frame coordinate preprocessor.
        engine.coord_preprocessor = self.filter_coords

    # ------------------------------------------------------------------
    # Preprocessor callback (called by CoreGestureEngine every frame)
    # ------------------------------------------------------------------

    def filter_coords(self, raw_x: int, raw_y: int) -> tuple[int, int]:
        """
        Entry point called by engine.coord_preprocessor each frame.

        If disabled, passes raw coordinates through unchanged so M1's
        built-in EMA smoothing still applies.
        """
        if not self.enabled:
            return raw_x, raw_y
        return self.filter.update(raw_x, raw_y)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def toggle(self) -> bool:
        """
        Toggle the Kalman filter on/off.

        Resets filter state on disable so stale velocity estimates do not
        corrupt the first frames after re-enabling.

        Returns:
            New enabled state (True = on).
        """
        self.enabled = not self.enabled
        if not self.enabled:
            self.filter.reset()
        return self.enabled

    def set_tremor_mode(self) -> None:
        """Strong smoothing preset — suitable for Parkinson's / high-tremor."""
        self.filter.set_process_noise(0.01)
        self.filter.set_measurement_noise(0.5)
        self.enabled = True

    def set_normal_mode(self) -> None:
        """Minimal smoothing preset — for users without tremors."""
        self.filter.set_process_noise(1.0)
        self.filter.set_measurement_noise(0.1)
        self.enabled = True

    # ------------------------------------------------------------------
    # HUD properties (consumed by M5)
    # ------------------------------------------------------------------

    @property
    def status_text(self) -> str:
        return "SMOOTH ON" if self.enabled else "SMOOTH OFF"

    @property
    def status_color(self) -> tuple:
        return config.COLOR_GREEN if self.enabled else config.COLOR_RED
