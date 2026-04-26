"""
config.py — Central configuration for all modules.

All tunable thresholds, Kalman params, sensitivity profiles, cooldowns, and
colors live here.  Logic code imports from this file; never hard-code constants
inside modules.
"""

import pyautogui

# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX  = 0       # Change to 1 if a second camera is preferred
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
TARGET_FPS    = 30

# ---------------------------------------------------------------------------
# MediaPipe Hands
# ---------------------------------------------------------------------------
MAX_HANDS             = 2
DETECTION_CONFIDENCE  = 0.7
TRACKING_CONFIDENCE   = 0.5

# ---------------------------------------------------------------------------
# Cursor movement
# ---------------------------------------------------------------------------
# Exponential moving-average factor applied in M1 BEFORE Kalman (M2).
# 0.0 = cursor never moves   |   1.0 = cursor jumps instantly to raw position
CURSOR_SMOOTHING = 0.8

# ---------------------------------------------------------------------------
# Click detection  (distances measured in frame pixels, e.g. 640×480 space)
# ---------------------------------------------------------------------------
LEFT_CLICK_THRESHOLD  = 30   # px  — thumb tip ↔ index tip
RIGHT_CLICK_THRESHOLD = 40   # px  — index tip ↔ middle tip
CLICK_COOLDOWN        = 0.20  # seconds between repeated triggers

# ---------------------------------------------------------------------------
# Scroll (M1 basic) — M3 overrides scroll_amount with velocity computation
# ---------------------------------------------------------------------------
SCROLL_DEADZONE   = 3     # px  — ignore micro-movements
SCROLL_FIXED      = 10    # scroll units used by the M1 fallback
SCROLL_SENSITIVITY = 0.8  # default multiplier (M4 context profiles override)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices  (0–20, standard 21-point hand model)
# ---------------------------------------------------------------------------
LM_WRIST          = 0
LM_THUMB_CMC      = 1
LM_THUMB_MCP      = 2
LM_THUMB_IP       = 3
LM_THUMB_TIP      = 4
LM_INDEX_MCP      = 5
LM_INDEX_PIP      = 6
LM_INDEX_DIP      = 7
LM_INDEX_TIP      = 8
LM_MIDDLE_MCP     = 9
LM_MIDDLE_PIP     = 10
LM_MIDDLE_DIP     = 11
LM_MIDDLE_TIP     = 12
LM_RING_MCP       = 13
LM_RING_PIP       = 14
LM_RING_DIP       = 15
LM_RING_TIP       = 16
LM_PINKY_MCP      = 17
LM_PINKY_PIP      = 18
LM_PINKY_DIP      = 19
LM_PINKY_TIP      = 20

# ---------------------------------------------------------------------------
# Kalman filter (M2)
# ---------------------------------------------------------------------------
# Low  Q → strong smoothing (Parkinson's / tremor mode)
# High Q → minimal smoothing (normal user)
KALMAN_PROCESS_NOISE     = 0.1    # Q diagonal value
KALMAN_MEASUREMENT_NOISE = 0.1    # R diagonal value
KALMAN_ENABLED_DEFAULT   = True

# ---------------------------------------------------------------------------
# Velocity-proportional scroll (M3)
# ---------------------------------------------------------------------------
VELOCITY_SCROLL_BUFFER_SIZE = 5     # frames kept in rolling wrist-y buffer
VELOCITY_SCROLL_DEADZONE    = 5     # px delta-y below which scroll is ignored

# ---------------------------------------------------------------------------
# Pinch-to-zoom (M3)
# ---------------------------------------------------------------------------
ZOOM_THRESHOLD = 20     # px change in inter-thumb distance to trigger zoom
ZOOM_COOLDOWN  = 0.10   # seconds

# ---------------------------------------------------------------------------
# Air Whiteboard (M3)
# ---------------------------------------------------------------------------
WHITEBOARD_THICKNESS   = 3
WHITEBOARD_ERASE_RADIUS = 60
WHITEBOARD_ALPHA       = 0.3   # canvas blend weight over webcam frame
WHITEBOARD_COLORS = [
    (0,   0,   255),  # red
    (0,   255,  0),   # green
    (255,  0,   0),   # blue
    (255, 255, 255),  # white
    (0,   255, 255),  # yellow
]

# ---------------------------------------------------------------------------
# Context-aware profiles (M4)
# ---------------------------------------------------------------------------
# Each profile supplies:
#   scroll_speed      — multiplier applied to scroll_amount in M3
#   cursor_sensitivity — multiplier applied to cursor delta in M1
#   special           — string tag consumed by M3/M4 for macro behaviour
CONTEXT_PROFILES: dict = {
    "browser": {
        "keywords":           ["chrome", "firefox", "edge", "opera", "brave", "safari"],
        "scroll_speed":       1.2,
        "cursor_sensitivity": 1.0,
        "special":            "standard",
    },
    "ide": {
        "keywords":           ["code", "pycharm", "sublime", "atom", "vim", "notepad++", "spyder"],
        "scroll_speed":       0.4,
        "cursor_sensitivity": 0.6,
        "special":            "precise",
    },
    "media": {
        "keywords":           ["vlc", "youtube", "spotify", "netflix", "mpv", "media player"],
        "scroll_speed":       1.0,
        "cursor_sensitivity": 1.0,
        "special":            "media_macros",
    },
    "present": {
        "keywords":           ["powerpoint", "impress", "keynote", "slides"],
        "scroll_speed":       0.0,
        "cursor_sensitivity": 1.0,
        "special":            "slide_nav",
    },
    "default": {
        "keywords":           [],
        "scroll_speed":       0.8,
        "cursor_sensitivity": 1.0,
        "special":            "standard",
    },
}
CONTEXT_POLL_INTERVAL = 0.5   # seconds between active-window checks

# ---------------------------------------------------------------------------
# Voice commands (M4)
# ---------------------------------------------------------------------------
VOICE_COMMANDS = {
    "open browser":       "open_browser",
    "take screenshot":    "screenshot",
    "enable smooth mode": "toggle_kalman",
    "disable smooth mode":"toggle_kalman",
    "whiteboard on":      "whiteboard_on",
    "whiteboard off":     "whiteboard_off",
    "scroll up fast":     "scroll_fast_up",
    "scroll down fast":   "scroll_fast_down",
    "switch to mouse":    "mode_mouse",
    "switch to whiteboard": "mode_whiteboard",
    "switch to zoom":     "mode_zoom",
    "switch to voice":    "mode_voice",
}
VOICE_SCROLL_BOOST_DURATION = 3.0   # seconds that "scroll fast" boost lasts
VOICE_SCROLL_BOOST_FACTOR   = 2.0

# ---------------------------------------------------------------------------
# Application modes
# ---------------------------------------------------------------------------
MODES = ["MOUSE", "WHITEBOARD", "ZOOM", "VOICE"]

# ---------------------------------------------------------------------------
# HUD colors (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (0,   255,   0)
COLOR_RED    = (0,     0, 255)
COLOR_BLUE   = (255,   0,   0)
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0,   255, 255)
COLOR_CYAN   = (255, 255,   0)
COLOR_BLACK  = (0,     0,   0)
COLOR_GRAY   = (100, 100, 100)

MODE_COLORS = {
    "MOUSE":      COLOR_GREEN,
    "WHITEBOARD": COLOR_CYAN,
    "ZOOM":       COLOR_YELLOW,
    "VOICE":      COLOR_RED,
}
