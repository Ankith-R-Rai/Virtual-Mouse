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
VELOCITY_SCROLL_DEADZONE    = 8     # px delta-y below which scroll is ignored
SCROLL_AMOUNT               = 8     # scroll-wheel clicks per event
SCROLL_COOLDOWN_INTERVAL    = 0.05  # seconds between repeated scroll events

# ---------------------------------------------------------------------------
# Pinch-to-zoom (M3) — single-hand thumb-index spread
# ---------------------------------------------------------------------------
ZOOM_THRESHOLD = 25     # px cumulative thumb-index distance change to trigger zoom
ZOOM_COOLDOWN  = 0.30   # seconds between zoom triggers

# ---------------------------------------------------------------------------
# Drag & Drop (fist gesture in MOUSE mode)
# ---------------------------------------------------------------------------
DRAG_COOLDOWN = 0.3     # seconds before drag can re-engage after a drop

# ---------------------------------------------------------------------------
# Double-click (thumb tip ↔ ring tip pinch)
# ---------------------------------------------------------------------------
DOUBLE_CLICK_THRESHOLD = 35    # px — thumb tip ↔ ring finger tip
DOUBLE_CLICK_COOLDOWN  = 0.5   # seconds

# ---------------------------------------------------------------------------
# Volume / Brightness (open-palm wrist tracking in MOUSE mode)
# ---------------------------------------------------------------------------
VOLUME_DEADZONE     = 8    # px — min wrist-X delta per frame
BRIGHTNESS_DEADZONE = 8    # px — min wrist-Y delta per frame
VOLUME_STEP         = 2    # key-presses per trigger
BRIGHTNESS_STEP     = 5    # percentage points per trigger
MEDIA_COOLDOWN      = 0.15 # seconds between repeated volume/brightness steps

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
        "keywords":           ["chrome", "firefox", "edge", "safari", "opera", "brave", "chromium"],
        "scroll_speed":       2.5,
        "cursor_sensitivity": 1.0,
        "special":            "standard",
    },
    "ide": {
        "keywords":           ["visual studio code", "vscode", "vs code", "cursor", "pycharm", "intellij", "sublime", "notepad++", "vim", "nvim", "emacs", "code", "atom", "spyder"],
        "scroll_speed":       0.5,
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
        "scroll_speed":       1.0,
        "cursor_sensitivity": 1.0,
        "special":            "standard",
    },
}
CONTEXT_POLL_INTERVAL = 0.5   # seconds between active-window checks

# ---------------------------------------------------------------------------
# Voice commands (M4)
# ---------------------------------------------------------------------------
VOICE_COMMANDS = {
    # ---- Mode switching -----------------------------------------------
    "switch to mouse":      "mode_mouse",
    "switch to whiteboard":  "mode_whiteboard",
    "switch to zoom":        "mode_zoom",
    "switch to voice":       "mode_voice",
    "switch to media":       "mode_media",
    "whiteboard on":         "whiteboard_on",
    "whiteboard off":        "whiteboard_off",

    # ---- Browser / system ---------------------------------------------
    "open browser":          "open_browser",
    "take screenshot":       "screenshot",

    # ---- Kalman filter ------------------------------------------------
    "enable smooth mode":    "toggle_kalman",
    "disable smooth mode":   "toggle_kalman",

    # ---- Scroll boost -------------------------------------------------
    "scroll up fast":        "scroll_fast_up",
    "scroll down fast":      "scroll_fast_down",

    # ---- Scroll (single-fire) -----------------------------------------
    "scroll up":             "scroll_up",
    "scroll down":           "scroll_down",
    "page up":               "page_up",
    "page down":             "page_down",

    # ---- Volume / brightness ------------------------------------------
    "volume up":             "volume_up",
    "increase volume":       "volume_up",
    "louder":                "volume_up",
    "volume down":           "volume_down",
    "decrease volume":       "volume_down",
    "lower volume":          "volume_down",
    "mute":                  "mute",
    "unmute":                "mute",
    "brightness up":         "brightness_up",
    "increase brightness":   "brightness_up",
    "brighter":              "brightness_up",
    "brightness down":       "brightness_down",
    "decrease brightness":   "brightness_down",
    "dimmer":                "brightness_down",

    # ---- Hotkeys ------------------------------------------------------
    "copy":                  "hotkey_copy",
    "paste":                 "hotkey_paste",
    "cut":                   "hotkey_cut",
    "undo":                  "hotkey_undo",
    "redo":                  "hotkey_redo",
    "select all":            "hotkey_select_all",
    "save":                  "hotkey_save",
    "new tab":               "hotkey_new_tab",
    "close tab":             "hotkey_close_tab",
    "refresh":               "hotkey_refresh",
    "reload":                "hotkey_refresh",
    "go back":               "hotkey_browser_back",
    "go forward":            "hotkey_browser_forward",
    "zoom in":               "hotkey_zoom_in",
    "zoom out":              "hotkey_zoom_out",
    "find":                  "hotkey_find",
    "print":                 "hotkey_print",
    "enter":                 "hotkey_enter",
    "delete":                "hotkey_delete",
    "new window":            "hotkey_new_window",
    "close window":          "hotkey_close_window",
    "switch window":         "hotkey_switch_window",
    "minimize":              "hotkey_minimize",
    "maximize":              "hotkey_maximize",
    "bold":                  "hotkey_bold",
    "italic":                "hotkey_italic",
    "underline":             "hotkey_underline",

    # ---- Mouse clicks -------------------------------------------------
    "click":                 "mouse_left_click",
    "left click":            "mouse_left_click",
    "right click":           "mouse_right_click",
    "double click":          "mouse_double_click",

    # ---- Canvas -------------------------------------------------------
    "clear canvas":          "canvas_clear",
    "clear whiteboard":      "canvas_clear",
    "eraser mode":           "canvas_erase_mode",
    "pen mode":              "canvas_pen_mode",
    "save canvas":           "canvas_save",
    "save drawing":          "canvas_save",
    "thicker":               "canvas_thick",
    "thinner":               "canvas_thin",
    "increase brush":        "canvas_thick",
    "decrease brush":        "canvas_thin",
}

# Hotkey map: action suffix -> tuple of keys for pyautogui.hotkey()
HOTKEY_MAP: dict[str, tuple[str,...]] = {
    "copy":            ("ctrl", "c"),
    "paste":           ("ctrl", "v"),
    "cut":             ("ctrl", "x"),
    "undo":            ("ctrl", "z"),
    "redo":            ("ctrl", "y"),
    "enter":           ("enter",),
    "select_all":      ("ctrl", "a"),
    "save":            ("ctrl", "s"),
    "new_tab":         ("ctrl", "t"),
    "close_tab":       ("ctrl", "w"),
    "refresh":         ("f5",),
    "browser_back":    ("alt", "left"),
    "browser_forward": ("alt", "right"),
    "zoom_in":         ("ctrl", "="),
    "zoom_out":        ("ctrl", "-"),
    "delete":          ("delete",),
    "find":            ("ctrl", "f"),
    "print":           ("ctrl", "p"),
    "new_window":      ("ctrl", "n"),
    "close_window":    ("alt", "f4"),
    "switch_window":   ("alt", "tab"),
    "minimize":        ("win", "down"),
    "maximize":        ("win", "up"),
    "bold":            ("ctrl", "b"),
    "italic":          ("ctrl", "i"),
    "underline":       ("ctrl", "u"),
}

# App aliases for "open <app>" voice command (fuzzy matched)
APP_ALIASES: dict[str, str] = {
    "chrome":       "chrome",
    "firefox":      "firefox",
    "edge":         "msedge",
    "notepad":      "notepad",
    "vscode":       "code",
    "vs code":      "code",
    "code":         "code",
    "pycharm":      "pycharm",
    "cursor":       "cursor",
    "terminal":     "cmd",
    "calculator":   "calc",
    "explorer":     "explorer",
    "word":         "winword",
    "excel":        "excel",
    "powerpoint":   "powerpnt",
    "spotify":      "spotify",
    "discord":      "discord",
    "slack":        "slack",
    "vlc":          "vlc",
    "paint":        "mspaint",
    "obs":          "obs",
}

# Colors for "change color to <X>" canvas voice command (BGR)
VOICE_COLORS: dict[str, tuple[int,int,int]] = {
    "red":    (0,  0,  220), "green":  (0, 200,  0), "blue":  (220,  0,  0),
    "yellow": (0, 230, 230), "white":  (255,255,255), "black": (0,    0,  0),
    "orange": (0, 165, 255), "purple": (160,  0, 160), "pink": (200,150,255),
    "cyan":   (230,230,  0),
}

# Shapes for "draw a <X>" canvas voice command
VOICE_SHAPES = ["circle", "rectangle", "triangle", "line", "star"]

VOICE_SCROLL_BOOST_DURATION = 3.0   # seconds that "scroll fast" boost lasts
VOICE_SCROLL_BOOST_FACTOR   = 2.0
VOICE_FUZZY_THRESHOLD       = 72    # fuzzy match score threshold

# ---------------------------------------------------------------------------
# Application modes
# ---------------------------------------------------------------------------
MODES = ["MOUSE", "WHITEBOARD", "ZOOM", "VOICE", "MEDIA"]

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
    "MEDIA":      COLOR_BLUE,
}
