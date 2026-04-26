# Gesture and Voice Controlled Virtual Mouse
### Course: 24CSE48 — Mini Project
### Department of Computer Science and Engineering
### New Horizon College of Engineering, Bengaluru
### Academic Session: February 2026 – May 2026

| USN | Name | Section |
|-----|------|---------|
| 1NH24CS004 | Abhay Surya K S | A |
| 1NH24CS024 | Ankith R Rai | A |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Novelty Features](#2-novelty-features)
3. [System Architecture](#3-system-architecture)
4. [Project File Structure](#4-project-file-structure)
5. [Setup and Installation](#5-setup-and-installation)
6. [Configuration Reference (config.py)](#6-configuration-reference-configpy)
7. [Module M1 — Core Gesture Engine](#7-module-m1--core-gesture-engine)
8. [Module M2 — Kalman Filter (Tremor Smoothing)](#8-module-m2--kalman-filter-tremor-smoothing)
9. [Module M3 — Novel Gesture Features](#9-module-m3--novel-gesture-features)
10. [Module M4 — Voice Control + Context-Aware Profiles](#10-module-m4--voice-control--context-aware-profiles)
11. [Module M5 — Playground UI + System Integration](#11-module-m5--playground-ui--system-integration)
12. [Full Pipeline Walkthrough](#12-full-pipeline-walkthrough)
13. [Troubleshooting Guide](#13-troubleshooting-guide)
14. [Viva Questions and Answers](#14-viva-questions-and-answers)
15. [Build Summary](#15-build-summary)

---

## 1. Project Overview

Traditional human-computer interaction relies on physical input devices such as keyboards, mice, and touchscreens. These devices present limitations in contexts requiring hygiene, accessibility, or hands-free operation.

This project builds a **Python-based touchless human-computer interaction system** that enables users to control their computer entirely through:
- **Hand gestures** captured via a webcam (MediaPipe + OpenCV)
- **Voice commands** captured via a microphone (SpeechRecognition)

The system is built around five independent but interconnected modules, with a build order of M1 → M2 → M3 → M4 → M5.

### Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.x | Webcam capture, frame processing, HUD rendering |
| mediapipe | 0.10.x | 21-point 3-D hand landmark detection |
| pyautogui | 0.9.x | Cursor movement, click simulation, scroll, hotkeys |
| numpy | 1.x | Coordinate interpolation, Kalman filter math |
| SpeechRecognition | 3.10.x | Online/offline voice recognition |
| pyaudio | 0.2.x | Microphone input backend |
| pygetwindow | 0.0.9 | Active window title detection (context profiles) |

---

## 2. Novelty Features

The following six features distinguish this project from existing literature:

### NOVEL FEATURE 1 — Kalman Filter for Tremor Smoothing
A two-dimensional Kalman filter is applied to raw MediaPipe hand landmark coordinates before they reach the cursor control system. This filters high-frequency involuntary tremors (5–12 Hz range typical in Parkinson's disease) while preserving intentional slow movements. This makes the system usable by individuals with motor impairments — a feature absent in all surveyed literature.

### NOVEL FEATURE 2 — Context-Aware Window Profiles
The system polls the active window title every 500 ms and dynamically adjusts gesture sensitivity profiles. A browser window receives fast scroll speed and normal cursor sensitivity; a code editor receives slow precise scrolling; a media player activates gesture-based macros. The same physical gesture produces contextually appropriate behaviour depending on the active application.

### NOVEL FEATURE 3 — Velocity-Proportional Scrolling
Scroll speed is proportional to swipe velocity rather than fixed. The system tracks wrist y-position across a rolling 5-frame buffer, computes delta-y per frame, and maps it linearly to the scroll amount. Faster swipes produce faster scrolling, enabling both quick navigation and fine-grained positioning.

### NOVEL FEATURE 4 — Pinch-to-Zoom
By detecting both hands simultaneously (MediaPipe `max_num_hands=2`), the system measures the Euclidean distance between left and right thumb tips across consecutive frames. Expanding the pinch triggers Ctrl+Plus; contracting triggers Ctrl+Minus — enabling natural zoom gestures for browsers, documents, and image viewers.

### NOVEL FEATURE 5 — Air Whiteboard Mode
A transparent OpenCV canvas overlay enables freehand drawing in the air. When only the index finger is raised, the fingertip position is tracked and lines are drawn between consecutive frames. A closed fist gesture erases a circular region. Thumb+ring pinch cycles through a colour palette. This enables annotation, sketching, and presentation markup without physical contact.

### NOVEL FEATURE 6 — Playground UI Showcase
A live HUD overlay rendered on the webcam feed displays the current active mode, detected context profile, Kalman filter status, and frame rate. A mode-switcher panel allows seamless transitions between Mouse, Whiteboard, Zoom, and Voice modes via keyboard shortcut (Tab) or voice command.

---

## 3. System Architecture

### Pipeline (per frame)

```
Webcam frame (BGR)
        │
        ▼
[M1] CoreGestureEngine.process_frame()
        │  • cv2.flip(frame, 1)           — mirror
        │  • cv2.cvtColor(BGR → RGB)      — MediaPipe input
        │  • mp.solutions.hands.Hands()   — 21 landmarks
        │  • map_to_screen(lm8.x, lm8.y) — index tip → screen coords
        │  • ↕ coord_preprocessor hook ↕
[M2] KalmanFilter2D.update(raw_x, raw_y)
        │  • Predict:  x_pred = F @ x_state
        │  • Update:   K = P @ Hᵀ @ S⁻¹
        │  Returns (filtered_x, filtered_y) → M1 moves cursor
        │
        │  • Left click:  dist(LM4, LM8)  < 30 px → pyautogui.click()
        │  • Right click: dist(LM8, LM12) < 40 px → pyautogui.rightClick()
        │  • Scroll:      index+middle up, wrist moves
        │
        ▼
[M3] GestureModule.process(state)
        │  MOUSE mode:
        │    • wrist_y_buffer[-1] − [-2] = dy → pyautogui.scroll(−dy×sensitivity)
        │  ZOOM mode:
        │    • dist(left_thumb, right_thumb) → Ctrl+/−
        │  WHITEBOARD mode:
        │    • index only → cv2.line(canvas, prev_pt, curr_pt)
        │    • fist       → cv2.circle(canvas, erase)
        │    • thumb+ring → cycle color palette
        │  • cv2.addWeighted(frame, canvas) — blend overlay
        │
        ▼
[M4] VoiceContextModule.process_commands(state)
        │  Background daemon thread (SpeechRecognition):
        │    recognizer.listen() → recognize_google() → queue.put(action)
        │  Main thread drains queue:
        │    action → pyautogui hotkey / mode switch / kalman toggle / …
        │  Context poll (every 500 ms):
        │    pygetwindow.getActiveWindow() → keyword → profile → M3 sensitivity
        │
        ▼
[M5] UIModule.render_hud(state)
        │  • Semi-transparent top bar:
        │      [MODE badge]  CTX: BROWSER  SMOOTH ON  FPS 28.3
        │      Gesture: LEFT_CLICK   Voice: 'zoom in'
        │  • Bottom gesture guide bar (mode-specific)
        │  • Whiteboard colour swatch (WHITEBOARD mode only)
        │
        ▼
cv2.imshow("Gesture & Voice Virtual Mouse", frame)
        │
        ▼
Key handler → q=quit, TAB=cycle mode, k=kalman, c=clear canvas
```

### Module Dependency Graph

```
M1 (Core) ←── M2 (Kalman filter wraps M1 coord output)
    │
    ├── M3 (Novel gestures — reads wrist buffer + landmarks from M1)
    │
    ├── M4 (Voice + Context — sends commands back to M3/M2)
    │
    └── M5 (UI — reads status from all modules, renders HUD)
```

### Threading Model

```
Main thread (30 FPS loop)          Voice daemon thread
─────────────────────────          ─────────────────────
capture → M1 → M2 → M3            sr.Recognizer.listen()
→ M4.process_commands()    ←────   queue.put(action)
→ M5.render_hud()
→ cv2.imshow()
```

---

## 4. Project File Structure

```
gesture_mouse/
│
├── main.py                   ← Entry point — run this
├── config.py                 ← All thresholds, profiles, constants
├── requirements.txt          ← pip dependencies
│
├── modules/
│   ├── __init__.py
│   ├── core_engine.py        M1 — webcam, MediaPipe, cursor, click, scroll
│   ├── kalman_filter.py      M2 — Kalman filter, tremor smoothing
│   ├── gestures.py           M3 — velocity scroll, pinch-zoom, whiteboard
│   ├── voice_context.py      M4 — voice thread, context-aware profiles
│   └── ui.py                 M5 — HUD overlay, mode switcher
│
├── utils/
│   ├── __init__.py
│   ├── math_utils.py         euclidean_distance, map_to_screen, smooth_value
│   └── gesture_utils.py      is_finger_up, is_fist, is_index_only, …
│
├── test_m1.py                Standalone test for M1
├── test_m2.py                Unit test + live trail comparison for M2
├── test_m3.py                Unit test + live mode-switching demo for M3
└── test_m4.py                Unit test + live voice + context demo for M4
```

---

## 5. Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Webcam connected
- Microphone connected (required for M4 voice commands)
- Windows 10/11 (tested); Linux supported with minor changes

### Install Dependencies

```bash
cd gesture_mouse
pip install -r requirements.txt
```

If `pyaudio` fails on Windows:
```bash
pip install pipwin
pipwin install pyaudio
```

### Run the Full System

```bash
python main.py
```

### Run Individual Module Tests

```bash
python test_m1.py   # M1 core engine (webcam required)
python test_m2.py   # M2 Kalman filter (unit test + webcam)
python test_m3.py   # M3 gestures (webcam required)
python test_m4.py   # M4 voice + context (webcam + mic required)
```

### Keyboard Controls (main.py)

| Key | Action |
|-----|--------|
| `TAB` | Cycle modes: MOUSE → WHITEBOARD → ZOOM → VOICE |
| `k` | Toggle Kalman filter on/off |
| `c` | Clear whiteboard canvas |
| `q` | Quit |
| Move mouse to top-left corner | Emergency stop (PyAutoGUI failsafe) |

---

## 6. Configuration Reference (config.py)

All tunable parameters are centralised in `config.py`. No magic numbers exist in module code.

### Camera & Screen

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | Webcam index (change to `1` for secondary camera) |
| `FRAME_WIDTH` | `640` | Capture width in pixels |
| `FRAME_HEIGHT` | `480` | Capture height in pixels |
| `TARGET_FPS` | `30` | Target frame rate |

### MediaPipe

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_HANDS` | `2` | Maximum hands to detect simultaneously |
| `DETECTION_CONFIDENCE` | `0.7` | Minimum detection confidence threshold |
| `TRACKING_CONFIDENCE` | `0.5` | Minimum tracking confidence threshold |

### Cursor & Click

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CURSOR_SMOOTHING` | `0.5` | EMA blend factor (0=frozen, 1=instant) |
| `LEFT_CLICK_THRESHOLD` | `30` px | Thumb–index distance to trigger left click |
| `RIGHT_CLICK_THRESHOLD` | `40` px | Index–middle distance to trigger right click |
| `CLICK_COOLDOWN` | `0.30` s | Minimum time between repeated clicks |

### Scroll

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCROLL_DEADZONE` | `3` px | Minimum wrist delta-y to trigger scroll |
| `SCROLL_SENSITIVITY` | `0.8` | Default scroll multiplier |
| `VELOCITY_SCROLL_BUFFER_SIZE` | `5` | Rolling wrist-y buffer length |

### Kalman Filter

| Parameter | Default | Effect |
|-----------|---------|--------|
| `KALMAN_PROCESS_NOISE` | `0.01` | Low = strong smoothing (Parkinson's mode) |
| `KALMAN_MEASUREMENT_NOISE` | `0.1` | High = trust prediction over observation |
| `KALMAN_ENABLED_DEFAULT` | `True` | Filter on at startup |

### Zoom

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ZOOM_THRESHOLD` | `15` px | Minimum inter-thumb delta to trigger zoom |
| `ZOOM_COOLDOWN` | `0.40` s | Minimum time between zoom triggers |

### Whiteboard

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHITEBOARD_THICKNESS` | `3` | Drawing line thickness in pixels |
| `WHITEBOARD_ERASE_RADIUS` | `30` | Erase circle radius in pixels |
| `WHITEBOARD_ALPHA` | `0.3` | Canvas blend weight over webcam frame |

### Context Profiles

| Profile | Keywords | Scroll | Cursor | Special |
|---------|----------|--------|--------|---------|
| browser | chrome, firefox, edge | 1.2× | 1.0× | standard |
| ide | code, pycharm, sublime | 0.4× | 0.6× | precise |
| media | vlc, youtube, spotify | 1.0× | 1.0× | media_macros |
| present | powerpoint, slides | 0× | 1.0× | slide_nav |
| default | (unmatched) | 0.8× | 1.0× | standard |

---

## 7. Module M1 — Core Gesture Engine

**File:** `modules/core_engine.py`  
**Depends on:** Nothing  
**Estimated effort:** 6–8 hours

### Overview

Module 1 is the foundational layer. It opens the webcam, runs MediaPipe hand detection, maps landmark coordinates to screen space, and executes basic mouse actions via PyAutoGUI. Every other module either preprocesses data for M1 (M2) or extends its capabilities (M3, M4, M5).

### Key Components

**Webcam Capture**
```python
cap = cv2.VideoCapture(config.CAMERA_INDEX)
frame = cv2.flip(frame, 1)          # mirror horizontally
rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe expects RGB
```

**MediaPipe Hands**
```python
self.hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
results = self.hands.process(rgb_frame)
# results.multi_hand_landmarks → list of 21-point landmark sets
```

**Coordinate Mapping**
```python
# Landmark 8 = INDEX_FINGER_TIP, normalized [0,1]
raw_x = np.interp(lm8.x, [0.0, 1.0], [0, screen_width])
raw_y = np.interp(lm8.y, [0.0, 1.0], [0, screen_height])
```

**EMA Smoothing (pre-Kalman)**
```python
cursor_x = prev_x + 0.5 * (raw_x - prev_x)   # CURSOR_SMOOTHING = 0.5
```

**Click Detection**
```python
left_dist  = euclidean_distance(thumb_px, index_px)    # LM4 ↔ LM8
right_dist = euclidean_distance(index_px, middle_px)   # LM8 ↔ LM12

if left_dist  < 30 and cooldown_ok: pyautogui.click()
if right_dist < 40 and cooldown_ok: pyautogui.rightClick()
```

**M2 Integration Hook**
```python
self.coord_preprocessor = None   # M2 sets this to kalman.filter_coords
# In process_frame:
if self.coord_preprocessor:
    target_x, target_y = self.coord_preprocessor(raw_x, raw_y)
```

### State Dictionary (output of process_frame)

```python
{
    "landmarks"     : list[list[NormalizedLandmark]],  # per hand
    "handedness"    : list[str],                        # "Left"/"Right"
    "hand_count"    : int,
    "cursor_pos"    : (int, int),                       # screen pixels
    "raw_cursor"    : (int, int) | None,                # before smoothing
    "gesture"       : str,                              # "LEFT_CLICK" etc.
    "wrist_y_buffer": deque,                            # for M3 scroll
    "frame"         : np.ndarray,                       # annotated frame
    "frame_w"       : int,
    "frame_h"       : int,
}
```

### Gesture Map (M1 only)

| Gesture | Trigger | Action |
|---------|---------|--------|
| Cursor move | Index finger moves | `pyautogui.moveTo(x, y)` |
| Left click | Thumb+index pinch < 30 px | `pyautogui.click()` |
| Right click | Index+middle pinch < 40 px | `pyautogui.rightClick()` |
| Scroll | Index+middle up, wrist moves | `pyautogui.scroll(±3)` |

---

## 8. Module M2 — Kalman Filter (Tremor Smoothing)

**File:** `modules/kalman_filter.py`  
**Depends on:** M1  
**Estimated effort:** 4–6 hours  
**Academic importance:** Most novel component — explain clearly in viva.

### Overview

Module 2 is a preprocessing intercept positioned in the coordinate pipeline between M1's raw landmark output and the cursor action dispatcher. It implements a discrete-time Kalman filter that models the cursor as a point mass with position and velocity.

### Accessibility Impact

Parkinson's disease causes:
- Resting tremors: 4–6 Hz
- Action tremors: 5–12 Hz

The Kalman filter attenuates frequencies above ~3 Hz while preserving intentional DC (slow) movements. The cursor follows the user's intended trajectory rather than their physiological noise.

### Mathematical Design

**State vector** (4 × 1 column):
```
x = [px, py, vx, vy]ᵀ
```

**State transition matrix F** (constant-velocity model):
```
F(dt) = | 1  0  dt  0 |     px' = px + vx·dt
        | 0  1   0  dt|     py' = py + vy·dt
        | 0  0   1   0|     vx' = vx
        | 0  0   0   1|     vy' = vy
```

**Observation matrix H** (we observe position only):
```
H = | 1  0  0  0 |
    | 0  1  0  0 |
```

**PREDICT step:**
```
x_pred = F @ x
P_pred = F @ P @ Fᵀ + Q
```

**UPDATE step:**
```
S = H @ P_pred @ Hᵀ + R        (2×2 innovation covariance)
K = P_pred @ Hᵀ @ S⁻¹          (4×2 Kalman gain)
x = x_pred + K @ (z − H @ x_pred)
P = (I − K @ H) @ P_pred
```

**Tuning:**
- Low Q (0.01) → trust model → strong smoothing → Parkinson's mode
- High Q (1.0) → trust observation → minimal smoothing → normal mode

### Implementation (~35 lines of NumPy)

```python
class KalmanFilter2D:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self._state = np.zeros((4, 1))          # [px, py, vx, vy]
        self._P     = np.eye(4) * 1000.0        # high initial uncertainty
        self._H     = np.array([[1,0,0,0],[0,1,0,0]])
        self._Q     = np.eye(4) * process_noise
        self._R     = np.eye(2) * measurement_noise
        self._I4    = np.eye(4)

    def update(self, raw_x, raw_y):
        dt     = elapsed_since_last_call()
        F      = make_F(dt)
        x_pred = F @ self._state
        P_pred = F @ self._P @ F.T + self._Q
        z      = np.array([[raw_x], [raw_y]])
        S      = self._H @ P_pred @ self._H.T + self._R
        K      = P_pred @ self._H.T @ np.linalg.inv(S)
        self._state = x_pred + K @ (z - self._H @ x_pred)
        self._P     = (self._I4 - K @ self._H) @ P_pred
        return int(self._state[0,0]), int(self._state[1,0])
```

### Integration with M1

```python
m2 = KalmanFilterModule(engine)
# This single line registers M2 as M1's coordinate preprocessor:
engine.coord_preprocessor = self.filter_coords
# Now every process_frame() call routes raw coords through Kalman automatically.
```

### Presets

| Preset | Q | R | Use case |
|--------|---|---|---------|
| Tremor mode (`set_tremor_mode()`) | 0.01 | 0.5 | Parkinson's / high tremor |
| Normal mode (`set_normal_mode()`) | 1.0 | 0.1 | Healthy user, responsive cursor |

---

## 9. Module M3 — Novel Gesture Features

**File:** `modules/gestures.py`  
**Depends on:** M1, M2  
**Estimated effort:** 10–14 hours (largest module)

### Overview

Module 3 extends M1's gesture vocabulary with three novel interaction features. All features consume Kalman-smoothed coordinates. The module is a stateful dispatcher — each frame, `process(state)` routes to the correct feature handler based on the current mode.

---

### Feature A — Velocity-Proportional Scroll

**Problem:** Standard scroll uses a fixed ±3 unit increment regardless of hand speed.

**Solution:** Map wrist vertical velocity to scroll amount.

**Algorithm:**
1. Require index + middle fingers raised (scroll trigger posture).
2. Read the 5-frame rolling wrist-y buffer populated by M1.
3. Per-frame delta: `dy = buffer[-1] − buffer[-2]`
4. Apply deadzone: ignore `|dy| < 3 px`
5. `scroll_amount = int(dy × sensitivity_factor)`
6. `sensitivity_factor` overridden by M4 context profile (browser: 1.2, IDE: 0.4)
7. `pyautogui.scroll(−scroll_amount)`

```python
dy = wrist_y_buffer[-1] - wrist_y_buffer[-2]
if abs(dy) >= VELOCITY_SCROLL_DEADZONE:
    pyautogui.scroll(-int(dy * self._scroll_sensitivity))
```

**Gesture trigger:** Both index and middle fingers raised, ring and pinky curled.

---

### Feature B — Pinch-to-Zoom

**Problem:** No natural zoom gesture exists in existing virtual mouse systems.

**Solution:** Two-hand thumb distance → Ctrl+ / Ctrl−.

**Algorithm:**
1. Require both hands detected (`hand_count ≥ 2`).
2. Identify left/right hands from MediaPipe `handedness` classification.
3. Compute `D = euclidean_distance(left_thumb_px, right_thumb_px)`.
4. `D_current − D_previous > ZOOM_THRESHOLD (15 px)` → `Ctrl++`
5. `D_current − D_previous < −ZOOM_THRESHOLD` → `Ctrl−`
6. Cooldown of 400 ms prevents rapid repeated triggers.

```python
delta = current_dist - self._prev_zoom_dist
if delta >  ZOOM_THRESHOLD: pyautogui.hotkey("ctrl", "+")
if delta < -ZOOM_THRESHOLD: pyautogui.hotkey("ctrl", "-")
```

---

### Feature C — Air Whiteboard

**Problem:** No touchless annotation tool exists for presentations.

**Solution:** Transparent canvas overlay with gesture-based drawing.

**Canvas:** `np.zeros((h, w, 3), dtype=np.uint8)` — same size as webcam frame.

**Gesture → Action Mapping:**

| Gesture | Detection | Action |
|---------|-----------|--------|
| Index finger only raised | `is_index_only(landmarks)` | `cv2.line(canvas, prev_pt, curr_pt, color, 3)` |
| Closed fist | `is_fist(landmarks)` | `cv2.circle(canvas, wrist_center, 30, (0,0,0), -1)` |
| Thumb tip near ring tip (< 35 px) | distance check | Cycle colour palette |
| Any other posture | — | Pen-up (break line) |

**Colour palette:** `[red, green, blue, white, yellow]`

**Canvas overlay:**
```python
# Mask-based blend — only drawn pixels alter the frame
mask         = np.any(canvas != 0, axis=2)
output[mask] = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)[mask]
```

**Canvas persistence:** The canvas is NOT cleared on mode switch — the user can return to their drawing when switching back to WHITEBOARD mode.

---

## 10. Module M4 — Voice Control + Context-Aware Profiles

**File:** `modules/voice_context.py`  
**Depends on:** M1 (M2 and M3 for cross-references)  
**Estimated effort:** 8–10 hours

### Sub-system 1 — Voice Command Recognition

**Architecture:** SpeechRecognition in a daemon thread pushes actions into a `queue.Queue`. The main loop drains the queue once per frame.

```python
# Thread start
threading.Thread(target=self._voice_listener, daemon=True).start()

# Voice thread body
audio = recognizer.listen(mic, timeout=3, phrase_time_limit=4)
text  = recognizer.recognize_google(audio)   # online
# fallback: recognizer.recognize_sphinx(audio)  # offline
if phrase in text:
    self._cmd_queue.put(action)

# Main thread (each frame)
while not self._cmd_queue.empty():
    action = self._cmd_queue.get_nowait()
    self._execute(action, state)
```

**Why daemon thread?**  
`recognizer.listen()` is blocking. Running it in the main loop would freeze the webcam at 0 FPS. A daemon thread is automatically killed when the main process exits — no orphaned processes.

**Why `queue.Queue`?**  
Thread-safe by design — uses internal locks to prevent data races between the producer (voice thread) and consumer (main thread).

### Supported Voice Commands

| Say | Action |
|-----|--------|
| "open browser" | `pyautogui.hotkey("ctrl", "t")` — new tab |
| "take screenshot" | `pyautogui.screenshot(path)` — saves PNG |
| "enable smooth mode" | Toggle Kalman filter (M2) on/off |
| "whiteboard on" | Switch to WHITEBOARD mode |
| "whiteboard off" | Switch to MOUSE mode |
| "zoom in" | `pyautogui.hotkey("ctrl", "+")` |
| "zoom out" | `pyautogui.hotkey("ctrl", "-")` |
| "scroll up fast" | 2× scroll speed for 3 seconds |
| "scroll down fast" | 2× scroll speed for 3 seconds |
| "switch to mouse" | MOUSE mode |
| "switch to whiteboard" | WHITEBOARD mode |
| "switch to zoom" | ZOOM mode |

### Sub-system 2 — Context-Aware Profiles

**How it works:**
1. Poll `pygetwindow.getActiveWindow().title` every 500 ms.
2. Convert to lowercase.
3. Check each profile's keyword list for a substring match.
4. On profile change: call `m3.set_scroll_sensitivity(profile["scroll_speed"])`.

```python
def _detect_profile(window_title):
    for profile_name, profile in CONTEXT_PROFILES.items():
        for keyword in profile["keywords"]:
            if keyword in window_title:
                return profile_name
    return "default"
```

**Context Profile Table:**

| Active Window | Profile | Scroll Speed | Cursor Sensitivity |
|---------------|---------|-------------|-------------------|
| Chrome / Firefox / Edge | browser | 1.2× | 1.0× |
| VS Code / PyCharm / Sublime | ide | 0.4× | 0.6× |
| VLC / YouTube / Spotify | media | 1.0× | 1.0× |
| PowerPoint / Google Slides | present | 0× | 1.0× |
| (unrecognised) | default | 0.8× | 1.0× |

---

## 11. Module M5 — Playground UI + System Integration

**Files:** `modules/ui.py`, `main.py`  
**Depends on:** M1, M2, M3, M4  
**Estimated effort:** 5–7 hours

### HUD Layout

```
┌──────────────────────────────────────────────────────────────┐
│ [MOUSE]  CTX: BROWSER   SMOOTH ON              FPS  28.3    │
│          Gesture: LEFT_CLICK     Voice: 'zoom in'           │
│          k=kalman  c=clear  TAB=mode  q=quit                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    WEBCAM FEED                               │
│               (with hand skeleton overlay)                   │
│                                              ┌────┐          │
│                                              │CLR │ ← swatch │
│                                              └────┘          │
├──────────────────────────────────────────────────────────────┤
│ Move: index tip  |  L-Click: pinch  |  Scroll: index+middle │
└──────────────────────────────────────────────────────────────┘
```

### HUD Elements

| Element | Location | Data source |
|---------|----------|-------------|
| Mode badge (coloured pill) | Top-left | `m5.mode` |
| Context profile | Top-left | `m4.profile_display` |
| Kalman status | Top-left | `m2.status_text / status_color` |
| FPS counter | Top-right | Rolling 10-frame average |
| Voice status | Top-right | `m4.voice_status` |
| Gesture label | Top-left | `state["m3_gesture"]` |
| Keyboard hint | Top-right | Static text |
| Gesture guide | Bottom bar | `_GESTURE_GUIDE[mode]` |
| Colour swatch | Top-right | `m3._draw_color` (WHITEBOARD only) |

### Mode Switcher

```python
# Keyboard: Tab key
self._mode_idx = (self._mode_idx + 1) % len(config.MODES)
self._mode     = config.MODES[self._mode_idx]
m3.set_mode(self._mode)

# Voice: "switch to whiteboard" → M4 sets state["mode"] → M5 syncs
m5.mode = state.get("mode", m5.mode)
```

### Integration Loop (main.py)

```python
while True:
    ret, frame = cap.read()
    frame  = cv2.flip(frame, 1)

    state  = engine.process_frame(frame)   # M1 + M2
    state["mode"] = m5.mode

    state  = m3.process(state)             # M3
    state  = m4.process_commands(state)    # M4

    m5.mode = state.get("mode", m5.mode)   # sync mode from M4 voice

    frame  = m5.render_hud(state)          # M5
    cv2.imshow("Gesture & Voice Virtual Mouse", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    m5.handle_key(key, state)
    m3.mode = m5.mode
    m4.mode = m5.mode
```

---

## 12. Full Pipeline Walkthrough

### Scenario: User moves hand right to move cursor

1. **M1** captures frame, flips it, converts to RGB.
2. **M1** runs MediaPipe → detects index finger tip at normalized (0.6, 0.4).
3. **M1** calls `coord_preprocessor(raw_x=384, raw_y=240)` (screen-mapped coords).
4. **M2** `KalmanFilter2D.update(384, 240)`:
   - Predicts `x_pred = x + vx·dt`
   - Updates with new observation
   - Returns `(382, 239)` — slightly smoothed
5. **M1** applies EMA: `cursor_x = prev + 0.5 × (382 − prev)`
6. **M1** calls `pyautogui.moveTo(381, 238)`.
7. **M3** `_velocity_scroll()` — index+middle not both up → returns `"NONE"`.
8. **M4** drains empty queue, polls context → `"browser"` profile → sets scroll sensitivity 1.2.
9. **M5** renders HUD: mode=MOUSE, CTX=BROWSER, SMOOTH ON, FPS=29.1.
10. Frame displayed. User sees their hand with skeleton and the cursor moved.

### Scenario: User speaks "zoom in"

1. **M4 voice thread** (background): `listen()` captures audio.
2. `recognize_google()` returns `"zoom in"`.
3. `"zoom in" in text` matches `config.VOICE_COMMANDS["zoom in"] = "zoom_in"`.
4. `queue.put("zoom_in")`.
5. **Next frame** — main thread drains queue: `_execute("zoom_in", state)`.
6. `pyautogui.hotkey("ctrl", "+")` → browser zooms in.
7. HUD shows `Voice: 'zoom in'`.

---

## 13. Troubleshooting Guide

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Cannot open camera` | No webcam / wrong index | Change `CAMERA_INDEX` in config.py |
| Cursor too jumpy / shaky | Kalman filter off or Q too high | Press `k` to enable; lower `KALMAN_PROCESS_NOISE` in config.py |
| Click fires continuously | Click threshold too large | Lower `LEFT_CLICK_THRESHOLD` (e.g. 25) in config.py |
| No scroll response | Wrong posture or too slow | Raise both index + middle fingers; move wrist faster |
| Zoom fires accidentally | ZOOM_THRESHOLD too low | Raise `ZOOM_THRESHOLD` (e.g. 25) in config.py |
| Whiteboard draws continuously | `is_index_only` misclassifying | Curl ring and pinky more firmly |
| `No microphone` error | Mic not connected or no permission | Windows Settings → Privacy → Microphone → Allow |
| `SpeechRecognition not installed` | Missing package | `pip install SpeechRecognition pyaudio` |
| `pyaudio` install fails on Windows | Wheel not available | `pip install pipwin && pipwin install pyaudio` |
| Voice recognition slow | Network latency | Use offline Sphinx: install `pocketsphinx` |
| Context always DEFAULT | `pygetwindow` not installed | `pip install pygetwindow` |
| Low FPS (< 15) | High CPU load | Lower `FRAME_WIDTH=320, FRAME_HEIGHT=240` in config.py |
| `pyautogui.FailSafeException` | Mouse reached top-left corner | Move mouse away; this is an intentional emergency stop |
| MediaPipe import error | Wrong version | `pip install mediapipe==0.10.14` |

---

## 14. Viva Questions and Answers

### Module M1 — Core Gesture Engine

**Q1. Why is the webcam frame flipped horizontally?**

Without flipping, moving your hand right moves the cursor left (mirrored). `cv2.flip(frame, 1)` creates a mirror effect so hand direction matches cursor direction intuitively — the same way a selfie camera works.

**Q2. Why convert BGR to RGB before passing to MediaPipe?**

OpenCV reads frames in BGR channel order by default. MediaPipe's neural network was trained on RGB images. Passing BGR would swap the red and blue channels and degrade landmark detection accuracy.

**Q3. What is exponential moving average (EMA) smoothing and why is it used in M1?**

EMA blends the new raw value and the previous output:  
`output = prev + factor × (new − prev)`  
With `factor=0.5`, the cursor is pulled halfway toward the raw position each frame, dampening high-frequency jitter without adding noticeable lag. M2 replaces this with a physics-based Kalman filter for superior noise removal.

**Q4. How is click detection made robust against accidental repeated triggers?**

A cooldown timer stores `time.time()` at each click. The next click fires only if `now − last_click > CLICK_COOLDOWN (0.3 s)`. This prevents a held pinch from firing 30 clicks per second at 30 FPS.

**Q5 (Advanced). Why is `pyautogui.PAUSE` set to 0?**

PyAutoGUI inserts a 0.1-second sleep after every `moveTo()` by default as a developer safety guard. At 30 FPS, this would cap the effective cursor update rate at 10 FPS. Setting `PAUSE = 0` removes the ceiling. Safety is preserved because `FAILSAFE = True` is kept — the pause was a convenience guard, not a security control.

---

### Module M2 — Kalman Filter

**Q1. What is the Kalman state vector and why does it include velocity?**

The state is `[x, y, vx, vy]`. Including velocity allows the filter to *predict* where the cursor will be in the next frame using the motion model `x' = x + vx·dt`. Without velocity, the filter would only blend the current observation with the last position — essentially EMA. With velocity, it distinguishes "intentional fast movement" from "jitter in place."

**Q2. What is the Kalman gain K and what does it control?**

K is a 4×2 matrix: `K = P_pred @ Hᵀ @ S⁻¹`. It is the optimal weighting between the model's prediction and the new observation. K ≈ 0 → trust the prediction (ignore noisy sensor). K ≈ 1 → trust the observation (ignore the model). K is computed automatically from Q and R — this is what makes the filter "optimal."

**Q3. Why is the initial covariance P set to 1000·I?**

P represents uncertainty in the current state estimate. Setting it large at startup means "I have no idea where the cursor is" — the first observation is trusted almost completely (K ≈ 1), so the cursor immediately snaps to the detected fingertip rather than starting at the screen centre.

**Q4. How does the filter handle the hand leaving and re-entering the frame?**

`reset()` is called when the hand disappears: P returns to 1000·I, the velocity estimate is zeroed. Without reset, stale velocity from the previous tracking session would cause the cursor to drift rapidly the moment the hand reappears.

**Q5 (Advanced). Why does low Q specifically attenuate Parkinson's tremors?**

Q is the variance of the process noise — how much the motion model is allowed to "surprise" itself. A low Q forces the filter to trust its own velocity prediction over the incoming observation. Parkinson's tremors are high-frequency (5–12 Hz) with zero mean over one period. The constant-velocity model cannot reproduce a zero-mean oscillation, so with low Q the Kalman gain is small and the oscillating observation barely perturbs the smooth prediction — the tremor is effectively low-pass filtered. Intentional slow movements accumulate a consistent velocity direction over multiple frames, so the model converges to them correctly.

---

### Module M3 — Novel Gesture Features

**Q1. How does velocity-proportional scroll differ from standard fixed scroll?**

Standard scroll sends fixed ±3 units per trigger. Velocity scroll tracks the wrist y-position across a 5-frame rolling buffer, computes `dy = buffer[-1] − buffer[-2]` per frame, and multiplies by a sensitivity factor. Fast swipes → large dy → large scroll. Slow movement → fine-grained positioning. This matches the natural "flick" behaviour users expect from touchscreens.

**Q2. Why is a deadzone applied in velocity scroll?**

At rest, MediaPipe landmarks jitter by 1–3 pixels even with Kalman filtering. Without a deadzone, this jitter would trigger continuous tiny scrolls. `VELOCITY_SCROLL_DEADZONE = 3 px` ignores all dy below 3 pixels, keeping the page still when the hand is stationary.

**Q3. How does pinch-to-zoom identify which hand is left and which is right?**

MediaPipe returns a `handedness` classification alongside each detected hand. Each entry is `"Left"` or `"Right"`. The module iterates both hands, assigns thumb tip pixels to the correct label, then measures Euclidean distance between the two thumb tips. Without handedness, the system cannot distinguish which thumb belongs to which hand.

**Q4. How does the whiteboard canvas persist across mode switches?**

The canvas is stored as `self._canvas` — a numpy array kept alive for the lifetime of the GestureModule instance. `set_mode()` intentionally does **not** call `clear_canvas()`. Only an explicit fist gesture (erase) or voice command clears it.

---

### Module M4 — Voice Control + Context-Aware Profiles

**Q1. Why does the voice listener run in a daemon thread?**

`recognizer.listen()` is blocking — it waits for speech. Running it in the main loop would freeze the webcam at 0 FPS while listening. A daemon thread runs concurrently. `daemon=True` ensures the thread is automatically killed when the main program exits — no orphaned processes.

**Q2. What is a thread-safe queue and why is it needed here?**

`queue.Queue` uses internal locks so multiple threads can `put()` and `get()` safely without data races. The voice thread writes into the queue; the main thread reads from it. Without a thread-safe structure, concurrent list access would cause race conditions and unpredictable crashes.

**Q3. How does context-aware profiling work without ML or NLP?**

Simple substring matching: `if keyword in window_title`. The active window title is retrieved via `pygetwindow` every 500 ms, converted to lowercase, and checked against each profile's keyword list. "chrome" appears in "Google Chrome — New Tab" → browser profile. No ML is needed because window titles are deterministic.

**Q4. What is `adjust_for_ambient_noise()` and why call it once at startup?**

It samples the microphone for ~1 second and sets `energy_threshold` to the observed background noise level, calibrating the "silence baseline." Called once at startup so the recogniser can reliably distinguish speech from ambient noise without manual threshold tuning.

**Q5 (Advanced). How would you make voice recognition work fully offline?**

Replace `recognize_google()` with `recognize_vosk()` using a downloaded Vosk model. Vosk runs entirely on-device, supports 20+ languages, runs at ~2–5× real time on CPU, and provides word-level confidence scores. The existing code already implements `recognize_sphinx()` as a network-failure fallback. For production offline use, Vosk is preferred over Sphinx due to better accuracy and active maintenance.

---

### Module M5 — Playground UI + System Integration

**Q1. What is the role of the shared state dict in the pipeline?**

The `state` dictionary is a per-frame data bus. M1 writes landmarks, cursor position, and the annotated frame into it. M3 reads the wrist buffer and writes `m3_gesture`. M4 reads/writes `mode`, `context_profile`, and `voice_status`. M5 reads all of these to render the HUD. One dict passed through the pipeline is simpler and more extensible than threading individual variables between module calls.

**Q2. Why is M4's voice thread a daemon thread?**

Daemon threads are automatically killed when the main thread exits. If the voice thread were non-daemon, the process would hang after the user presses `q` because `recognizer.listen()` would still be blocking on the microphone.

**Q3. How does mode switching stay consistent across M3, M4, and M5?**

All three read/write `state["mode"]`. M5's `handle_key()` updates `m5.mode` and `state["mode"]`. At the end of each loop iteration the main loop explicitly syncs: `m3.mode = m5.mode` and `m4.mode = m5.mode`. This single-source-of-truth pattern via the state dict prevents the three modules from diverging.

**Q4. Why is `cv2.addWeighted()` used for the HUD panel instead of solid rectangles?**

A solid black rectangle would fully obscure the webcam feed behind it. `addWeighted` blends the opaque overlay with the original frame at 55/45, producing a semi-transparent panel. The user can see their hand behind the HUD, which is important for real-time gesture feedback.

**Q5 (Advanced). How would you optimise the system to reach stable 30 FPS on a low-end laptop?**

Three targeted changes:  
(1) Reduce MediaPipe input resolution — the model uses an internal 192×192 image regardless. Setting `FRAME_WIDTH=320, FRAME_HEIGHT=240` cuts decoding work with no detection quality loss.  
(2) Set `model_complexity=0` in `mp.solutions.hands.Hands()` — uses the lightweight hand model (2–3× faster, slightly less accurate at extreme angles).  
(3) Move `cv2.imshow()` to a separate display thread with a frame queue — decouples render latency from capture latency so a slow render does not delay the next MediaPipe cycle.

---

## 15. Build Summary

| Module | Features | Novel? | Effort | Depends On |
|--------|---------|--------|--------|-----------|
| M1 — Core Engine | Cursor, click, scroll | Base | 6–8 hrs | None |
| M2 — Kalman Filter | Tremor smoothing, accessibility | **YES** | 4–6 hrs | M1 |
| M3 — Novel Gestures | Velocity scroll, pinch-zoom, whiteboard | **YES** | 10–14 hrs | M1, M2 |
| M4 — Voice + Context | Speech commands, app-aware profiles | **YES** | 8–10 hrs | M1 |
| M5 — UI + Integration | HUD, playground, main loop | **YES** | 5–7 hrs | M1–M4 |
| **Total** | **4 novel features** | | **33–45 hrs** | |

### Build Order

```
M1 (build + test fully)
  └── M2 (add Kalman filter, test tremor reduction)
        ├── M3 (novel gestures — built by one team member)
        └── M4 (voice + context — built by other team member in parallel)
              └── M5 (integration + HUD — built last when M1–M4 stable)
```

### How to Run

```bash
cd gesture_mouse
pip install -r requirements.txt
python main.py
```

---

*Document generated for 24CSE48 Mini Project · New Horizon College of Engineering · 2026*
