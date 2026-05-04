# Gesture and Voice Controlled Virtual Mouse

## Overview
The Gesture and Voice Controlled Virtual Mouse is a robust, hands-free computer control application. Designed for accessibility, presentations, and interactive displays, it allows users to control their computer using hand gestures (tracked via webcam) and voice commands. It solves the problem of requiring physical input devices, providing a seamless, futuristic, and accessible interface for interacting with any desktop environment.

## Features

### Core Mouse Features
*   **Cursor Control:** Precise cursor movement mapped to the right-hand index finger.
*   **Mouse Clicks:** Left, right, and double clicks triggered by specific thumb-to-finger pinches.
*   **Drag & Drop:** Fist gesture with hysteresis to securely grab, drag, and drop items.
*   **Smooth Cursor Movement:** Integrated Kalman Filter to reduce hand tremors and jitter.

### Advanced Gesture Features
*   **Two-Handed Scrolling:** Scroll up/down using left-hand finger counts.
*   **Continuous Scrolling:** Sustained scroll actions for seamless navigation.
*   **Context-Aware Scrolling:** Automatically adjusts scroll sensitivity based on the active application (e.g., faster in browsers, slower in IDEs).
*   **Pinch-to-Zoom:** Open palm to zoom in, fist to zoom out.
*   **Media Control:** Adjust system volume and brightness using intuitive hand movements.

### Air Whiteboard Features
*   **Transparent Overlay:** Draw directly on the screen over active applications.
*   **Drawing & Erasing:** Draw with the index finger, erase with a fist.
*   **Color Palette:** Pinch to cycle through different brush colors.
*   **Shape Drawing:** Voice-activated shape drawing (circles, rectangles, stars, etc.).
*   **Save Canvas:** Export the current drawing to a PNG file.

### Voice Control Features
*   **App Launching:** Fuzzy-matched voice commands to open applications (e.g., "open chrome", "launch spotify").
*   **Web Search & Navigation:** "search for [query]" or "go to [website.com]".
*   **Keyboard Shortcuts:** Trigger common hotkeys (copy, paste, undo, save) via voice.
*   **Dictation:** Type text directly using "type [phrase]".
*   **Mode Switching:** Seamlessly switch between Mouse, Whiteboard, Zoom, and Media modes.

## Tech Stack

### Frontend / UI
*   **OpenCV (`cv2`):** Real-time webcam feed processing, HUD rendering, and Air Whiteboard canvas.

### Core Processing & ML
*   **MediaPipe:** High-performance, real-time hand landmark detection.
*   **NumPy:** Mathematical operations and Kalman Filter matrix calculations.

### System Control & Integration
*   **PyAutoGUI:** Cross-platform mouse and keyboard simulation.
*   **PyGetWindow:** Active window detection for context-aware profiles.
*   **Screen Brightness Control (`screen_brightness_control`):** Adjusting display brightness.

### Voice & Natural Language
*   **SpeechRecognition:** Audio capture and speech-to-text conversion (Google Speech API & CMU Sphinx fallback).
*   **FuzzyWuzzy:** Fuzzy string matching for highly accurate and forgiving voice command execution.

## Architecture
The application follows a modular, multi-threaded architecture driven by a 30 FPS core processing loop:
1.  **Input Layer:** A webcam feed is captured via OpenCV and processed frame-by-frame. Voice audio is continuously listened to in an asynchronous background daemon thread.
2.  **Tracking Layer:** MediaPipe extracts 3D hand landmarks from the frame.
3.  **Smoothing Layer:** Raw coordinates pass through a Kalman Filter to stabilize movement.
4.  **Action Mapping Layer:** Hand postures and positions are mapped to OS-level inputs via PyAutoGUI. Concurrently, the Voice engine parses transcribed text through regex and fuzzy matching to trigger macros.
5.  **Context Layer:** The system polls the OS for the active window title to dynamically swap physics profiles (e.g., scroll speed).
6.  **Presentation Layer:** A visual HUD and whiteboard canvas are alpha-blended over the webcam feed and displayed to the user.

## Folder Structure
```text
gesture_mouse/
├── config.py                 # Centralized settings, thresholds, and command mappings
├── main.py                   # Application entry point and main execution loop
├── GESTURE_GUIDE.txt         # User guide for physical gestures
├── VOICE_COMMANDS.txt        # Comprehensive list of 70+ voice commands
├── requirements.txt          # Python dependencies
├── modules/                  # Core feature modules
│   ├── core_engine.py        # M1: MediaPipe integration, cursor, clicks, and drag logic
│   ├── kalman.py             # M2: Kalman filter implementation for tremor reduction
│   ├── gestures.py           # M3: Scroll, Zoom, Media, and Whiteboard logic
│   ├── voice_context.py      # M4: Speech recognition daemon and context-aware polling
│   └── ui.py                 # M5: Heads-Up Display (HUD) and overlay rendering
└── utils/                    # Utility scripts and helpers
```

## Installation & Setup

### Prerequisites
*   Python 3.10+
*   A working webcam
*   Microphone (for voice commands)

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Ankith-R-Rai/Virtual-Mouse.git
    cd Virtual-Mouse/gesture_mouse
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Voice & OS Dependencies (Linux/Mac only):**
    *   *Linux:* `sudo apt install xdotool portaudio19-dev python3-pyaudio`
    *   *Mac:* `brew install portaudio`

## Usage
Run the application using the main script:
```bash
python main.py
```
*   **Cycle Modes:** Press `TAB` on your keyboard (or say "switch to [mode]") to cycle between MOUSE, WHITEBOARD, ZOOM, VOICE, and MEDIA modes.
*   **Clear Canvas:** Press `c` on your keyboard (or say "clear canvas").
*   **Toggle Smoothing:** Press `k` (or say "enable smooth mode") to toggle the Kalman Filter.
*   **Quit:** Press `q` while the webcam window is focused to exit the application.
*   **Emergency Stop:** Move your physical mouse to the very top-left corner of your screen to trigger PyAutoGUI's fail-safe.

## Key Components / Modules
*   **`config.py`**: The brain of the configuration. Contains all tuning parameters, context profiles (browsers vs IDEs), fuzzy matching thresholds, and the dictionary mapping voice phrases to internal actions.
*   **`modules/core_engine.py` (CoreGestureEngine)**: Handles the MediaPipe lifecycle. Responsible for detecting strict hand postures (e.g., fists) and mapping the index finger to screen coordinates.
*   **`modules/voice_context.py` (VoiceContextModule)**: Runs a separate `threading.Thread` to constantly listen to the microphone without blocking the video feed. Uses Regex for dynamic commands and `fuzzywuzzy` for static macros. Also handles `pygetwindow` polling for context awareness.
*   **`modules/gestures.py` (GestureModule)**: Contains the state machines for complex physical interactions, such as hysteresis-based drag-and-drop, sticky scrolling, and OpenCV shape drawing for the whiteboard.

## Environment Variables
> Note: Inferred based on code structure. The application does not currently require explicit `.env` variables, as API integrations (like Google Speech) use default unauthenticated/free tiers provided by the `SpeechRecognition` library.

## Future Improvements
*   **Hand-Label Disambiguation:** Fix instances where MediaPipe swaps "Left" and "Right" hand labels when hands cross.
*   **Depth-Insensitive Zoom:** Filter out Z-axis (depth) movements to prevent accidental zoom micro-steps when the hand moves closer to the camera.
*   **Auto-Save Canvas:** Automatically persist the OpenCV whiteboard canvas to disk upon exit.
*   **Local NLP Engine:** Completely replace the Google Speech API with a local, offline LLM or advanced Sphinx model for zero-latency voice execution.
