"""
modules/voice_context.py — M4: Voice Control + Context-Aware Profiles

Two sub-systems:

1. Voice Command Recognition
   -------------------------
   SpeechRecognition runs in a daemon thread so audio capture never blocks
   the 30-FPS main loop.  Recognised phrases are pushed into a thread-safe
   queue.Queue.  The main loop drains the queue once per frame via
   process_commands().

   Online:  recognize_google()  (requires internet)
   Offline: recognize_sphinx()  (requires pocketsphinx, no internet needed)

2. Context-Aware Profiles
   ----------------------
   The active window title is polled every CONTEXT_POLL_INTERVAL seconds.
   Detected keywords map the window to a sensitivity profile stored in
   config.CONTEXT_PROFILES.  Profile changes silently update M3's scroll
   sensitivity and are displayed in the M5 HUD.
"""

import sys
import os
import re
import time
import queue
import subprocess
import threading
import urllib.parse
import webbrowser

import pyautogui

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process as fuzz_proc
    _HAS_FUZZY = True
except ImportError:
    _HAS_FUZZY = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config


# ---------------------------------------------------------------------------
# Window title helpers (platform-agnostic)
# ---------------------------------------------------------------------------

def _get_active_window_title() -> str:
    """
    Return the title of the currently focused window as a lower-case string.

    Tries pygetwindow (Windows / macOS) first, then falls back to
    xdotool on Linux.  Returns an empty string on any failure.
    """
    # Windows / macOS via pygetwindow
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        return win.title.lower() if win else ""
    except Exception:
        pass

    # Linux via xdotool
    try:
        import subprocess
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            capture_output=True, text=True, timeout=0.3,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _detect_profile(window_title: str) -> str:
    """
    Map a window title to a context profile name.

    Iterates over CONTEXT_PROFILES (skipping "default") and returns the
    first profile whose keyword list has a match in the title string.
    Falls back to "default" if no keyword matches.
    """
    for profile_name, profile in config.CONTEXT_PROFILES.items():
        if profile_name == "default":
            continue
        for keyword in profile["keywords"]:
            if keyword in window_title:
                return profile_name
    return "default"


# ---------------------------------------------------------------------------
# VoiceContextModule
# ---------------------------------------------------------------------------

class VoiceContextModule:
    """
    M4 — Voice Control + Context-Aware Profiles.

    Public interface
    ----------------
    start_voice()                  Launch the background voice listener.
    stop_voice()                   Signal the thread to stop.
    process_commands(state) -> dict  Drain queue, poll context, return state.
    profile_display                Property — profile name for HUD.
    voice_status                   Property — last recognised phrase for HUD.
    """

    def __init__(self, engine, gesture_module=None, kalman_module=None) -> None:
        """
        Args:
            engine         : CoreGestureEngine (M1) — kept for future expansion.
            gesture_module : GestureModule (M3)     — receives sensitivity updates.
            kalman_module  : KalmanFilterModule (M2) — toggled by voice command.
        """
        self.engine  = engine
        self.gesture = gesture_module
        self.kalman  = kalman_module

        # ---- Thread-safe voice command queue -------------------------
        self._cmd_queue: queue.Queue    = queue.Queue()
        self._voice_thread: threading.Thread | None = None
        self._voice_running: bool       = False
        self._last_voice_text: str      = "not started"

        # ---- Context profile state -----------------------------------
        self.current_profile_name: str  = "default"
        self.current_profile: dict      = config.CONTEXT_PROFILES["default"]
        self._last_context_poll: float  = 0.0
        self._active_window_title: str  = ""

        # ---- Scroll boost state (from "scroll fast" voice cmd) -------
        self._scroll_boost_until: float = 0.0
        self._continuous_scroll: tuple[str, float] = ("", 0.0)

        # ---- Shared mode reference (kept in sync with M3 / M5) -------
        self.mode: str = "MOUSE"

    # ------------------------------------------------------------------
    # Voice listener thread
    # ------------------------------------------------------------------

    def start_voice(self) -> None:
        """
        Launch the background voice listener daemon thread.

        Safe to call multiple times — does nothing if already running.
        """
        if self._voice_thread and self._voice_thread.is_alive():
            return
        self._voice_running = True
        self._voice_thread  = threading.Thread(
            target=self._voice_listener,
            daemon=True,
            name="VoiceListener",
        )
        self._voice_thread.start()
        print("[M4] Voice listener started.")

    def stop_voice(self) -> None:
        """Signal the voice thread to exit at the next loop iteration."""
        self._voice_running = False

    def _voice_listener(self) -> None:
        """
        Daemon thread body: listens continuously and pushes command strings
        into _cmd_queue.

        Flow
        ----
        1. Import SpeechRecognition (graceful error if not installed).
        2. Calibrate microphone energy threshold once for ambient noise.
        3. Loop: listen() → recognise → match against VOICE_COMMANDS dict
           → put(action) into queue.
        4. Fallback to recognize_sphinx() when Google returns RequestError.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            print("[M4] ERROR: SpeechRecognition not installed.\n"
                  "       Run: pip install SpeechRecognition pyaudio")
            self._last_voice_text = "SpeechRecognition not installed"
            return

        recognizer                          = sr.Recognizer()
        recognizer.pause_threshold           = 1.2    # longer silence gap to prevent cutting off
        recognizer.energy_threshold          = 300    # ambient mic sensitivity
        recognizer.dynamic_energy_threshold  = True   # auto-adjust on the fly

        # Open the microphone once to confirm it exists.
        try:
            mic = sr.Microphone()
        except Exception as exc:
            print(f"[M4] ERROR: No microphone detected — {exc}")
            self._last_voice_text = "No microphone"
            return

        # One-time calibration for ambient noise (1 second).
        with mic as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[M4] Microphone calibrated for ambient noise.")
            except Exception:
                pass   # Non-fatal — proceed with default threshold

        self._last_voice_text = "Listening…"
        print("[M4] Listening for voice commands…")

        with mic as source:
            while self._voice_running:
                try:
                    # timeout=5   : wait 5s for speech
                    # phrase_time_limit=8 : max utterance length
                    audio = recognizer.listen(
                        source, timeout=5, phrase_time_limit=8,
                    )

                    # ---- Try online recognition --------------------------
                    try:
                        text = recognizer.recognize_google(audio).lower().strip()
                    except sr.UnknownValueError:
                        # Google heard audio but could not decode speech.
                        continue
                    except sr.RequestError:
                        # Network unavailable — try offline Sphinx fallback.
                        try:
                            text = recognizer.recognize_sphinx(audio).lower().strip()
                        except Exception:
                            continue

                    if not text:
                        continue

                    print(f"[M4] Recognised: '{text}'")
                    self._last_voice_text = f"'{text}'"

                    # ---- Match: regex patterns first, then fuzzy -----
                    action = self._match_command(text)
                    if action:
                        self._cmd_queue.put(action)

                except sr.WaitTimeoutError:
                    # No speech in the 3-second window — loop and try again.
                    continue
                except Exception as exc:
                    print(f"[M4] Voice thread error: {exc}")
                    time.sleep(0.5)

    # ------------------------------------------------------------------
    # Context profile polling
    # ------------------------------------------------------------------

    def _poll_context(self) -> None:
        """
        Check the active window every CONTEXT_POLL_INTERVAL seconds and
        update M3 sensitivity parameters if the profile has changed.
        """
        now = time.time()
        if now - self._last_context_poll < config.CONTEXT_POLL_INTERVAL:
            return
        self._last_context_poll = now

        title        = _get_active_window_title()
        profile_name = _detect_profile(title)

        if profile_name != self.current_profile_name:
            self.current_profile_name = profile_name
            self.current_profile      = config.CONTEXT_PROFILES[profile_name]
            self._apply_profile()
            print(
                f"[M4] Context → {profile_name.upper()}  "
                f"(window: '{title[:50]}')"
            )

        self._active_window_title = title

    def _apply_profile(self) -> None:
        """Push scroll sensitivity from the new profile into M3."""
        if self.gesture:
            self.gesture.set_scroll_sensitivity(
                self.current_profile["scroll_speed"]
            )

    # ------------------------------------------------------------------
    # Per-frame processing (called from the main loop)
    # ------------------------------------------------------------------

    def process_commands(self, state: dict) -> dict:
        """
        Must be called from the main thread once per frame.

        Actions
        -------
        1. Poll active window and switch context profile if needed.
        2. Apply scroll boost if active ("scroll up/down fast" voice cmd).
        3. Drain the voice command queue and execute each pending action.
        4. Annotate state with context profile name and voice status for HUD.
        """
        self._poll_context()

        # Apply temporary scroll boost if within the boost window.
        if time.time() < self._scroll_boost_until and self.gesture:
            boosted = self.current_profile["scroll_speed"] * config.VOICE_SCROLL_BOOST_FACTOR
            self.gesture.set_scroll_sensitivity(boosted)

        # Apply continuous 4-second scroll
        if time.time() < self._continuous_scroll[1]:
            direction = self._continuous_scroll[0]
            if direction == "up":
                pyautogui.scroll(15)
            elif direction == "down":
                pyautogui.scroll(-15)

        # Drain voice command queue.
        while not self._cmd_queue.empty():
            try:
                action = self._cmd_queue.get_nowait()
                self._execute(action, state)
            except queue.Empty:
                break

        state["context_profile"] = self.current_profile_name
        state["voice_status"]    = self._last_voice_text
        return state

    # ------------------------------------------------------------------
    # Voice command matching (regex + fuzzy)
    # ------------------------------------------------------------------

    # Regex patterns for dynamic commands (checked before fuzzy dict match)
    _DYNAMIC_PATTERNS = [
        # Web search: "search for python tutorials"
        (r"(?:search(?:\s+for)?|google|look\s+up)\s+(.+)", "search"),
        # Navigate: "go to youtube.com"
        (r"(?:go\s+to|navigate\s+to)\s+(.+)", "navigate"),
        # Open app: "open chrome" (handled separately with fuzzy app match)
        (r"(?:open|launch|start|run)\s+(.+)", "open_app"),
        # Type text: "type hello world"
        (r"^type\s+(.+)$", "type_text"),
        (r"^write\s+(.+)$", "type_text"),
        # Change color: "change color to red"
        (r"(?:change\s+(?:the\s+)?color|set\s+color|color)\s+(?:to\s+)?(.+)", "change_color"),
        # Draw shape: "draw a circle"
        (r"draw\s+(?:a\s+|an\s+)?(.+)", "draw_shape"),
    ]

    def _match_command(self, text: str) -> str | None:
        """
        Match recognized text to a command action string.

        Pipeline:
        1. Check regex patterns for dynamic commands (search, open app, etc.)
        2. Exact substring match against VOICE_COMMANDS dict.
        3. Fuzzy match via fuzzywuzzy (token_sort_ratio + partial_ratio fallback).
        """
        t = text.lower().strip()

        # ---- 1. Dynamic regex patterns --------------------------------
        for pattern, cmd_type in self._DYNAMIC_PATTERNS:
            m = re.search(pattern, t)
            if m:
                payload = m.group(1).strip()
                if cmd_type == "open_app":
                    # Check if it matches a known app; if not fall through
                    app = self._resolve_app(payload)
                    if app:
                        return f"launch_app:{app}"
                    # Not an app → might be "open browser" etc., fall through
                elif cmd_type == "search":
                    return f"web_search:{payload}"
                elif cmd_type == "navigate":
                    return f"web_navigate:{payload}"
                elif cmd_type == "type_text":
                    return f"type_text:{payload}"
                elif cmd_type == "change_color":
                    if payload in config.VOICE_COLORS:
                        return f"canvas_color:{payload}"
                elif cmd_type == "draw_shape":
                    if _HAS_FUZZY:
                        match = fuzz_proc.extractOne(
                            payload, config.VOICE_SHAPES, score_cutoff=65
                        )
                        if match:
                            return f"canvas_shape:{match[0]}"
                    elif payload in config.VOICE_SHAPES:
                        return f"canvas_shape:{payload}"

        # ---- 2. Exact substring match ---------------------------------
        for phrase, action in config.VOICE_COMMANDS.items():
            if phrase in t:
                return action

        # ---- 3. Fuzzy match -------------------------------------------
        if _HAS_FUZZY:
            phrases = list(config.VOICE_COMMANDS.keys())
            threshold = config.VOICE_FUZZY_THRESHOLD

            # Pass 1: token_sort_ratio (best for word-reordering)
            match = fuzz_proc.extractOne(
                t, phrases, scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold
            )
            if match:
                return config.VOICE_COMMANDS[match[0]]

            # Pass 2: partial_ratio fallback (catches substrings)
            for phrase in phrases:
                if fuzz.partial_ratio(phrase, t) >= threshold:
                    return config.VOICE_COMMANDS[phrase]

        return None

    @staticmethod
    def _resolve_app(name: str) -> str | None:
        """Fuzzy-match an app name against APP_ALIASES."""
        name = name.lower().strip()
        if name in config.APP_ALIASES:
            return config.APP_ALIASES[name]
        if _HAS_FUZZY:
            match = fuzz_proc.extractOne(
                name, list(config.APP_ALIASES.keys()),
                scorer=fuzz.token_sort_ratio, score_cutoff=75
            )
            if match:
                return config.APP_ALIASES[match[0]]
        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _execute(self, action: str, state: dict) -> None:
        """
        Execute a single voice command action.

        All pyautogui calls are wrapped in try/except to suppress
        FailSafeException and any unexpected runtime errors.
        """
        print(f"[M4] Executing action: {action}")
        try:
            # ---- Browser / utility -----------------------------------
            if action == "open_browser":
                webbrowser.open("https://google.com")

            elif action == "screenshot":
                path = f"screenshot_{int(time.time())}.png"
                pyautogui.screenshot(path)
                print(f"[M4] Screenshot saved → {path}")

            # ---- Web search / navigate --------------------------------
            elif action.startswith("web_search:"):
                query = action.split(":", 1)[1]
                url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
                webbrowser.open(url)
                print(f"[M4] Searching: {query}")

            elif action.startswith("web_navigate:"):
                site = action.split(":", 1)[1]
                if not site.startswith(("http://", "https://")):
                    site = "https://" + site
                webbrowser.open(site)
                print(f"[M4] Navigating to: {site}")

            # ---- App launch -------------------------------------------
            elif action.startswith("launch_app:"):
                exe = action.split(":", 1)[1]
                try:
                    subprocess.Popen(f'start "" "{exe}"', shell=True)
                except Exception:
                    subprocess.Popen(exe, shell=True, start_new_session=True)
                print(f"[M4] Launching: {exe}")

            # ---- Type text --------------------------------------------
            elif action.startswith("type_text:"):
                text = action.split(":", 1)[1]
                pyautogui.typewrite(text, interval=0.03)
                print(f"[M4] Typed: {text}")

            # ---- Kalman filter toggle --------------------------------
            elif action == "toggle_kalman":
                if self.kalman:
                    enabled = self.kalman.toggle()
                    print(f"[M4] Kalman filter {'ON' if enabled else 'OFF'}")

            # ---- Whiteboard on / off ---------------------------------
            elif action == "whiteboard_on":
                self._switch_mode("WHITEBOARD", state)

            elif action == "whiteboard_off":
                self._switch_mode("MOUSE", state)

            # ---- Scroll speed boost ---------------------------------
            elif action in ("scroll_fast_up", "scroll_fast_down"):
                self._scroll_boost_until = (
                    time.time() + config.VOICE_SCROLL_BOOST_DURATION
                )
                print(f"[M4] Scroll boost active for "
                      f"{config.VOICE_SCROLL_BOOST_DURATION} s")

            # ---- Single scroll commands ------------------------------
            elif action == "scroll_up":
                self._continuous_scroll = ("up", time.time() + 4.0)
            elif action == "scroll_down":
                self._continuous_scroll = ("down", time.time() + 4.0)
            elif action == "page_up":
                pyautogui.scroll(15)
            elif action == "page_down":
                pyautogui.scroll(-15)

            # ---- Volume / brightness ---------------------------------
            elif action == "volume_up":
                for _ in range(3):
                    pyautogui.press("volumeup")
            elif action == "volume_down":
                for _ in range(3):
                    pyautogui.press("volumedown")
            elif action == "mute":
                pyautogui.press("volumemute")
            elif action == "brightness_up":
                try:
                    import screen_brightness_control as sbc
                    cur = sbc.get_brightness(display=0)
                    if isinstance(cur, list): cur = cur[0]
                    sbc.set_brightness(min(100, cur + 10), display=0)
                except Exception:
                    pass
            elif action == "brightness_down":
                try:
                    import screen_brightness_control as sbc
                    cur = sbc.get_brightness(display=0)
                    if isinstance(cur, list): cur = cur[0]
                    sbc.set_brightness(max(0, cur - 10), display=0)
                except Exception:
                    pass

            # ---- Hotkey commands -------------------------------------
            elif action.startswith("hotkey_"):
                key_name = action[len("hotkey_"):]
                keys = config.HOTKEY_MAP.get(key_name)
                if keys:
                    pyautogui.hotkey(*keys)
                    print(f"[M4] Hotkey: {keys}")

            # ---- Mouse clicks ----------------------------------------
            elif action == "mouse_left_click":
                pyautogui.click()
            elif action == "mouse_right_click":
                pyautogui.rightClick()
            elif action == "mouse_double_click":
                pyautogui.doubleClick()

            # ---- Canvas operations -----------------------------------
            elif action == "canvas_clear":
                if self.gesture:
                    self.gesture.clear_canvas()
                    print("[M4] Canvas cleared")
            elif action == "canvas_thick":
                # Signal handled by gesture module if method exists
                print("[M4] Brush: thicker")
            elif action == "canvas_thin":
                print("[M4] Brush: thinner")
            elif action.startswith("canvas_color:"):
                color_name = action.split(":", 1)[1]
                bgr = config.VOICE_COLORS.get(color_name)
                if bgr and self.gesture:
                    self.gesture._draw_color = bgr
                    print(f"[M4] Color → {color_name}")
            elif action.startswith("canvas_shape:"):
                shape = action.split(":", 1)[1]
                if self.gesture and hasattr(self.gesture, "draw_shape"):
                    self.gesture.draw_shape(shape)
                print(f"[M4] Draw shape: {shape}")
            elif action == "canvas_save":
                if self.gesture:
                    import cv2
                    path = f"canvas_{int(time.time())}.png"
                    cv2.imwrite(path, self.gesture._canvas)
                    print(f"[M4] Canvas saved → {path}")

            # ---- Mode switch ----------------------------------------
            elif action.startswith("mode_"):
                new_mode = action[len("mode_"):].upper()
                self._switch_mode(new_mode, state)

        except pyautogui.FailSafeException:
            pass
        except Exception as exc:
            print(f"[M4] Command execution error ({action}): {exc}")

    def _switch_mode(self, new_mode: str, state: dict) -> None:
        """Update mode on self, M3, and the shared state dict."""
        if new_mode not in config.MODES:
            return
        self.mode      = new_mode
        state["mode"]  = new_mode
        if self.gesture:
            self.gesture.set_mode(new_mode)
        print(f"[M4] Mode → {new_mode}")

    # ------------------------------------------------------------------
    # HUD properties (consumed by M5)
    # ------------------------------------------------------------------

    @property
    def profile_display(self) -> str:
        """Profile name formatted for the HUD badge."""
        return self.current_profile_name.upper()

    @property
    def voice_status(self) -> str:
        """Last recognised phrase or status string for the HUD."""
        return self._last_voice_text
