"""
AI-Based Air Writing System Using Hand Gesture Recognition
=============================================================

Draw in the air using your index finger, tracked via webcam + MediaPipe Hands.

Controls
--------
Gestures:
    Index finger only      -> Draw
    Index + middle finger   -> Selection mode (pause, no drawing)
    Open palm (5 fingers)    -> Clear entire canvas
    Closed fist              -> Pause drawing
    Thumbs up                -> Save canvas as PNG

Keyboard:
    R / G / B / Y / W   -> Change brush color
    +  / -               -> Increase / decrease brush size
    U                    -> Undo last stroke
    S                    -> Save drawing
    C                    -> Clear canvas
    Q                    -> Quit

Run:
    python air_writing.py
"""

import time
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CAM_WIDTH, CAM_HEIGHT = 1280, 720
SMOOTHING_WINDOW = 5          # number of points averaged for stroke smoothing
MIN_DRAW_DISTANCE = 2         # px - ignore jitter smaller than this
SAVE_DIR = "saved_drawings"
DEFAULT_BRUSH_SIZE = 6
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 40

COLOR_MAP = {
    "r": (0, 0, 255),     # Red   (BGR)
    "g": (0, 255, 0),     # Green
    "b": (255, 0, 0),     # Blue
    "y": (0, 255, 255),   # Yellow
    "w": (255, 255, 255), # White
}

GESTURE_HOLD_FRAMES = 3  # frames a gesture must persist before it "locks in"
THUMBS_UP_COOLDOWN = 2.0  # seconds between thumbs-up triggered saves


# --------------------------------------------------------------------------
# Hand landmark utilities
# --------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# MediaPipe landmark indices for fingertip and pip (lower) joints
TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def fingers_up(hand_landmarks, handedness_label):
    """
    Return a dict {finger_name: bool} indicating which fingers are extended.

    For the thumb we compare x-coordinates (since it moves sideways, not up/down),
    and flip the comparison depending on whether it's the left or right hand.
    For the other four fingers we compare the tip's y-coordinate against the pip
    joint's y-coordinate (lower y = higher up = extended, since image origin is
    top-left).
    """
    lm = hand_landmarks.landmark
    fingers = {}

    # Thumb: compare horizontal position of tip vs. the joint below it (IP joint).
    thumb_tip_x = lm[TIP_IDS["thumb"]].x
    thumb_ip_x = lm[3].x
    if handedness_label == "Right":
        fingers["thumb"] = thumb_tip_x < thumb_ip_x
    else:
        fingers["thumb"] = thumb_tip_x > thumb_ip_x

    # Other four fingers: tip above pip joint (smaller y) means extended.
    for name in ("index", "middle", "ring", "pinky"):
        tip_y = lm[TIP_IDS[name]].y
        pip_y = lm[PIP_IDS[name]].y
        fingers[name] = tip_y < pip_y

    return fingers


def classify_gesture(fingers):
    """
    Map the fingers_up dict to one of our supported gesture names.
    Order of checks matters: more specific patterns first.
    """
    up_count = sum(fingers.values())
    non_thumb_up = sum(fingers[f] for f in ("index", "middle", "ring", "pinky"))

    # Open palm: all five fingers extended -> clear canvas
    if up_count == 5:
        return "open_palm"

    # Closed fist: nothing extended at all (including thumb) -> pause
    if up_count == 0:
        return "fist"

    # Thumbs up: ONLY thumb extended, all four other fingers curled -> save
    # This check happens before the fist check's "near-zero" cases so a
    # raised thumb with a curled fist doesn't get misread as a plain fist.
    if fingers["thumb"] and non_thumb_up == 0:
        return "thumbs_up"

    # Index + middle only -> selection mode
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "selection"

    # Index only -> draw
    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "draw"

    return "neutral"


# --------------------------------------------------------------------------
# Stroke / canvas management
# --------------------------------------------------------------------------

@dataclass
class Stroke:
    color: tuple
    size: int
    points: list = field(default_factory=list)


class CanvasManager:
    """Holds drawing state: strokes, undo history, and renders to a canvas image."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.strokes = []          # list[Stroke] completed and in-progress
        self.current_stroke = None
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def start_stroke(self, color, size):
        self.current_stroke = Stroke(color=color, size=size)
        self.strokes.append(self.current_stroke)

    def add_point(self, point):
        if self.current_stroke is None:
            return
        self.current_stroke.points.append(point)

    def end_stroke(self):
        self.current_stroke = None

    def undo(self):
        if self.strokes:
            self.strokes.pop()
        self._redraw()

    def clear(self):
        self.strokes = []
        self.current_stroke = None
        self.canvas[:] = 0

    def _redraw(self):
        """Fully re-render the canvas from the stroke list (used after undo)."""
        self.canvas[:] = 0
        for stroke in self.strokes:
            pts = stroke.points
            for i in range(1, len(pts)):
                cv2.line(self.canvas, pts[i - 1], pts[i], stroke.color, stroke.size, cv2.LINE_AA)

    def draw_segment(self, p1, p2, color, size):
        """Incrementally draw a single segment directly onto the persistent canvas."""
        cv2.line(self.canvas, p1, p2, color, size, cv2.LINE_AA)

    def get_canvas(self):
        return self.canvas


class PointSmoother:
    """Moving-average smoother to reduce MediaPipe landmark jitter."""

    def __init__(self, window=SMOOTHING_WINDOW):
        self.window = window
        self.buffer = deque(maxlen=window)

    def smooth(self, point):
        self.buffer.append(point)
        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

    def reset(self):
        self.buffer.clear()


class GestureStabilizer:
    """
    Requires a gesture to persist for N consecutive frames before it's
    considered "active". Prevents a single noisy frame from triggering
    clears/saves/mode switches.
    """

    def __init__(self, hold_frames=GESTURE_HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.last_gesture = None
        self.count = 0
        self.stable_gesture = "neutral"

    def update(self, gesture):
        if gesture == self.last_gesture:
            self.count += 1
        else:
            self.last_gesture = gesture
            self.count = 1

        if self.count >= self.hold_frames:
            self.stable_gesture = gesture

        return self.stable_gesture


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------

class AirWritingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check your camera connection/permissions.")

        # Read actual resolution (camera may not support requested size)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAM_WIDTH
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAM_HEIGHT

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self.canvas_mgr = CanvasManager(self.width, self.height)
        self.smoother = PointSmoother()
        self.gesture_stabilizer = GestureStabilizer()

        self.brush_color = COLOR_MAP["r"]
        self.brush_color_name = "RED"
        self.brush_size = DEFAULT_BRUSH_SIZE

        self.prev_point = None
        self.is_drawing = False
        self.last_save_time = 0.0
        self.status_message = ""
        self.status_message_until = 0.0

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.fps_history = deque(maxlen=30)
        self.prev_time = time.time()

    # ---- helpers ----------------------------------------------------

    def set_status(self, msg, duration=1.5):
        self.status_message = msg
        self.status_message_until = time.time() + duration

    def save_canvas(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"drawing_{timestamp}.png")
        cv2.imwrite(filename, self.canvas_mgr.get_canvas())
        self.set_status(f"Saved: {filename}")
        return filename

    def handle_gesture(self, gesture, fingertip):
        """Apply the stable gesture to drawing state. Returns nothing."""
        if gesture == "draw":
            if not self.is_drawing:
                # Just started drawing - begin a new stroke
                self.is_drawing = True
                self.canvas_mgr.start_stroke(self.brush_color, self.brush_size)
                self.smoother.reset()
                self.prev_point = None

            smoothed = self.smoother.smooth(fingertip)
            self.canvas_mgr.add_point(smoothed)

            if self.prev_point is not None:
                dist = np.hypot(smoothed[0] - self.prev_point[0], smoothed[1] - self.prev_point[1])
                if dist >= MIN_DRAW_DISTANCE:
                    self.canvas_mgr.draw_segment(self.prev_point, smoothed, self.brush_color, self.brush_size)
                    self.prev_point = smoothed
            else:
                self.prev_point = smoothed

        else:
            # Any non-draw gesture ends the current stroke
            if self.is_drawing:
                self.canvas_mgr.end_stroke()
                self.is_drawing = False
                self.prev_point = None
                self.smoother.reset()

            if gesture == "open_palm":
                self.canvas_mgr.clear()
                self.set_status("Canvas Cleared")

            elif gesture == "thumbs_up":
                now = time.time()
                if now - self.last_save_time > THUMBS_UP_COOLDOWN:
                    self.save_canvas()
                    self.last_save_time = now

            # "fist" and "selection" just pause drawing; nothing else to do

    def draw_hud(self, frame, gesture):
        """Overlay status text: gesture, color, brush size, FPS, instructions."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
        frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        gesture_labels = {
            "draw": "DRAWING",
            "selection": "SELECTION MODE",
            "open_palm": "CLEARING...",
            "fist": "PAUSED (Fist)",
            "thumbs_up": "SAVING...",
            "neutral": "IDLE",
        }
        gesture_text = gesture_labels.get(gesture, gesture.upper())

        cv2.putText(frame, f"Gesture: {gesture_text}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Brush: {self.brush_color_name}  Size: {self.brush_size}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Color swatch
        cv2.rectangle(frame, (280, 35), (310, 60), self.brush_color, -1)
        cv2.rectangle(frame, (280, 35), (310, 60), (255, 255, 255), 1)

        # FPS
        fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "R/G/B/Y/W color | +/- size | U undo | S save | C clear | Q quit",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # Transient status message (e.g. "Canvas Cleared", "Saved: ...")
        if self.status_message and time.time() < self.status_message_until:
            text_size = cv2.getTextSize(self.status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x = (w - text_size[0]) // 2
            cv2.putText(frame, self.status_message, (x, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    def handle_key(self, key):
        if key == ord('q'):
            return False
        elif key == ord('s'):
            self.save_canvas()
        elif key == ord('c'):
            self.canvas_mgr.clear()
            self.set_status("Canvas Cleared")
        elif key == ord('u'):
            self.canvas_mgr.undo()
            self.set_status("Undo")
        elif key in (ord('+'), ord('=')):
            self.brush_size = min(MAX_BRUSH_SIZE, self.brush_size + 2)
        elif key == ord('-'):
            self.brush_size = max(MIN_BRUSH_SIZE, self.brush_size - 2)
        elif chr(key) in COLOR_MAP if 0 <= key < 256 else False:
            name = chr(key)
            self.brush_color = COLOR_MAP[name]
            self.brush_color_name = {
                "r": "RED", "g": "GREEN", "b": "BLUE", "y": "YELLOW", "w": "WHITE"
            }[name]
        return True

    # ---- main loop ----------------------------------------------------

    def run(self):
        print("Air Writing System started. Press 'Q' in the window to quit.")
        running = True

        while running:
            success, frame = self.cap.read()
            if not success:
                print("Warning: failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            frame = cv2.resize(frame, (self.width, self.height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)

            gesture = "neutral"
            fingertip = None

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                finger_state = fingers_up(hand_landmarks, handedness_label)
                raw_gesture = classify_gesture(finger_state)
                gesture = self.gesture_stabilizer.update(raw_gesture)

                index_tip = hand_landmarks.landmark[TIP_IDS["index"]]
                fingertip = (int(index_tip.x * self.width), int(index_tip.y * self.height))

                # Visual cue: highlight fingertip
                cv2.circle(frame, fingertip, 8, (0, 255, 255), cv2.FILLED)

                self.handle_gesture(gesture, fingertip)
            else:
                # No hand detected -> treat as neutral, end any active stroke
                gesture = self.gesture_stabilizer.update("neutral")
                if self.is_drawing:
                    self.canvas_mgr.end_stroke()
                    self.is_drawing = False
                    self.prev_point = None

            # Composite: blend canvas onto camera feed preview, and show canvas separately
            canvas = self.canvas_mgr.get_canvas()
            blended = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

            self.draw_hud(blended, gesture)

            # FPS calculation
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps_history.append(1.0 / dt)

            cv2.imshow("Air Writing - Camera Feed", blended)
            cv2.imshow("Air Writing - Canvas", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # a key was actually pressed
                running = self.handle_key(key)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Application closed.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    app = AirWritingApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
