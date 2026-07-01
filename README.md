# AI-Based Air Writing System

A real-time air-writing application built with **Python, OpenCV, MediaPipe, and NumPy**.
Use your index finger as a virtual pen and write/draw in the air — captured live by your
webcam, smoothed, and rendered to an on-screen canvas.

## Features

- Real-time webcam hand tracking with 21 MediaPipe hand landmarks
- Index-finger-only drawing with moving-average stroke smoothing
- Gesture controls (no keyboard needed for core actions):
  - ☝️ Index finger only → Draw
  - ✌️ Index + middle finger → Selection mode (pause)
  - 🖐️ Open palm (5 fingers) → Clear canvas
  - ✊ Closed fist → Pause drawing
  - 👍 Thumbs up → Save canvas as PNG
- Two live windows: camera feed (with hand skeleton + canvas overlay) and a separate
  clean canvas view
- On-screen HUD: current gesture, brush color, brush size, FPS
- Keyboard shortcuts for color, brush size, undo, save, clear, quit

## Requirements

- Python 3.9–3.12 (MediaPipe wheels are not yet available for very new Python versions —
  check the [MediaPipe PyPI page](https://pypi.org/project/mediapipe/) if installation fails)
- A working webcam
- Windows, macOS, or Linux

## Installation

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Running

### Option 1: Web Application (Viya) - Recommended 🚀

Viya is served locally on localhost and runs entirely in your web browser. It features a modern dark cockpit, hover selection menus, hand-gesture controls, and asynchronous saving.

1. Start the local web server:
   ```bash
   python server.py 3000
   ```
2. Open your web browser and navigate to: [http://localhost:3000](http://localhost:3000)

*Note: You can also access the online deployed version hosted on GitHub Pages: [https://shravnidambe583-create.github.io/Air_Writing/](https://shravnidambe583-create.github.io/Air_Writing/)*

### Option 2: Desktop Python GUI

Run the original OpenCV desktop application:
```bash
python air_writing.py
```
Two windows will open:
1. **Air Writing - Camera Feed** — your webcam with hand landmarks and your drawing overlaid
2. **Air Writing - Canvas** — just the drawing, on a black background

## Controls

### Gestures (hold for a few frames to register — this prevents accidental triggers)

| Gesture | Action |
|---|---|
| Index finger up only | Draw |
| Index + middle finger up | Pause / selection mode |
| All 5 fingers up (open palm) | Clear entire canvas |
| Closed fist | Pause drawing |
| Thumb up, all other fingers curled | Save canvas as PNG (2s cooldown between saves) |

### Keyboard

| Key | Action |
|---|---|
| `R` / `G` / `B` / `Y` / `W` | Set brush color (Red/Green/Blue/Yellow/White) |
| `+` | Increase brush size |
| `-` | Decrease brush size |
| `U` | Undo last stroke |
| `S` | Save drawing as PNG |
| `C` | Clear canvas |
| `Q` | Quit application |

Saved drawings go into the `saved_drawings/` folder (created automatically) with a
timestamped filename, e.g. `drawing_20260630_142210.png`.

## How it works (brief)

1. **Hand detection** — MediaPipe Hands locates 21 landmarks per frame on a single hand.
2. **Gesture classification** — Finger "up/down" state is computed by comparing each
   fingertip's position to its lower joint (and the thumb by horizontal offset, since it
   moves sideways rather than vertically). The resulting finger pattern maps to one of
   five gestures.
3. **Gesture stabilization** — A gesture must persist for a few consecutive frames before
   it's treated as "active," which prevents a single noisy detection frame from triggering
   an accidental clear or save.
4. **Drawing** — While the "draw" gesture is active, the index fingertip position is
   smoothed with a moving average (to reduce landmark jitter) and connected to the
   previous point with an anti-aliased line, drawn directly onto a persistent canvas image.
5. **Undo** — Strokes are stored as point lists; undo removes the last stroke and
   re-renders the canvas from the remaining strokes.

## Known limitations

- Single-hand tracking only (by design, for simplicity and gesture reliability).
- Works best with the hand fully inside the frame and reasonable lighting; MediaPipe's
  accuracy drops in very low light or with a cluttered/skin-toned background.
- "Thumbs up" is intentionally interpreted strictly (thumb extended **and** all other
  fingers curled) to avoid being confused with a closed fist.
- This is a single-hand, single-window-pair simple version — no OCR, no shape recognition,
  no OS-level transparent overlay across other applications.

## Project structure

```
air_writing/
├── air_writing.py      # Main application (all logic)
├── requirements.txt
├── README.md
└── saved_drawings/     # Created automatically on first save
```

## Troubleshooting

- **Webcam doesn't open**: Make sure no other application is using the camera, and that
  your OS has granted camera permission to your terminal/IDE.
- **`ImportError: mediapipe`**: Confirm your Python version is supported by the installed
  MediaPipe wheel (`pip show mediapipe`), and that you're in the correct virtual environment.
- **Low FPS**: Lower `CAM_WIDTH`/`CAM_HEIGHT` at the top of `air_writing.py`, close other
  apps using the camera/GPU, or ensure you're not running inside a heavily virtualized
  environment with no camera passthrough acceleration.
- **Gestures misfiring**: Increase `GESTURE_HOLD_FRAMES` in `air_writing.py` for stricter
  (slower but more reliable) gesture recognition.
