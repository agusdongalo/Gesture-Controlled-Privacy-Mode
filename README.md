# Gesture Controlled Privacy Mode

Real-time webcam app that switches privacy filters based on hand gestures using MediaPipe Hands, Selfie Segmentation, and Face Detection.

## Requirements

- Windows
- Python 3.11
- Webcam

## Setup

Install Python 3.11 if not already installed. The project uses a local virtual environment at `.venv311` (not committed to git).

Create the environment and install dependencies:

```powershell
cd C:\xampp\htdocs\GestureControlledPrivacyMode
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install opencv-python mediapipe==0.10.11 numpy
```

If `.venv311` is missing, re-run the commands above to recreate it.

## Run

```powershell
cd C:\xampp\htdocs\GestureControlledPrivacyMode
.\.venv311\Scripts\python.exe app.py
```

This opens the webcam window. Press `Q` or `Esc` to exit.

## Test Checklist

UI should show:
- `Mode: CLEAR` at start
- Help legend at top
- FPS at bottom-left

Gesture tests (hold ~1 second each):
1. Thumbs Up -> `CLEAR`
2. Open Palm -> `BG BLUR`
3. Peace Sign -> `PIXELATE`
4. OK Sign -> `FACE BLUR`

Stability expectations:
- No flicker while holding a gesture
- 0.7s cooldown between mode switches

## Troubleshooting

If the camera window does not open:
1. Close other apps using the webcam (Zoom, Teams, browser).
2. Check Windows camera privacy settings:
   - Settings -> Privacy & Security -> Camera -> allow access for desktop apps.
3. Quick camera check:

```powershell
.\.venv311\Scripts\python.exe -c "import cv2; cap=cv2.VideoCapture(0); print('opened', cap.isOpened()); ret, frame = cap.read(); print('frame', ret); cap.release()"
```

If `opened False`, try different device indexes (1, 2, etc.) in `app.py` by changing `cv2.VideoCapture(0)`.

## Project Structure

- `app.py` - main application entry point

## Notes

MediaPipe `solutions` is not available on Python 3.12+ in the current wheels. This project pins to Python 3.11 with `mediapipe==0.10.11` to keep the `solutions` API.
