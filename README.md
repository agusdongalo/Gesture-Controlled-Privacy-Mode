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
.\.venv311\Scripts\python.exe app.py
```

If `.venv311` is missing, re-run the commands above to recreate it.

If `py -3.11` isn't found, run:

```powershell
py -0p
```

## Run

```powershell
cd C:\xampp\htdocs\GestureControlledPrivacyMode
.\.venv311\Scripts\python.exe app.py
```

This opens the webcam window. Press `Q` or `Esc` to exit.

## Test Checklist

UI should show:
- `Mode: CLEAR` at start
- FPS at bottom-left

Gesture tests (hold ~1 second each):
1. Fist (closed palm) -> `CLEAR`
2. Open Palm -> `BG BLUR`
3. Peace Sign -> `PIXELATE`
4. Three Fingers (index, middle, ring) -> `FACE BLUR`
5. Middle Finger -> `HAND PIXEL`
6. Both Fists -> `BG SELECT` (background selection mode)
7. Index Finger -> `SIGN` (sign language mode; stays active until exit)
8. In SIGN mode: Rock-n-roll on either hand -> shows `I LOVE YOU` (only while held)
9. In SIGN mode: Middle Finger -> exits to `CLEAR`

Stability expectations:
- No flicker while holding a gesture
- Majority vote over last 8 frames (no cooldown)

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

## Background Selection (Local Images)

Place your background images in `backgrounds/` inside the project folder (JPG/PNG/BMP).

How it works:
1. Show **both fists** to enter background selection mode.
2. **Open right palm** to move to the next image (close and open again to step).
3. **Open left palm** to move to the previous image.
4. **Both fists for ~2 seconds** to exit selection (the selected background stays applied).

When both fists are shown during selection, the overlay displays `BG SELECT: x/n`.

## Project Structure

- `app.py` - main application entry point

## Notes

MediaPipe `solutions` is not available on Python 3.12+ in the current wheels. This project pins to Python 3.11 with `mediapipe==0.10.11` to keep the `solutions` API.
