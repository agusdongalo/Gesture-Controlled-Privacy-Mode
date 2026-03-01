import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
GESTURE_BUFFER_SIZE = 8
MODE_SWITCH_COOLDOWN = 0.7  # seconds
OK_DISTANCE_THRESHOLD = 0.05  # normalized distance in landmark space
PIXELATE_WIDTH = 32

MODE_CLEAR = "CLEAR"
MODE_BG_BLUR = "BG BLUR"
MODE_PIXELATE = "PIXELATE"
MODE_FACE_BLUR = "FACE BLUR"

GESTURE_TO_MODE = {
    "OPEN_PALM": MODE_BG_BLUR,
    "PEACE": MODE_PIXELATE,
    "OK": MODE_FACE_BLUR,
    "THUMBS_UP": MODE_CLEAR,
}


# ------------------------------
# Gesture detection
# ------------------------------
def _finger_states(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    if handedness_label == "Right":
        thumb_up = lm[4].x > lm[3].x
    else:
        thumb_up = lm[4].x < lm[3].x

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def detect_gesture(hand_landmarks, handedness_label):
    """
    Detect gesture based on 21 MediaPipe hand landmarks.
    Returns one of: OPEN_PALM, PEACE, OK, THUMBS_UP, or None.
    """
    lm = hand_landmarks.landmark
    thumb_up, index_up, middle_up, ring_up, pinky_up = _finger_states(
        hand_landmarks, handedness_label
    )

    # OK sign: thumb and index tips are close, other fingers mostly extended
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < OK_DISTANCE_THRESHOLD and middle_up and ring_up and pinky_up:
        return "OK"

    # Open palm: all fingers extended
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "OPEN_PALM"

    # Peace sign: index and middle up, ring and pinky down
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # Thumbs up: thumb up, other fingers down
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "THUMBS_UP"

    return None


# ------------------------------
# Filters
# ------------------------------
def apply_background_blur(frame, selfie_segmentation):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb)
    if results.segmentation_mask is None:
        return frame

    mask = results.segmentation_mask
    mask = (mask > 0.1).astype(np.uint8)
    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)

    blurred = cv2.GaussianBlur(frame, (25, 25), 0)
    output = np.where(mask_3c == 1, frame, blurred)
    return output


def apply_pixelate(frame, pixelate_width=PIXELATE_WIDTH):
    h, w = frame.shape[:2]
    target_w = max(1, min(pixelate_width, w))
    target_h = max(1, int(h * (target_w / w)))
    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


def apply_face_blur(frame, face_detection):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    if not results.detections:
        return frame

    h, w = frame.shape[:2]
    output = frame.copy()

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        face_roi = output[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(face_roi, (25, 25), 0)
        output[y1:y2, x1:x2] = blurred

    return output


# ------------------------------
# UI
# ------------------------------
def draw_ui(frame, mode_label, fps):
    color = (0, 255, 0)
    cv2.putText(
        frame,
        f"Mode: {mode_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "✋ BG Blur | ✌️ Pixelate | 👌 Face Blur | 👍 Clear",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_selfie = mp.solutions.selfie_segmentation
    mp_face = mp.solutions.face_detection

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    gesture_history = deque(maxlen=GESTURE_BUFFER_SIZE)
    current_mode = MODE_CLEAR
    last_switch_time = 0.0

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)

            gesture = None
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                handedness_label = (
                    hand_results.multi_handedness[0].classification[0].label
                )
                gesture = detect_gesture(hand_landmarks, handedness_label)

            gesture_history.append(gesture or "NONE")

            counts = Counter(g for g in gesture_history if g != "NONE")
            stable_gesture = None
            if counts:
                stable_gesture = counts.most_common(1)[0][0]

            now = time.time()
            if stable_gesture and stable_gesture in GESTURE_TO_MODE:
                target_mode = GESTURE_TO_MODE[stable_gesture]
                if target_mode != current_mode and (now - last_switch_time) >= MODE_SWITCH_COOLDOWN:
                    current_mode = target_mode
                    last_switch_time = now

            if current_mode == MODE_BG_BLUR:
                output = apply_background_blur(frame, selfie_segmentation)
            elif current_mode == MODE_PIXELATE:
                output = apply_pixelate(frame)
            elif current_mode == MODE_FACE_BLUR:
                output = apply_face_blur(frame, face_detection)
            else:
                output = frame

            # FPS calculation (simple smoothing)
            curr_time = time.time()
            dt = max(1e-6, curr_time - prev_time)
            instant_fps = 1.0 / dt
            fps = fps * 0.9 + instant_fps * 0.1 if fps > 0 else instant_fps
            prev_time = curr_time

            draw_ui(output, current_mode, fps)
            cv2.imshow("Gesture Controlled Privacy Mode", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        selfie_segmentation.close()
        face_detection.close()


if __name__ == "__main__":
    main()
