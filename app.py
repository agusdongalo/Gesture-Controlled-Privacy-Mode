import time
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import os

# ------------------------------
# Configuration
# ------------------------------
GESTURE_BUFFER_SIZE = 8
MODE_SWITCH_COOLDOWN = 0.0  # seconds
PIXELATE_WIDTH = 32
STABLE_GESTURE_MIN_COUNT = 5  # at least 5 of 8 frames
STABLE_GESTURE_MIN_RATIO = 0.6  # 60% of buffer
FINGER_EXTEND_ANGLE = 160.0
FINGER_FOLD_ANGLE = 95.0
THUMB_EXTEND_ANGLE = 150.0
THUMB_FOLD_ANGLE = 100.0
BACKGROUND_DIR = "backgrounds"
WRIST_CROSS_DISTANCE_THRESHOLD = 0.12
ROCK_BUFFER_SIZE = 5
ROCK_MIN_COUNT = 3
BG_TOGGLE_HOLD_SECONDS = 2.0

MODE_CLEAR = "CLEAR"
MODE_BG_BLUR = "BG BLUR"
MODE_PIXELATE = "PIXELATE"
MODE_FACE_BLUR = "FACE BLUR"
MODE_HAND_PIXELATE = "HAND PIXEL"
MODE_BG_IMAGE = "BG IMAGE"
MODE_SIGN = "SIGN"

GESTURE_TO_MODE = {
    "OPEN_PALM": MODE_BG_BLUR,
    "PEACE": MODE_PIXELATE,
    "THREE_FINGERS": MODE_FACE_BLUR,
    "FIST": MODE_CLEAR,
    "MIDDLE_FINGER": MODE_HAND_PIXELATE,
    "INDEX": MODE_SIGN,
}


# ------------------------------
# Gesture detection
# ------------------------------
def _angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _finger_states(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark

    def v(i):
        return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float32)

    # Angles at PIP joints for fingers
    index_angle = _angle_deg(v(8), v(6), v(5))
    middle_angle = _angle_deg(v(12), v(10), v(9))
    ring_angle = _angle_deg(v(16), v(14), v(13))
    pinky_angle = _angle_deg(v(20), v(18), v(17))

    # Fallback y-rules (tip above PIP means extended for a mirrored webcam view)
    index_y_up = lm[8].y < lm[6].y
    middle_y_up = lm[12].y < lm[10].y
    ring_y_up = lm[16].y < lm[14].y
    pinky_y_up = lm[20].y < lm[18].y

    index_up = index_angle >= FINGER_EXTEND_ANGLE or index_y_up
    middle_up = middle_angle >= FINGER_EXTEND_ANGLE or middle_y_up
    ring_up = ring_angle >= FINGER_EXTEND_ANGLE or ring_y_up
    pinky_up = pinky_angle >= FINGER_EXTEND_ANGLE or pinky_y_up

    index_folded = index_angle <= FINGER_FOLD_ANGLE or not index_y_up
    middle_folded = middle_angle <= FINGER_FOLD_ANGLE or not middle_y_up
    ring_folded = ring_angle <= FINGER_FOLD_ANGLE or not ring_y_up
    pinky_folded = pinky_angle <= FINGER_FOLD_ANGLE or not pinky_y_up

    # Thumb: use angle + handedness x-rule for robustness
    thumb_angle = _angle_deg(v(4), v(3), v(2))
    if handedness_label == "Right":
        thumb_x = lm[4].x > lm[3].x
    else:
        thumb_x = lm[4].x < lm[3].x
    thumb_up = thumb_angle >= THUMB_EXTEND_ANGLE or thumb_x
    thumb_folded = thumb_angle <= THUMB_FOLD_ANGLE and not thumb_x

    metrics = {
        "index_angle": index_angle,
        "middle_angle": middle_angle,
        "ring_angle": ring_angle,
        "pinky_angle": pinky_angle,
        "index_y_up": index_y_up,
        "middle_y_up": middle_y_up,
        "ring_y_up": ring_y_up,
        "pinky_y_up": pinky_y_up,
    }

    return (
        thumb_up,
        index_up,
        middle_up,
        ring_up,
        pinky_up,
        thumb_folded,
        index_folded,
        middle_folded,
        ring_folded,
        pinky_folded,
        metrics,
    )


def detect_gesture(hand_landmarks, handedness_label):
    """
    Detect gesture based on 21 MediaPipe hand landmarks.
    Returns one of: OPEN_PALM, PEACE, THREE_FINGERS, INDEX, ROCK, FIST, or None.
    """
    (
        thumb_up,
        index_up,
        middle_up,
        ring_up,
        pinky_up,
        thumb_folded,
        index_folded,
        middle_folded,
        ring_folded,
        pinky_folded,
        metrics,
    ) = _finger_states(hand_landmarks, handedness_label)

    # Open palm: all fingers extended
    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "OPEN_PALM"

    # Middle finger: middle up, others strictly folded + middle higher than other tips
    index_folded_strict = (
        metrics["index_angle"] <= FINGER_FOLD_ANGLE and not metrics["index_y_up"]
    )
    ring_folded_strict = (
        metrics["ring_angle"] <= FINGER_FOLD_ANGLE and not metrics["ring_y_up"]
    )
    pinky_folded_strict = (
        metrics["pinky_angle"] <= FINGER_FOLD_ANGLE and not metrics["pinky_y_up"]
    )
    middle_is_top = (
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[8].y
        and hand_landmarks.landmark[12].y < hand_landmarks.landmark[16].y
        and hand_landmarks.landmark[12].y < hand_landmarks.landmark[20].y
    )
    if middle_up and index_folded_strict and ring_folded_strict and pinky_folded_strict and middle_is_top:
        return "MIDDLE_FINGER"

    # Peace sign: index and middle strictly up, ring and pinky strictly folded
    index_up_strict = metrics["index_angle"] >= FINGER_EXTEND_ANGLE and metrics["index_y_up"]
    middle_up_strict = metrics["middle_angle"] >= FINGER_EXTEND_ANGLE and metrics["middle_y_up"]
    if index_up_strict and middle_up_strict and ring_folded_strict and pinky_folded_strict:
        return "PEACE"

    # Three fingers: index, middle, ring up; pinky folded (thumb can vary)
    if index_up and middle_up and ring_up and pinky_folded:
        return "THREE_FINGERS"

    # Rock-n-roll: index and pinky up, middle and ring folded (thumb can vary)
    if index_up and pinky_up and middle_folded and ring_folded:
        return "ROCK"

    # Index finger: index up, others folded (thumb can vary)
    if index_up and middle_folded and ring_folded and pinky_folded:
        return "INDEX"

    # Fist (closed palm): all fingers folded (thumb typically folded too)
    if index_folded and middle_folded and ring_folded and pinky_folded:
        return "FIST"

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


def apply_hand_pixelate(frame, hand_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    # Add a small margin
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return frame

    output = frame.copy()
    hand_roi = output[y1:y2, x1:x2]
    roi_h, roi_w = hand_roi.shape[:2]
    if roi_w < 2 or roi_h < 2:
        return output
    small_w = max(4, roi_w // 12)
    small_h = max(4, roi_h // 12)
    small = cv2.resize(hand_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    output[y1:y2, x1:x2] = pixelated
    return output


def apply_background_image(frame, bg_image, selfie_segmentation):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb)
    if results.segmentation_mask is None:
        return frame

    h, w = frame.shape[:2]
    bg_resized = cv2.resize(bg_image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (results.segmentation_mask > 0.1).astype(np.uint8)
    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
    output = np.where(mask_3c == 1, frame, bg_resized)
    return output


def load_backgrounds(folder):
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in exts:
            path = os.path.join(folder, name)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images


# ------------------------------
# UI
# ------------------------------
def draw_ui(frame, mode_label, fps, bg_label=None, message=None):
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

    if bg_label:
        cv2.putText(
            frame,
            bg_label,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if message:
        cv2.putText(
            frame,
            message,
            (int(frame.shape[1] * 0.25), int(frame.shape[0] * 0.6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
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
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=1)
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    gesture_history = deque(maxlen=GESTURE_BUFFER_SIZE)
    current_mode = MODE_CLEAR
    last_switch_time = 0.0
    bg_images = load_backgrounds(BACKGROUND_DIR)
    bg_index = 0
    bg_select_mode = False
    prev_left_open = False
    prev_right_open = False
    prev_both_fists = False
    prev_cross = False
    rock_history = deque(maxlen=ROCK_BUFFER_SIZE)
    both_fists_start = None

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

            hand_data = []
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for lm, handed in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                ):
                    label = handed.classification[0].label
                    gesture = detect_gesture(lm, label)
                    hand_data.append(
                        {"label": label, "landmarks": lm, "gesture": gesture}
                    )

            # Prefer x-position ordering for stability (mirrored webcam)
            left_hand = None
            right_hand = None
            if len(hand_data) == 2:
                h1, h2 = hand_data[0], hand_data[1]
                x1 = h1["landmarks"].landmark[0].x
                x2 = h2["landmarks"].landmark[0].x
                if x1 <= x2:
                    left_hand, right_hand = h1, h2
                else:
                    left_hand, right_hand = h2, h1
            else:
                left_hand = next((h for h in hand_data if h["label"] == "Left"), None)
                right_hand = next((h for h in hand_data if h["label"] == "Right"), None)

            left_gesture = left_hand["gesture"] if left_hand else None
            right_gesture = right_hand["gesture"] if right_hand else None

            left_open = left_gesture == "OPEN_PALM"
            right_open = right_gesture == "OPEN_PALM"
            left_fist = left_gesture == "FIST"
            right_fist = right_gesture == "FIST"
            both_fists = left_fist and right_fist

            cross_arms = False
            if left_hand and right_hand:
                left_lm = left_hand["landmarks"].landmark
                right_lm = right_hand["landmarks"].landmark
                left_wrist = left_lm[0]
                right_wrist = right_lm[0]
                left_center_x = np.mean([left_lm[i].x for i in (0, 5, 9, 13, 17)])
                right_center_x = np.mean([right_lm[i].x for i in (0, 5, 9, 13, 17)])
                dist = np.linalg.norm(
                    np.array([left_wrist.x, left_wrist.y])
                    - np.array([right_wrist.x, right_wrist.y])
                )
                wrists_close = dist < WRIST_CROSS_DISTANCE_THRESHOLD
                cross_arms = wrists_close and (left_wrist.x - right_wrist.x) * (
                    left_center_x - right_center_x
                ) < 0

            if current_mode == MODE_SIGN:
                exit_sign = left_gesture == "MIDDLE_FINGER" or right_gesture == "MIDDLE_FINGER"
                if exit_sign:
                    current_mode = MODE_CLEAR
                    gesture_history.clear()
                    bg_select_mode = False
                    both_fists_start = None
                    prev_both_fists = False
            else:
                if bg_images:
                    if both_fists:
                        if both_fists_start is None:
                            both_fists_start = time.time()
                        elif time.time() - both_fists_start >= BG_TOGGLE_HOLD_SECONDS and not prev_both_fists:
                            bg_select_mode = not bg_select_mode
                            gesture_history.clear()
                            if bg_select_mode:
                                current_mode = MODE_BG_IMAGE
                            prev_both_fists = True
                    else:
                        both_fists_start = None
                        prev_both_fists = False

                    if bg_select_mode:
                        if right_open and not prev_right_open:
                            bg_index = (bg_index + 1) % len(bg_images)
                        if left_open and not prev_left_open:
                            bg_index = (bg_index - 1) % len(bg_images)

            prev_left_open = left_open
            prev_right_open = right_open
            prev_cross = cross_arms

            primary_hand = right_hand or left_hand
            primary_gesture = primary_hand["gesture"] if primary_hand else None
            primary_landmarks = primary_hand["landmarks"] if primary_hand else None

            if not bg_select_mode and current_mode != MODE_SIGN:
                gesture_history.append(primary_gesture or "NONE")

                counts = Counter(gesture_history)
                stable_gesture = None
                if counts:
                    top_gesture, top_count = counts.most_common(1)[0]
                    ratio = top_count / len(gesture_history)
                    if (
                        top_gesture != "NONE"
                        and top_count >= STABLE_GESTURE_MIN_COUNT
                        and ratio >= STABLE_GESTURE_MIN_RATIO
                    ):
                        stable_gesture = top_gesture

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
            elif current_mode == MODE_HAND_PIXELATE and primary_landmarks is not None:
                output = apply_hand_pixelate(frame, primary_landmarks)
            elif current_mode == MODE_BG_IMAGE and bg_images:
                output = apply_background_image(
                    frame, bg_images[bg_index], selfie_segmentation
                )
            else:
                output = frame

            # FPS calculation (simple smoothing)
            curr_time = time.time()
            dt = max(1e-6, curr_time - prev_time)
            instant_fps = 1.0 / dt
            fps = fps * 0.9 + instant_fps * 0.1 if fps > 0 else instant_fps
            prev_time = curr_time

            bg_label = None
            if bg_select_mode and both_fists and bg_images:
                bg_label = f"BG SELECT: {bg_index + 1}/{len(bg_images)}"

            sign_message = None
            if current_mode == MODE_SIGN:
                left_is_rock = left_gesture == "ROCK"
                right_is_rock = right_gesture == "ROCK"
                rock_now = left_is_rock or right_is_rock
                rock_history.append(1 if rock_now else 0)
                if sum(rock_history) >= ROCK_MIN_COUNT:
                    sign_message = "I LOVE YOU"
            else:
                rock_history.clear()

            draw_ui(output, current_mode, fps, bg_label=bg_label, message=sign_message)
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
