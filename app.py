
"""
app.py
------
Main real-time Gesture Recognition Application.

Opens webcam, extracts hand landmarks using MediaPipe Solutions API,
and translates gestures in real time using a custom pre-trained model.

Key Controls:
- "SPACE"     : Add current recognized gesture to sentence buffer
- "BACKSPACE" : Delete last word from sentence buffer
- "C"         : Clear sentence buffer
- "Q" or "ESC": Exit translation app
"""

import os
import sys
import time
from collections import deque, Counter
import cv2
import numpy as np
import joblib

# Hand tracking utility
from utils.hand_tracker import HandTracker

# Configuration & Constants
MODEL_PATH = os.path.join("models", "sign_language_model.pkl")
CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW_SIZE = 7


def draw_hud(
    img: np.ndarray,
    sentence: list,
    current_prediction: str,
    confidence: float,
    fps: float,
):
    h, w, _ = img.shape
    # 1. Top Header Banner (Dark Translucent Bar)
    header_height = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    # Title
    cv2.putText(
        img,
        "REAL-TIME GESTURE RECOGNITION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # Render Active Sentence
    sentence_str = " ".join(sentence) if sentence else "Press SPACE to append gestures..."
    cv2.putText(
        img,
        f"Sentence: {sentence_str}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # Bottom Status Footer
    footer_height = 40
    cv2.rectangle(img, (0, h - footer_height), (w, h), (20, 20, 20), -1)
    status_text = (
        f"FPS: {fps:.1f} | "
        f"Detected: {current_prediction.upper() if current_prediction else 'None'} "
        f"({confidence*100:.1f}%) | "
        f"Controls: [Space] Add Word | [Backspace] Delete | [C] Clear | [Q] Quit"
    )
    cv2.putText(
        img,
        status_text,
        (15, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def draw_prediction_badge(
    img: np.ndarray, bbox: tuple, label: str, confidence: float
):
    x_min, y_min, x_max, y_max = bbox
    if confidence >= 0.80:
        box_color = (0, 255, 0)
        bg_color = (0, 150, 0)
    elif confidence >= 0.60:
        box_color = (0, 255, 255)
        bg_color = (0, 150, 150)
    else:
        box_color = (0, 165, 255)
        bg_color = (0, 100, 200)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), box_color, 2)
    text = f"{label} ({confidence * 100:.1f}%)"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    card_y_min = max(0, y_min - text_h - 18)
    card_y_max = max(text_h + 18, y_min)
    cv2.rectangle(img, (x_min, card_y_min), (x_min + text_w + 20, card_y_max), bg_color, -1)
    cv2.rectangle(img, (x_min, card_y_min), (x_min + text_w + 20, card_y_max), box_color, 1)
    cv2.putText(
        img,
        text,
        (x_min + 10, card_y_max - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    print("=" * 60)
    print("        REAL-TIME GESTURE RECOGNITION APP")
    print("=" * 60)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Custom model not found at: '{MODEL_PATH}'")
        print("Please run train_model.py to train your sign language model first.")
        sys.exit(1)
    print("Loading custom trained model...")
    model = joblib.load(MODEL_PATH)
    # Initialise HandTracker
    hand_tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_path=os.path.join("models", "hand_landmarker.task"),
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam. Ensure your camera is connected.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    sentence_buffer = []
    prev_time = time.time()
    current_prediction = ""
    current_confidence = 0.0
    print("\n[READY] Recognition window active.")
    print("Show gestures in front of the camera.\n")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame from camera.")
                continue
            frame = cv2.flip(frame, 1)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            timestamp_ms = int(curr_time * 1000)
            frame, results = hand_tracker.find_hands(frame, draw=False, timestamp_ms=timestamp_ms)
            landmarks = hand_tracker.get_landmarks(frame, results)
            detected_gesture = ""
            detected_confidence = 0.0
            bbox = None
            if landmarks:
                norm_feat = HandTracker.normalize_landmarks(landmarks)
                if len(norm_feat) == 63:
                    features = np.array(norm_feat).reshape(1, -1)
                    try:
                        pred = model.predict(features)[0]
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(features)[0]
                            max_proba = np.max(proba)
                            detected_confidence = max_proba
                            detected_gesture = pred
                        else:
                            detected_gesture = pred
                            detected_confidence = 1.0
                    except Exception as e:
                        print(f"Prediction error: {e}")
                bbox = HandTracker.get_bounding_box(frame, landmarks)
            if detected_gesture and detected_confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(detected_gesture)
            else:
                prediction_history.append("Unknown")
            if prediction_history:
                most_common, _ = Counter(prediction_history).most_common(1)[0]
                if most_common != "Unknown":
                    current_prediction = most_common
                    if most_common == detected_gesture:
                        current_confidence = float(detected_confidence)
                else:
                    current_prediction = ""
                    current_confidence = 0.0
            if current_prediction and bbox:
                draw_prediction_badge(frame, bbox, current_prediction, current_confidence)
            draw_hud(frame, sentence_buffer, current_prediction, current_confidence, fps)
            cv2.imshow("Sign Language Recognition App", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] Exiting App...")
                break
            elif key == 32:
                if current_prediction:
                    sentence_buffer.append(current_prediction)
                    print(f"[SENTENCE] Added word: '{current_prediction}' | Full: {' '.join(sentence_buffer)}")
            elif key in [8, 127]:
                if sentence_buffer:
                    removed = sentence_buffer.pop()
                    print(f"[SENTENCE] Removed word: '{removed}'")
            elif key in [ord('c'), ord('C')]:
                sentence_buffer.clear()
                print("[SENTENCE] Sentence cleared.")
    except KeyboardInterrupt:
        print("[INFO] Application stopped by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected runtime exception: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[COMPLETE] App shut down cleanly.")

if __name__ == "__main__":
    main()
