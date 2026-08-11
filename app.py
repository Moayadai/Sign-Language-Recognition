"""
app.py
------
Main real-time Gesture Recognition Application.

Opens webcam, extracts hand landmarks using MediaPipe Solutions API,
and translates gestures in real time using a custom pre-trained model.

Key Controls:
- 'SPACE'     : Add current recognized gesture to sentence buffer
- 'BACKSPACE' : Delete last word from sentence buffer
- 'C'         : Clear sentence buffer
- 'Q' or 'ESC': Exit translation app
"""

import os
import sys
import time
from collections import deque, Counter
import cv2
import numpy as np
import joblib

import mediapipe as mp

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

    # 2. Bottom Status Footer
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

    # Select color based on confidence level
    if confidence >= 0.80:
        box_color = (0, 255, 0)      # Green
        bg_color = (0, 150, 0)
    elif confidence >= 0.60:
        box_color = (0, 255, 255)    # Yellow
        bg_color = (0, 150, 150)
    else:
        box_color = (0, 165, 255)    # Orange
        bg_color = (0, 100, 200)

    # Bounding Box around hand
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), box_color, 2)

    # Text Card above bounding box
    text = f"{label} ({confidence * 100:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )

    card_y_min = max(0, y_min - text_h - 18)
    card_y_max = max(text_h + 18, y_min)

    # Draw rounded-corner look rectangle card
    cv2.rectangle(
        img,
        (x_min, card_y_min),
        (x_min + text_w + 20, card_y_max),
        bg_color,
        -1,
    )
    cv2.rectangle(
        img,
        (x_min, card_y_min),
        (x_min + text_w + 20, card_y_max),
        box_color,
        1,
    )

    # Draw label text
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

    mp_hands_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join("models", "hand_landmarker.task")),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=VisionRunningMode.VIDEO)
    
    landmarker = HandLandmarker.create_from_options(options)

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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(curr_time * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            detected_gesture = ""
            detected_confidence = 0.0
            
            bbox = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                
                h, w, _ = frame.shape
                for connection in mp_hands_connections:
                    start_idx, end_idx = connection
                    lm_start = hand_landmarks[start_idx]
                    lm_end = hand_landmarks[end_idx]
                    x_start, y_start = int(lm_start.x * w), int(lm_start.y * h)
                    x_end, y_end = int(lm_end.x * w), int(lm_end.y * h)
                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)
                    
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
                row = []
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                try:
                    features = np.array(row).reshape(1, -1)
                    prediction = model.predict(features)[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)[0]
                        max_proba = np.max(proba)
                        detected_confidence = max_proba
                        detected_gesture = prediction
                    else:
                        detected_gesture = prediction
                        detected_confidence = 1.0
                except Exception as e:
                    print(f"Prediction error: {e}")

                padding = 20
                bbox = (
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    min(w, x_max + padding),
                    min(h, y_max + padding)
                )

            if detected_gesture and detected_confidence >= CONFIDENCE_THRESHOLD:
                prediction_history.append(detected_gesture)
            else:
                prediction_history.append("Unknown")

            if len(prediction_history) > 0:
                most_common, _ = Counter(prediction_history).most_common(1)[0]
                if most_common != "Unknown":
                    current_prediction = most_common
                    if most_common == detected_gesture:
                        current_confidence = float(detected_confidence)
                else:
                    current_prediction = ""
                    current_confidence = 0.0

            if current_prediction and bbox is not None:
                draw_prediction_badge(frame, bbox, current_prediction, current_confidence)

            draw_hud(frame, sentence_buffer, current_prediction, current_confidence, fps)

            cv2.imshow("Sign Language Recognition App", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
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
            elif key in [ord("c"), ord("C")]:
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
