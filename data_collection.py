
"""
data_collection.py
---------------
Collect hand gesture data for training.
"""
import os
import time
import argparse
import urllib.request
import cv2
import numpy as np
# Hand tracking utility
from utils.hand_tracker import HandTracker

def main():
    parser = argparse.ArgumentParser(description="Collect hand gesture data.")
    parser.add_argument("--label", type=str, required=True, help="Label for the gesture being collected.")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to collect.")
    args = parser.parse_args()
    label = args.label
    num_samples = args.samples
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        header = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")
        with open(csv_path, "w") as f:
            f.write(",".join(header) + "\n")
    model_path = os.path.join("models", "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
    hand_tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_path=model_path,
    )
    cap = cv2.VideoCapture(0)
    print(f"Collecting data for '{label}'. Press 's' to start collecting...")
    collecting = False
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)
        frame, results = hand_tracker.find_hands(frame, draw=True, timestamp_ms=timestamp_ms)
        landmarks = hand_tracker.get_landmarks(frame, results)
        if landmarks and collecting:
            norm_feat = HandTracker.normalize_landmarks(landmarks)
            if len(norm_feat) == 63:
                row = norm_feat + [label]
                with open(csv_path, "a") as f:
                    f.write(",".join(map(str, row)) + "\n")
                count += 1
                print(f"Collected {count}/{num_samples} for '{label}'")
                if count >= num_samples:
                    print("Done collecting.")
                    collecting = False
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {count}/{num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not collecting:
            cv2.putText(frame, "Press 's' to start", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Collecting...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            collecting = True
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
