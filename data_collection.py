import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import argparse
import urllib.request

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

    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

    cap = cv2.VideoCapture(0)
    print(f"Collecting data for '{label}'. Press 's' to start collecting...")

    collecting = False
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    idx1, idx2 = connection
                    if idx1 < len(hand_landmarks) and idx2 < len(hand_landmarks):
                        lm1, lm2 = hand_landmarks[idx1], hand_landmarks[idx2]
                        x1, y1 = int(lm1.x * frame.shape[1]), int(lm1.y * frame.shape[0])
                        x2, y2 = int(lm2.x * frame.shape[1]), int(lm2.y * frame.shape[0])
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                
                if collecting:
                    row = []
                    for lm in hand_landmarks:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label)
                    
                    with open(csv_path, "a") as f:
                        f.write(",".join(map(str, row)) + "\n")
                    
                    count += 1
                    print(f"Collected {count}/{num_samples} for '{label}'")
                    
                    if count >= num_samples:
                        print("Done collecting.")
                        collecting = False
                        break
        
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
