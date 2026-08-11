# Real-Time Sign Language Recognition System 🖐️💬

A modular, real-time Computer Vision system built with **Python**, **OpenCV**, **MediaPipe**, and **Scikit-learn** to track hand gestures via webcam and translate sign language into text on the screen.

---

## 📁 Project Architecture

```text
Ai/
├── data/                       # Directory where CSV datasets are saved
│   └── hand_landmarks.csv      # Extracted hand landmarks & target gesture labels
├── models/                     # Directory for trained classifier models
│   └── sign_language_model.pkl # Trained Random Forest model & label encoder payload
├── utils/                      # Helper modules package
│   ├── __init__.py
│   └── hand_tracker.py         # MediaPipe tracker wrapper & feature normalization
├── data_collection.py          # Interactive webcam landmark collection script
├── train_model.py              # ML classifier training, evaluation & model saving script
├── app.py                      # Real-time webcam sign language translation app
├── requirements.txt            # Python dependencies specification
└── README.md                   # Project documentation & execution guide
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Workspace Directory
Ensure you are in the project root directory:
```bash
cd Sign-Language-Recognition
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# On Windows (cmd/powershell):
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Collect Gesture Data (`data_collection.py`)
Run the data collection script to record hand landmark coordinates for custom signs (e.g., `"Hello"`, `"Thank You"`, `"Yes"`, `"No"`):

```bash
python data_collection.py
```

#### Interactive Controls:
- **`R` or `SPACE`**: Toggle recording **ON/OFF** for the active gesture.
- **`N`**: Switch to **NEXT** gesture label.
- **`P`**: Switch to **PREVIOUS** gesture label.
- **`C`**: Clear dataset and restart collection.
- **`Q` or `ESC`**: Save dataset to `data/hand_landmarks.csv` and exit.

> **Tip:** Record approximately **150 to 200 frames per gesture**, varying hand position, angle, and distance slightly to maximize model robustness.

---

### Step 2: Train Classification Model (`train_model.py`)
Train a lightweight Random Forest Classifier (or MLP Neural Network) on the recorded landmark dataset:

```bash
python train_model.py
```

#### What this script does:
1. Loads normalized features from `data/hand_landmarks.csv`.
2. Validates sample counts and encodes target gesture string labels.
3. Splits dataset into 80% training / 20% testing split.
4. Performs 5-Fold Cross Validation & evaluates test Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
5. Saves the trained model bundle to `models/sign_language_model.pkl`.

---

### Step 3: Run Real-Time Translator App (`app.py`)
Launch the real-time Sign Language Translator webcam application:

```bash
python app.py
```

#### Live Application Features:
- **Position & Scale Invariance:** Normalizes landmark coordinates relative to the wrist and bounding box span, so gestures are recognized anywhere in camera view.
- **Temporal Majority Vote Smoothing:** Uses a 7-frame sliding window filter to eliminate prediction flickering.
- **Interactive Sentence Builder:**
  - Press **`SPACE`** to append the currently recognized sign to your sentence.
  - Press **`BACKSPACE`** to remove the last word.
  - Press **`C`** to clear the entire sentence.
  - Press **`Q`** to quit.

---

## 🧪 Feature Normalization Details

To ensure hand gestures are recognized regardless of where the hand is positioned on screen or how far it is from the webcam:

1. **Origin Translation:** The wrist landmark (Landmark 0) coordinate $(x_0, y_0, z_0)$ is subtracted from all 21 hand landmarks:
   $$x_{rel} = x_i - x_0, \quad y_{rel} = y_i - y_0, \quad z_{rel} = z_i - z_0$$
2. **Scale Normalization:** All relative coordinates are scaled by the maximum Euclidean distance from the wrist:
   $$D_{max} = \max_i \sqrt{x_{rel}^2 + y_{rel}^2 + z_{rel}^2}$$
   $$x_{norm} = \frac{x_{rel}}{D_{max}}, \quad y_{norm} = \frac{y_{rel}}{D_{max}}, \quad z_{norm} = \frac{z_{rel}}{D_{max}}$$

This outputs a **63-element feature vector** invariant to translation and distance.
