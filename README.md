# Real-Time Sign Language Recognition System 🖐️💬

A modular, real-time computer vision system built with Python, OpenCV, MediaPipe, and Scikit-learn to detect hand gestures from a webcam and classify them as sign language labels in real time.

---

## 📁 Project Architecture

```text
Sign-Language-Recognition/
├── data/                          # Directory where training datasets are saved
│   └── dataset.csv               # Hand landmark features and gesture labels
├── models/                       # Directory for trained model assets
│   ├── hand_landmarker.task      # MediaPipe hand landmark detection model
│   └── sign_language_model.pkl   # Trained Random Forest classifier
├── utils/                        # Helper modules package
│   ├── __init__.py
│   └── hand_tracker.py           # MediaPipe wrapper and feature normalization
├── data_collection.py            # Collects gesture samples from the webcam
├── train_model.py               # Trains the classifier and saves the model
├── app.py                       # Real-time sign recognition and display app
├── requirements.txt             # Python dependencies
├── README.md                   # Project documentation and execution guide
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Workspace Directory
Ensure you are in the project root directory:

```bash
cd Sign-Language-Recognition
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# On Windows (cmd/powershell)
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Collect Gesture Data (`data_collection.py`)
Run the data collection script with a gesture label and sample count:

```bash
python data_collection.py --label <gesture_name> --samples 100
```

Wait for the camera to open, get your hand in position, and press `s` to start automatically capturing the 100 frames.

The dataset is stored in `data/dataset.csv`.

---

### Step 2: Train Classification Model (`train_model.py`)
Train the classifier on the collected landmark dataset:

```bash
python train_model.py
```

#### What this script does:
1. Loads normalized landmark features from `data/dataset.csv`.
2. Validates the dataset and separates labels from feature values.
3. Splits the dataset into 80% training and 20% testing data.
4. Trains a Random Forest classifier and evaluates model accuracy.
5. Saves the trained model to `models/sign_language_model.pkl`.

---

### Step 3: Run Real-Time Translator App (`app.py`)
Launch the real-time sign recognition application:

```bash
python app.py
```

The app opens the webcam, loads the trained model, predicts the current gesture, and displays the recognized text on the screen in real time.

---

## 🧪 Feature Normalization Details

To ensure gesture recognition is robust to hand position and distance from the camera:

1. Origin Translation: The wrist landmark coordinate $(x_0, y_0, z_0)$ is subtracted from all 21 hand landmarks:

$$
x_{rel} = x_i - x_0, \quad y_{rel} = y_i - y_0, \quad z_{rel} = z_i - z_0
$$

2. Scale Normalization: All relative coordinates are scaled by the maximum Euclidean distance from the wrist:

$$
D_{max} = \max_i \sqrt{x_{rel}^2 + y_{rel}^2 + z_{rel}^2}
$$

$$
x_{norm} = \frac{x_{rel}}{D_{max}}, \quad y_{norm} = \frac{y_{rel}}{D_{max}}, \quad z_{norm} = \frac{z_{rel}}{D_{max}}
$$

This produces a 63-dimensional feature vector that is invariant to translation and scale.
