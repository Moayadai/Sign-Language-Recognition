"""
hand_tracker.py
---------------
Modular MediaPipe Hand Tracking and Landmark Normalization Utility.

Provides:
- Hand detection & landmark extraction
- Position and scale invariant feature normalization
- Bounding box calculation for UI visual overlays
"""

import math
from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """
    Wrapper class around MediaPipe Hands for robust landmark detection
    and scale/position invariant feature extraction using Tasks API.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_path: str = "models/hand_landmarker.task"
    ):
        """
        Initialize MediaPipe Hands detector.

        :param static_image_mode: If True, treats input images as static photos (IMAGE mode).
        :param max_num_hands: Maximum number of hands to detect.
        :param min_detection_confidence: Minimum confidence threshold [0.0, 1.0].
        :param min_tracking_confidence: Minimum tracking confidence threshold [0.0, 1.0].
        :param model_path: Path to the downloaded hand_landmarker.task model.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Use IMAGE mode if static_image_mode is True, else VIDEO mode
        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_hand_presence_confidence=min_detection_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.running_mode = running_mode
        self.mp_draw = mp.solutions.drawing_utils if hasattr(mp, "solutions") else None
        self.mp_drawing_styles = mp.solutions.drawing_styles if hasattr(mp, "solutions") else None
        
        # For drawing we need the connections if solutions is not available
        self.HAND_CONNECTIONS = frozenset([
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ])

    def find_hands(self, img: np.ndarray, draw: bool = True, timestamp_ms: int = 0):
        """
        Process BGR image frame and optionally draw hand skeletal landmarks.

        :param img: BGR input image frame from OpenCV.
        :param draw: Whether to draw landmark connections on the image.
        :param timestamp_ms: The timestamp of the current frame in ms (required for video mode).
        :return: (img, results) tuple.
        """
        if img is None or img.size == 0:
            return img, None

        # Convert BGR image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        if self.running_mode == vision.RunningMode.VIDEO:
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            results = self.detector.detect(mp_image)

        if draw and results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Custom basic drawing if mp.solutions isn't available
                if self.mp_draw and self.mp_drawing_styles:
                    from mediapipe.framework.formats import landmark_pb2
                    proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                    proto_landmarks.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                    ])
                    self.mp_draw.draw_landmarks(
                        img,
                        proto_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
                else:
                    h, w, _ = img.shape
                    # Draw connections
                    for connection in self.HAND_CONNECTIONS:
                        pt1 = hand_landmarks[connection[0]]
                        pt2 = hand_landmarks[connection[1]]
                        x1, y1 = int(pt1.x * w), int(pt1.y * h)
                        x2, y2 = int(pt2.x * w), int(pt2.y * h)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw landmarks
                    for lm in hand_landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        return img, results

    def get_landmarks(
        self, img: np.ndarray, results, hand_num: int = 0
    ) -> List[Tuple[int, float, float, float]]:
        """
        Extract normalized landmark coordinates for a specific detected hand.

        :param img: BGR input image frame.
        :param results: MediaPipe process results object.
        :param hand_num: Index of the detected hand (0 for primary hand).
        :return: List of 21 landmark tuples: [(id, x, y, z), ...]
        """
        landmark_list = []
        if (
            results
            and results.hand_landmarks
            and hand_num < len(results.hand_landmarks)
        ):
            hand = results.hand_landmarks[hand_num]
            for lm_id, lm in enumerate(hand):
                landmark_list.append((lm_id, lm.x, lm.y, lm.z))

        return landmark_list

    @staticmethod
    def normalize_landmarks(
        landmarks: List[Tuple[int, float, float, float]]
    ) -> List[float]:
        """
        Normalize 3D landmark coordinates relative to the wrist (landmark 0)
        and scale by maximum distance to make features position & scale invariant.

        :param landmarks: List of (id, x, y, z) tuples for 21 hand points.
        :return: Flattened 1D list of 63 normalized coordinates [x0, y0, z0, ..., x20, y20, z20].
        """
        if not landmarks or len(landmarks) < 21:
            return []

        # Extract wrist coordinate (Landmark 0) as baseline origin
        wrist_x, wrist_y, wrist_z = landmarks[0][1], landmarks[0][2], landmarks[0][3]

        relative_coords = []
        max_dist = 0.0

        # Step 1: Translate origin to wrist & calculate maximum distance
        for _, x, y, z in landmarks:
            rel_x = x - wrist_x
            rel_y = y - wrist_y
            rel_z = z - wrist_z
            relative_coords.append((rel_x, rel_y, rel_z))

            dist = math.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
            if dist > max_dist:
                max_dist = dist

        # Avoid division by zero if hand points collapse
        if max_dist == 0.0:
            max_dist = 1e-6

        # Step 2: Scale coordinates by maximum distance
        normalized_features = []
        for rel_x, rel_y, rel_z in relative_coords:
            normalized_features.extend(
                [rel_x / max_dist, rel_y / max_dist, rel_z / max_dist]
            )

        return normalized_features

    @staticmethod
    def get_bounding_box(
        img: np.ndarray,
        landmarks: List[Tuple[int, float, float, float]],
        padding: int = 20,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate pixel bounding box around detected hand for visual overlay.

        :param img: BGR input image frame.
        :param landmarks: List of (id, x, y, z) tuples.
        :param padding: Pixel padding around hand bounding box.
        :return: (x_min, y_min, x_max, y_max) or None if landmarks are empty.
        """
        if not landmarks or img is None:
            return None

        h, w, _ = img.shape
        x_coords = [int(lm[1] * w) for lm in landmarks]
        y_coords = [int(lm[2] * h) for lm in landmarks]

        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(w, max(x_coords) + padding)
        y_max = min(h, max(y_coords) + padding)

        return x_min, y_min, x_max, y_max

    def close(self):
        """Clean up MediaPipe resources."""
        self.detector.close()
