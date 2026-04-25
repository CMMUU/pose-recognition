import cv2
import numpy as np

from src.emotion_classifier import EmotionPrediction
from src.face_detector import FaceDetection
from src.hand_gesture_recognizer import HAND_CONNECTIONS, HandPrediction


def draw_face_result(
    frame_bgr: np.ndarray,
    face: FaceDetection,
    prediction: EmotionPrediction,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    cv2.rectangle(canvas, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
    label = f"{prediction.label} {prediction.probabilities[prediction.label]:.1%}"
    cv2.putText(
        canvas,
        label,
        (face.x, max(face.y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_hand_result(
    frame_bgr: np.ndarray,
    hand: HandPrediction,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    x, y, w, h = hand.bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 170, 0), 2)

    points = hand.landmarks[:, :2].astype(np.int32)
    for start, end in HAND_CONNECTIONS:
        start_point = tuple(points[start])
        end_point = tuple(points[end])
        cv2.line(canvas, start_point, end_point, (0, 255, 255), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, tuple(point), 3, (0, 0, 255), -1, cv2.LINE_AA)

    label = hand.overlay_text
    cv2.putText(
        canvas,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 170, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
