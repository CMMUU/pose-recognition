from dataclasses import dataclass

import cv2
import numpy as np

from src.draw import draw_face_result, draw_hand_result
from src.emotion_classifier import EmotionClassifier, EmotionPrediction
from src.face_detector import FaceDetection, FaceDetector
from src.hand_gesture_recognizer import HandGestureRecognizer, HandPrediction


@dataclass
class ImageAnalysisResult:
    annotated_frame: np.ndarray
    face_predictions: list[tuple[FaceDetection, EmotionPrediction]]
    hand_predictions: list[HandPrediction]
    hand_message: str | None = None


class ImagePipeline:
    def __init__(
        self,
        detector: FaceDetector,
        classifier: EmotionClassifier,
        hand_recognizer: HandGestureRecognizer | None = None,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.hand_recognizer = hand_recognizer

    def analyze(self, frame_bgr: np.ndarray, enable_dynamic_actions: bool = True) -> ImageAnalysisResult:
        face_predictions: list[tuple[FaceDetection, EmotionPrediction]] = []
        annotated = frame_bgr.copy()
        faces = self.detector.detect(frame_bgr)
        for face in faces:
            face_image = self.detector.crop(frame_bgr, face)
            prediction = self.classifier.predict(face_image)
            annotated = draw_face_result(annotated, face, prediction)
            face_predictions.append((face, prediction))

        hand_predictions: list[HandPrediction] = []
        hand_message: str | None = None
        if self.hand_recognizer is not None:
            hand_predictions = self.hand_recognizer.detect(
                frame_bgr,
                faces=faces,
                enable_dynamic_actions=enable_dynamic_actions,
            )
            for hand_prediction in hand_predictions:
                annotated = draw_hand_result(annotated, hand_prediction)

        return ImageAnalysisResult(
            annotated_frame=annotated,
            face_predictions=face_predictions,
            hand_predictions=hand_predictions,
            hand_message=hand_message,
        )

    def analyze_file(self, file_path: str) -> ImageAnalysisResult:
        frame = cv2.imread(file_path)
        if frame is None:
            raise RuntimeError("无法读取图片 / Failed to read image")

        result = self.analyze(frame, enable_dynamic_actions=False)
        if self.hand_recognizer is not None and result.hand_predictions:
            result.hand_message = (
                "图片模式下仅展示当前手势，动态动作需要摄像头连续帧。\n"
                "In image mode, only the current hand gesture is shown. Dynamic actions require continuous camera frames."
            )
        return result
