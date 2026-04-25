from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_FACE_SIZE = (48, 48)
FACE_SCALE_FACTOR = 1.05
FACE_MIN_NEIGHBORS = 6
FACE_GROUP_EPS = 0.2


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"无法加载人脸检测模型: {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        normalized = cv2.equalizeHist(gray)
        faces = self.classifier.detectMultiScale(
            normalized,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )
        deduplicated = self._group_faces(faces)
        deduplicated.sort(key=lambda face: (face.w * face.h), reverse=True)
        return deduplicated

    @staticmethod
    def _group_faces(faces: np.ndarray) -> list[FaceDetection]:
        if len(faces) == 0:
            return []
        rectangles = [list(map(int, face)) for face in faces]
        grouped_rectangles, _ = cv2.groupRectangles(rectangles + rectangles, groupThreshold=1, eps=FACE_GROUP_EPS)
        if len(grouped_rectangles) == 0:
            grouped_rectangles = np.asarray(rectangles, dtype=np.int32)
        return [FaceDetection(int(x), int(y), int(w), int(h)) for x, y, w, h in grouped_rectangles]

    @staticmethod
    def crop(frame_bgr: np.ndarray, face: FaceDetection) -> np.ndarray:
        return frame_bgr[face.y:face.y + face.h, face.x:face.x + face.w]
