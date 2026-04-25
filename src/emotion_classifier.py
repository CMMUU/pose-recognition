from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src import LABELS


@dataclass
class EmotionPrediction:
    label: str
    probabilities: dict[str, float]


class EmotionClassifier:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, face_bgr: np.ndarray) -> EmotionPrediction:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        input_tensor = resized.astype(np.float32)[None, None, :, :]
        scores = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]
        probabilities = self._softmax(scores)
        label_index = int(np.argmax(probabilities))
        return EmotionPrediction(
            label=LABELS[label_index],
            probabilities={label: float(probabilities[idx]) for idx, label in enumerate(LABELS)},
        )

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exps = np.exp(shifted)
        return exps / np.sum(exps)
