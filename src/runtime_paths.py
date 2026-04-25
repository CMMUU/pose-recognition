from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
MODELS_DIR = BASE_DIR / "models"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"
HAND_MODEL_DIR = MODELS_DIR / "mediapipe_hand_gesture-onnx-float"
BUNDLED_HAARCASCADE_PATH = MODELS_DIR / "haarcascades" / "haarcascade_frontalface_default.xml"
