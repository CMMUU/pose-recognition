from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import onnxruntime as ort

from src.face_detector import FaceDetection

PALM_INPUT_SIZE = (256, 256)
LANDMARK_INPUT_SIZE = (224, 224)
MIN_DETECTOR_SCORE = 0.75
MIN_LANDMARK_SCORE = 0.2
NMS_IOU_THRESHOLD = 0.3
DETECT_DXY = 0.5
DETECT_DSCALE = 2.5
WRIST_CENTER_KEYPOINT_INDEX = 0
MIDDLE_FINGER_KEYPOINT_INDEX = 2
ROTATION_VECTOR_OFFSET_RADS = np.pi / 2
HISTORY_SIZE = 24
MISS_RESET_FRAMES = 6

HAND_MODEL_FILES = {
    "palm_detector": "palm_detector.onnx",
    "hand_landmark_detector": "hand_landmark_detector.onnx",
    "gesture_classifier": "canned_gesture_classifier.onnx",
}

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
]

GESTURE_LABELS = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
]

GESTURE_TEXT = {
    "None": ("未识别手势", "Unknown Gesture"),
    "Closed_Fist": ("握拳", "Closed Fist"),
    "Open_Palm": ("张开手掌", "Open Palm"),
    "Pointing_Up": ("向上指", "Pointing Up"),
    "Thumb_Down": ("倒拇指", "Thumb Down"),
    "Thumb_Up": ("点赞", "Thumb Up"),
    "Victory": ("V 手势", "Victory"),
    "ILoveYou": ("爱你手势", "I Love You"),
}

DYNAMIC_ACTION_TEXT = {
    "waving": ("挥手", "Waving"),
    "hand_raised": ("举手", "Hand Raised"),
    "open_close_transition": ("张开/握拳切换", "Open-Close Transition"),
}


@dataclass
class HandPrediction:
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    is_right_hand: bool
    static_gesture: str
    static_confidence: float
    dynamic_action: str | None = None

    @property
    def static_text(self) -> tuple[str, str]:
        return GESTURE_TEXT.get(self.static_gesture, (self.static_gesture, self.static_gesture))

    @property
    def dynamic_text(self) -> tuple[str, str] | None:
        if self.dynamic_action is None:
            return None
        return DYNAMIC_ACTION_TEXT.get(self.dynamic_action)

    @property
    def handedness_text(self) -> tuple[str, str]:
        return ("右手", "Right Hand") if self.is_right_hand else ("左手", "Left Hand")

    @property
    def overlay_text(self) -> str:
        if self.dynamic_action is not None:
            zh, en = self.dynamic_text or (self.dynamic_action, self.dynamic_action)
            return f"{zh} / {en}"
        zh, en = self.static_text
        return f"{zh} / {en}"


@dataclass
class _HandHistorySample:
    frame_index: int
    center_x: float
    center_y: float
    static_gesture: str


class HandGestureRecognizer:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._validate_model_files()

        self.palm_session = ort.InferenceSession(
            str(self.model_dir / HAND_MODEL_FILES["palm_detector"]),
            providers=["CPUExecutionProvider"],
        )
        self.landmark_session = ort.InferenceSession(
            str(self.model_dir / HAND_MODEL_FILES["hand_landmark_detector"]),
            providers=["CPUExecutionProvider"],
        )
        self.gesture_session = ort.InferenceSession(
            str(self.model_dir / HAND_MODEL_FILES["gesture_classifier"]),
            providers=["CPUExecutionProvider"],
        )

        self.palm_input_name = self.palm_session.get_inputs()[0].name
        self.palm_output_names = [output.name for output in self.palm_session.get_outputs()]
        self.landmark_input_name = self.landmark_session.get_inputs()[0].name
        self.landmark_output_names = [output.name for output in self.landmark_session.get_outputs()]
        self.gesture_input_names = [input_meta.name for input_meta in self.gesture_session.get_inputs()]
        self.gesture_output_name = self.gesture_session.get_outputs()[0].name

        self.frame_index = 0
        self.frames_without_hand = 0
        self.history: Deque[_HandHistorySample] = deque(maxlen=HISTORY_SIZE)
        self.active_dynamic_action: tuple[str, int] | None = None

    def _validate_model_files(self) -> None:
        missing = [name for name in HAND_MODEL_FILES.values() if not (self.model_dir / name).exists()]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"手部模型文件缺失 / Missing hand model files: {joined}")

    def detect(
        self,
        frame_bgr: np.ndarray,
        faces: list[FaceDetection] | None = None,
        enable_dynamic_actions: bool = True,
    ) -> list[HandPrediction]:
        self.frame_index += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        palm_input, scale, pad = self._resize_and_pad(frame_rgb, PALM_INPUT_SIZE)
        coords, scores = self.palm_session.run(
            self.palm_output_names,
            {self.palm_input_name: self._to_nchw_float32(palm_input)},
        )
        coords = np.asarray(coords)[0]
        scores = np.asarray(scores)[0].reshape(-1)

        selected_indices = self._nms(coords[:, :4], scores, MIN_DETECTOR_SCORE, NMS_IOU_THRESHOLD)
        predictions: list[HandPrediction] = []

        for index in selected_indices[:2]:
            box_and_points = coords[index].reshape(-1, 2)
            box_and_points = self._restore_coordinates(box_and_points, scale, pad)
            box_xyxy = box_and_points[:2]
            keypoints = box_and_points[2:]
            roi_corners = self._compute_hand_roi(box_xyxy, keypoints)
            crop = self._crop_roi(frame_rgb, roi_corners, LANDMARK_INPUT_SIZE)
            landmark_outputs = self.landmark_session.run(
                self.landmark_output_names,
                {self.landmark_input_name: self._to_nchw_float32(crop)},
            )
            landmarks = np.asarray(landmark_outputs[0]).reshape(21, 3)
            landmark_score = float(np.asarray(landmark_outputs[1]).reshape(-1)[0])
            handedness = float(np.asarray(landmark_outputs[2]).reshape(-1)[0])
            if landmark_score < MIN_LANDMARK_SCORE:
                continue

            mapped_landmarks = self._map_landmarks_to_frame(landmarks, roi_corners, LANDMARK_INPUT_SIZE)
            gesture_scores = self._run_gesture_classifier(mapped_landmarks, handedness)
            gesture_probs = self._softmax(gesture_scores)
            gesture_index = int(np.argmax(gesture_probs))
            gesture_label = GESTURE_LABELS[gesture_index]
            bbox = self._landmark_bbox(mapped_landmarks[:, :2], frame_bgr.shape[1], frame_bgr.shape[0])
            predictions.append(
                HandPrediction(
                    bbox=bbox,
                    landmarks=mapped_landmarks,
                    is_right_hand=round(handedness) == 1,
                    static_gesture=gesture_label,
                    static_confidence=float(gesture_probs[gesture_index]),
                )
            )

        predictions.sort(key=lambda item: (item.bbox[2] * item.bbox[3]), reverse=True)

        if not predictions:
            self.frames_without_hand += 1
            if self.frames_without_hand >= MISS_RESET_FRAMES:
                self.history.clear()
                self.active_dynamic_action = None
            return []

        self.frames_without_hand = 0
        if enable_dynamic_actions:
            action = self._update_dynamic_action(predictions[0], frame_bgr.shape[:2], faces or [])
            if action is not None:
                predictions[0].dynamic_action = action
        else:
            self.history.clear()
            self.active_dynamic_action = None

        return predictions

    def _resize_and_pad(
        self,
        image_rgb: np.ndarray,
        dst_size: tuple[int, int],
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        dst_h, dst_w = dst_size
        src_h, src_w = image_rgb.shape[:2]
        scale = min(dst_h / src_h, dst_w / src_w)
        new_h = max(1, int(np.floor(src_h * scale)))
        new_w = max(1, int(np.floor(src_w * scale)))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        pad_left = (dst_w - new_w) // 2
        pad_top = (dst_h - new_h) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, scale, (pad_left, pad_top)

    @staticmethod
    def _to_nchw_float32(image_rgb: np.ndarray) -> np.ndarray:
        return image_rgb.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0

    @staticmethod
    def _restore_coordinates(
        coordinates: np.ndarray,
        scale: float,
        pad: tuple[int, int],
    ) -> np.ndarray:
        restored = coordinates.copy().astype(np.float32)
        restored[..., 0] = (restored[..., 0] - pad[0]) / scale
        restored[..., 1] = (restored[..., 1] - pad[1]) / scale
        return restored

    @staticmethod
    def _nms(
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        score_threshold: float,
        iou_threshold: float,
    ) -> list[int]:
        candidates = np.where(scores >= score_threshold)[0]
        if len(candidates) == 0:
            return []
        order = candidates[np.argsort(scores[candidates])[::-1]]
        keep: list[int] = []
        while len(order) > 0:
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            current_box = boxes_xyxy[current]
            remaining = boxes_xyxy[order[1:]]
            ious = HandGestureRecognizer._bbox_iou(current_box, remaining)
            order = order[1:][ious < iou_threshold]
        return keep

    @staticmethod
    def _bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
        area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    @staticmethod
    def _compute_hand_roi(box_xyxy: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        center = (box_xyxy[0] + box_xyxy[1]) * 0.5
        size = box_xyxy[1] - box_xyxy[0]
        wrist = keypoints[WRIST_CENTER_KEYPOINT_INDEX]
        middle = keypoints[MIDDLE_FINGER_KEYPOINT_INDEX]
        vector = middle - wrist
        length = np.linalg.norm(vector) + 1e-6
        center = center + DETECT_DXY * size[0] * (vector / length)
        width = size[0] * DETECT_DSCALE
        height = size[1] * DETECT_DSCALE
        theta = np.arctan2(wrist[1] - middle[1], wrist[0] - middle[0]) - ROTATION_VECTOR_OFFSET_RADS
        base = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        base[:, 0] *= width / 2
        base[:, 1] *= height / 2
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )
        corners = base @ rotation.T
        corners[:, 0] += center[0]
        corners[:, 1] += center[1]
        return corners.astype(np.float32)

    @staticmethod
    def _crop_roi(
        frame_rgb: np.ndarray,
        roi_corners: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        output_h, output_w = output_size
        src = roi_corners[:3].astype(np.float32)
        dst = np.array(
            [[0, 0], [0, output_h - 1], [output_w - 1, 0]],
            dtype=np.float32,
        )
        affine = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(frame_rgb, affine, (output_w, output_h))

    @staticmethod
    def _map_landmarks_to_frame(
        landmarks: np.ndarray,
        roi_corners: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        output_h, output_w = output_size
        src = roi_corners[:3].astype(np.float32)
        dst = np.array(
            [[0, 0], [0, output_h - 1], [output_w - 1, 0]],
            dtype=np.float32,
        )
        affine = cv2.getAffineTransform(src, dst)
        inverse_affine = cv2.invertAffineTransform(affine)
        xy = landmarks[:, :2]
        homogeneous = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)
        mapped_xy = homogeneous @ inverse_affine.T
        mapped = landmarks.copy()
        mapped[:, :2] = mapped_xy
        return mapped

    def _run_gesture_classifier(self, landmarks: np.ndarray, handedness: float) -> np.ndarray:
        hand = self._preprocess_hand_x64(landmarks, handedness, mirror=False)
        mirrored_hand = self._preprocess_hand_x64(landmarks, handedness, mirror=True)
        output = self.gesture_session.run(
            [self.gesture_output_name],
            {
                self.gesture_input_names[0]: hand[None, ...].astype(np.float32),
                self.gesture_input_names[1]: mirrored_hand[None, ...].astype(np.float32),
            },
        )[0][0]
        return np.asarray(output, dtype=np.float32)

    @staticmethod
    def _preprocess_hand_x64(
        landmarks: np.ndarray,
        handedness: float,
        mirror: bool,
    ) -> np.ndarray:
        points = landmarks.copy().astype(np.float32)
        handedness_value = 1.0 if handedness >= 0.5 else 0.0
        if mirror:
            points[:, 0] *= -1.0
            handedness_value = 1.0 - handedness_value
        center_indices = np.array([0, 1, 5, 9, 13, 17], dtype=np.int32)
        center = points[center_indices].mean(axis=0, keepdims=True)
        normalized = points - center
        range_x = normalized[:, 0].max() - normalized[:, 0].min()
        range_y = normalized[:, 1].max() - normalized[:, 1].min()
        scale = max(range_x, range_y, 1e-5)
        flattened = (normalized / scale).reshape(-1)
        return np.concatenate([flattened, np.array([handedness_value], dtype=np.float32)])

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    @staticmethod
    def _landmark_bbox(landmarks_xy: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x_min = max(0, int(np.floor(np.min(landmarks_xy[:, 0])) - 20))
        y_min = max(0, int(np.floor(np.min(landmarks_xy[:, 1])) - 20))
        x_max = min(frame_width, int(np.ceil(np.max(landmarks_xy[:, 0])) + 20))
        y_max = min(frame_height, int(np.ceil(np.max(landmarks_xy[:, 1])) + 20))
        return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

    def _update_dynamic_action(
        self,
        prediction: HandPrediction,
        frame_size: tuple[int, int],
        faces: list[FaceDetection],
    ) -> str | None:
        x, y, w, h = prediction.bbox
        center_x = x + w / 2
        center_y = y + h / 2
        self.history.append(
            _HandHistorySample(
                frame_index=self.frame_index,
                center_x=center_x,
                center_y=center_y,
                static_gesture=prediction.static_gesture,
            )
        )

        new_action = self._infer_dynamic_action(frame_size, faces)
        if new_action is not None:
            self.active_dynamic_action = (new_action, self.frame_index + 8)
            return new_action
        if self.active_dynamic_action is None:
            return None
        action, expires_at = self.active_dynamic_action
        if self.frame_index <= expires_at:
            return action
        self.active_dynamic_action = None
        return None

    def _infer_dynamic_action(
        self,
        frame_size: tuple[int, int],
        faces: list[FaceDetection],
    ) -> str | None:
        if len(self.history) < 4:
            return None
        frame_height, frame_width = frame_size
        recent = list(self.history)[-12:]
        xs = np.array([sample.center_x for sample in recent], dtype=np.float32)
        ys = np.array([sample.center_y for sample in recent], dtype=np.float32)
        labels = [sample.static_gesture for sample in recent]

        if self._is_waving(xs, labels, frame_width):
            return "waving"
        if self._is_open_close_transition(labels):
            return "open_close_transition"
        if self._is_hand_raised(ys, frame_height, faces):
            return "hand_raised"
        return None

    @staticmethod
    def _is_waving(xs: np.ndarray, labels: list[str], frame_width: int) -> bool:
        if len(xs) < 6:
            return False
        dx = np.diff(xs)
        if len(dx) < 3:
            return False
        significant = np.abs(dx) > frame_width * 0.02
        directions = np.sign(dx[significant])
        if len(directions) < 3:
            return False
        direction_changes = np.sum(directions[:-1] * directions[1:] < 0)
        x_range = float(xs.max() - xs.min())
        open_like = sum(label in {"Open_Palm", "Pointing_Up", "Victory"} for label in labels)
        return x_range > frame_width * 0.18 and direction_changes >= 2 and open_like >= 3

    @staticmethod
    def _is_open_close_transition(labels: list[str]) -> bool:
        filtered = [label for label in labels if label in {"Open_Palm", "Closed_Fist"}]
        if len(filtered) < 3:
            return False
        changes = sum(prev != curr for prev, curr in zip(filtered, filtered[1:], strict=False))
        return changes >= 1 and {"Open_Palm", "Closed_Fist"}.issubset(set(filtered))

    @staticmethod
    def _is_hand_raised(ys: np.ndarray, frame_height: int, faces: list[FaceDetection]) -> bool:
        current_y = float(ys[-1])
        moved_up = float(np.max(ys[:-1]) - current_y) > frame_height * 0.12
        if faces:
            threshold = min(face.y + face.h * 0.35 for face in faces)
            return current_y < threshold and moved_up
        return current_y < frame_height * 0.35 and moved_up
