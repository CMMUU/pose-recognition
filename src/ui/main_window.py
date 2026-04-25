import cv2
import numpy as np
from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.camera import CameraStream
from src.draw import draw_face_result, draw_hand_result, to_rgb
from src.emotion_classifier import EmotionClassifier
from src.face_detector import FaceDetection, FaceDetector
from src.hand_gesture_recognizer import GESTURE_TEXT, HAND_MODEL_FILES, HandGestureRecognizer, HandPrediction
from src.image_pipeline import ImageAnalysisResult, ImagePipeline
from src.runtime_paths import EMOTION_MODEL_PATH, HAND_MODEL_DIR

TOP_K = 3
CAMERA_ANALYSIS_MAX_EDGE = 960
PREVIEW_SHARPEN_KERNEL = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 5.0, -1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float32)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("人脸与手部动作识别 / Face and Hand Action Recognition")
        self.resize(1200, 760)

        self.detector = FaceDetector()
        self.classifier: EmotionClassifier | None = None
        self.hand_recognizer: HandGestureRecognizer | None = None
        self.pipeline: ImagePipeline | None = None
        self.camera = CameraStream()
        self.current_frame_bgr: np.ndarray | None = None
        self.preview_source = "placeholder"

        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.update_camera_frame)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(560, 420)
        self.image_label.setStyleSheet("border: 1px solid #999; background: #111; color: #ddd;")

        self.status_label = QLabel("正在初始化模型... / Initializing models...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        self.open_image_button = QPushButton("选择图片 / Choose Image")
        self.open_image_button.clicked.connect(self.open_image)

        self.start_camera_button = QPushButton("打开摄像头 / Open Camera")
        self.start_camera_button.clicked.connect(self.start_camera)

        self.stop_camera_button = QPushButton("关闭摄像头 / Close Camera")
        self.stop_camera_button.clicked.connect(self.stop_camera)

        self._build_layout()
        self._clear_preview("请选择图片或打开摄像头 / Choose an image or open the camera")
        self._set_result_text("正在加载模型... / Loading models...")
        self._set_controls(camera_running=False)
        self._load_model()

    def _build_layout(self) -> None:
        controls = QHBoxLayout()
        controls.addWidget(self.open_image_button)
        controls.addWidget(self.start_camera_button)
        controls.addWidget(self.stop_camera_button)
        controls.addStretch(1)

        content = QVBoxLayout()
        content.addLayout(controls)
        content.addWidget(self.status_label)
        content.addWidget(self.image_label, stretch=4)
        content.addWidget(self.result_box, stretch=2)

        container = QWidget()
        container.setLayout(content)
        self.setCentralWidget(container)

    def _missing_hand_models(self) -> list[str]:
        return [name for name in HAND_MODEL_FILES.values() if not (HAND_MODEL_DIR / name).exists()]

    def _load_model(self) -> None:
        if not EMOTION_MODEL_PATH.exists():
            self.classifier = None
            self.hand_recognizer = None
            self.pipeline = None
            self._set_status(f"表情模型未找到：{EMOTION_MODEL_PATH.name} / Emotion model not found: {EMOTION_MODEL_PATH.name}")
            self._clear_preview("模型文件缺失 / Model file missing")
            self._set_result_text(
                "表情模型未找到，请先下载 models/emotion-ferplus-8.onnx 后再运行。\n"
                "Emotion model not found. Please download models/emotion-ferplus-8.onnx before running."
            )
            self._set_controls(camera_running=False)
            return

        missing_hand_models = self._missing_hand_models()
        if missing_hand_models:
            self.classifier = None
            self.hand_recognizer = None
            self.pipeline = None
            self._set_status("手部模型未就绪 / Hand models are not ready")
            self._clear_preview("手部模型文件缺失 / Hand model files missing")
            self._set_result_text(
                "缺少手部模型文件：\n"
                f"- {'\n- '.join(missing_hand_models)}\n\n"
                "请先准备 mediapipe_hand_gesture-onnx-float 目录中的 3 个 ONNX 文件后再运行。\n"
                "Missing hand model files. Please prepare the 3 ONNX files in mediapipe_hand_gesture-onnx-float before running."
            )
            self._set_controls(camera_running=False)
            return

        try:
            self.classifier = EmotionClassifier(EMOTION_MODEL_PATH)
            self.hand_recognizer = HandGestureRecognizer(HAND_MODEL_DIR)
            self.pipeline = ImagePipeline(self.detector, self.classifier, self.hand_recognizer)
        except Exception as exc:
            self.classifier = None
            self.hand_recognizer = None
            self.pipeline = None
            self._set_status("模型加载失败 / Model loading failed")
            self._clear_preview("模型加载失败 / Model loading failed")
            self._set_result_text(f"无法加载模型：{exc}\nFailed to load models: {exc}")
            self._set_controls(camera_running=False)
            return

        self._set_status(
            "模型已就绪，可进行表情与手部动作识别 / Models ready for facial emotion and hand action recognition"
        )
        self._clear_preview("请选择图片或打开摄像头 / Choose an image or open the camera")
        self._set_result_text(
            "等待识别。\n"
            "Waiting for analysis.\n\n"
            "支持：人脸表情识别、手势识别、摄像头下的动态手部动作识别。\n"
            "Supports facial emotion recognition, hand gesture recognition, and dynamic hand action recognition in camera mode."
        )
        self._set_controls(camera_running=False)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _set_result_text(self, text: str) -> None:
        self.result_box.setPlainText(text)

    def _set_controls(self, camera_running: bool) -> None:
        model_ready = self.pipeline is not None
        self.open_image_button.setEnabled(model_ready and not camera_running)
        self.start_camera_button.setEnabled(model_ready and not camera_running)
        self.stop_camera_button.setEnabled(model_ready and camera_running)

    def _clear_preview(self, message: str) -> None:
        self.current_frame_bgr = None
        self.preview_source = "placeholder"
        self.image_label.clear()
        self.image_label.setText(message)

    def _target_preview_size(self, source_size: QSize) -> QSize:
        label_size = self.image_label.size()
        if self.preview_source != "camera":
            return label_size
        width = min(label_size.width(), source_size.width())
        height = min(label_size.height(), source_size.height())
        return QSize(width, height)

    @staticmethod
    def _resize_for_analysis(frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        max_edge = max(height, width)
        if max_edge <= CAMERA_ANALYSIS_MAX_EDGE:
            return frame_bgr
        scale = CAMERA_ANALYSIS_MAX_EDGE / max_edge
        target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _scale_face_predictions(
        predictions: list[tuple[FaceDetection, object]],
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[FaceDetection, object]]:
        scaled_predictions: list[tuple[FaceDetection, object]] = []
        for face, prediction in predictions:
            scaled_face = FaceDetection(
                x=int(round(face.x * scale_x)),
                y=int(round(face.y * scale_y)),
                w=max(1, int(round(face.w * scale_x))),
                h=max(1, int(round(face.h * scale_y))),
            )
            scaled_predictions.append((scaled_face, prediction))
        return scaled_predictions

    @staticmethod
    def _scale_hand_predictions(
        predictions: list[HandPrediction],
        scale_x: float,
        scale_y: float,
    ) -> list[HandPrediction]:
        scaled_predictions: list[HandPrediction] = []
        for prediction in predictions:
            x, y, w, h = prediction.bbox
            scaled_landmarks = prediction.landmarks.copy()
            scaled_landmarks[:, 0] *= scale_x
            scaled_landmarks[:, 1] *= scale_y
            scaled_predictions.append(
                HandPrediction(
                    bbox=(
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                        max(1, int(round(w * scale_x))),
                        max(1, int(round(h * scale_y))),
                    ),
                    landmarks=scaled_landmarks,
                    is_right_hand=prediction.is_right_hand,
                    static_gesture=prediction.static_gesture,
                    static_confidence=prediction.static_confidence,
                    dynamic_action=prediction.dynamic_action,
                )
            )
        return scaled_predictions

    def _build_camera_preview_frame(
        self,
        preview_frame_bgr: np.ndarray,
        result: ImageAnalysisResult,
        analysis_size: tuple[int, int],
    ) -> np.ndarray:
        preview_height, preview_width = preview_frame_bgr.shape[:2]
        analysis_height, analysis_width = analysis_size
        scale_x = preview_width / analysis_width
        scale_y = preview_height / analysis_height
        annotated_preview = preview_frame_bgr.copy()
        for face, prediction in self._scale_face_predictions(result.face_predictions, scale_x, scale_y):
            annotated_preview = draw_face_result(annotated_preview, face, prediction)
        for prediction in self._scale_hand_predictions(result.hand_predictions, scale_x, scale_y):
            annotated_preview = draw_hand_result(annotated_preview, prediction)
        return annotated_preview

    @staticmethod
    def _sharpen_preview(frame_bgr: np.ndarray, source: str) -> np.ndarray:
        if source != "camera":
            return frame_bgr
        return cv2.filter2D(frame_bgr, -1, PREVIEW_SHARPEN_KERNEL)

    @staticmethod
    def _resize_preview_frame(frame_bgr: np.ndarray, target_size: QSize) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if target_size.width() <= 0 or target_size.height() <= 0:
            return frame_bgr
        scale = min(target_size.width() / width, target_size.height() / height)
        if scale <= 0:
            return frame_bgr
        if abs(scale - 1.0) < 0.01:
            return frame_bgr
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=interpolation)

    def _refresh_preview(self) -> None:
        if self.current_frame_bgr is None:
            return
        target_size = self._target_preview_size(QSize(self.current_frame_bgr.shape[1], self.current_frame_bgr.shape[0]))
        preview_frame = self._resize_preview_frame(self.current_frame_bgr, target_size)
        preview_frame = self._sharpen_preview(preview_frame, self.preview_source)
        rgb = to_rgb(preview_frame)
        height, width, channel = rgb.shape
        image = QImage(rgb.data, width, height, channel * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setText("")
        self.image_label.setPixmap(pixmap)

    def _show_frame(self, frame_bgr: np.ndarray, source: str = "image") -> None:
        self.current_frame_bgr = frame_bgr.copy()
        self.preview_source = source
        self._refresh_preview()

    def _format_face_predictions(self, predictions) -> list[str]:
        lines: list[str] = []
        for index, (_, prediction) in enumerate(predictions, start=1):
            ranked = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
            best_label, best_score = ranked[0]
            lines.append(f"人脸 {index} / Face {index}")
            lines.append(f"主结果：{best_label} ({best_score:.2%}) / Top result: {best_label} ({best_score:.2%})")
            lines.append(f"前 {min(TOP_K, len(ranked))} 项 / Top {min(TOP_K, len(ranked))}:")
            for rank, (label, score) in enumerate(ranked[:TOP_K], start=1):
                lines.append(f"  {rank}. {label}: {score:.2%}")
            lines.append("")
        return lines

    def _format_hand_predictions(self, predictions) -> list[str]:
        lines: list[str] = []
        for index, prediction in enumerate(predictions, start=1):
            zh_gesture, en_gesture = GESTURE_TEXT.get(
                prediction.static_gesture,
                (prediction.static_gesture, prediction.static_gesture),
            )
            zh_hand, en_hand = prediction.handedness_text
            lines.append(f"手部 {index} / Hand {index}")
            lines.append(f"当前手势：{zh_gesture} ({prediction.static_confidence:.2%}) / Current gesture: {en_gesture} ({prediction.static_confidence:.2%})")
            lines.append(f"左右手：{zh_hand} / Handedness: {en_hand}")
            if prediction.dynamic_text is not None:
                zh_action, en_action = prediction.dynamic_text
                lines.append(f"动态动作：{zh_action} / Dynamic action: {en_action}")
            lines.append("")
        return lines

    def _show_result(self, result: ImageAnalysisResult, source: str) -> None:
        face_predictions = result.face_predictions
        hand_predictions = result.hand_predictions

        if not face_predictions and not hand_predictions:
            self._set_status(f"{source}中未检测到人脸或手部 / No face or hand detected in {source}")
            self._set_result_text(
                f"{source}中未检测到人脸或手部，请尝试更清晰、正对镜头的图像。\n"
                f"No face or hand detected in {source}. Please try a clearer image facing the camera."
            )
            return

        status_parts = []
        if face_predictions:
            status_parts.append(f"人脸 {len(face_predictions)} / face(s) {len(face_predictions)}")
        if hand_predictions:
            status_parts.append(f"手部 {len(hand_predictions)} / hand(s) {len(hand_predictions)}")
        self._set_status(f"{source}识别完成：{'，'.join(status_parts)} / {source} analysis complete")

        lines: list[str] = []
        if face_predictions:
            lines.append("表情识别结果 / Facial Emotion Results")
            lines.append("-")
            lines.extend(self._format_face_predictions(face_predictions))
        else:
            lines.append("未检测到人脸 / No face detected")
            lines.append("")

        if hand_predictions:
            lines.append("手部识别结果 / Hand Recognition Results")
            lines.append("-")
            lines.extend(self._format_hand_predictions(hand_predictions))
        else:
            lines.append("未检测到手部 / No hand detected")
            lines.append("")

        if result.hand_message:
            lines.append(result.hand_message)

        self._set_result_text("\n".join(lines).strip())

    def open_image(self) -> None:
        if self.pipeline is None:
            QMessageBox.warning(
                self,
                "模型缺失 / Model Missing",
                "请先准备表情与手部模型文件。\n"
                "Please prepare the emotion and hand model files first.",
            )
            return
        if self.timer.isActive():
            self.stop_camera()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片 / Choose Image",
            "",
            "图片文件 / Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not file_path:
            self._set_status("已取消选择图片 / Image selection cancelled")
            return

        self._set_status("正在分析图片... / Analyzing image...")
        try:
            result = self.pipeline.analyze_file(file_path)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "识别失败 / Analysis Failed",
                str(exc),
            )
            self._set_status("图片识别失败 / Image analysis failed")
            self._set_result_text(f"无法完成图片识别：{exc}\nFailed to analyze image: {exc}")
            return

        self._show_frame(result.annotated_frame, source="image")
        self._show_result(result, "图片 / Image")

    def start_camera(self) -> None:
        if self.pipeline is None:
            QMessageBox.warning(
                self,
                "模型缺失 / Model Missing",
                "请先准备表情与手部模型文件。\n"
                "Please prepare the emotion and hand model files first.",
            )
            return
        try:
            self.camera.open()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "摄像头错误 / Camera Error",
                str(exc),
            )
            self._set_status("摄像头打开失败 / Failed to open camera")
            self._set_result_text(f"无法打开摄像头：{exc}\nFailed to open camera: {exc}")
            return

        self.timer.start()
        self._set_controls(camera_running=True)
        self._set_status("摄像头已开启，正在实时识别 / Camera opened, real-time analysis is running")
        self._set_result_text(
            "正在等待摄像头画面...\n"
            "Waiting for camera frames...\n\n"
            "动态动作支持：挥手、举手、张开/握拳切换。\n"
            "Dynamic actions supported: waving, hand raised, and open-close transitions."
        )

    def stop_camera(self) -> None:
        was_running = self.timer.isActive() or (self.camera.capture is not None)
        self.timer.stop()
        self.camera.release()
        self._set_controls(camera_running=False)
        if was_running:
            self._set_status(
                "摄像头已关闭，可继续选择图片或重新打开摄像头 / Camera closed. You can choose an image or reopen the camera"
            )

    def update_camera_frame(self) -> None:
        if self.pipeline is None:
            return
        try:
            frame = self.camera.read()
            mirrored_frame = cv2.flip(frame, 1)
            analysis_frame = self._resize_for_analysis(mirrored_frame)
            result = self.pipeline.analyze(analysis_frame, enable_dynamic_actions=True)
            preview_frame = self._build_camera_preview_frame(
                mirrored_frame,
                result,
                analysis_frame.shape[:2],
            )
        except Exception as exc:
            self.stop_camera()
            QMessageBox.critical(
                self,
                "摄像头错误 / Camera Error",
                str(exc),
            )
            self._set_status("摄像头识别中断 / Camera analysis interrupted")
            self._set_result_text(f"实时识别已停止：{exc}\nReal-time analysis stopped: {exc}")
            return

        self._show_frame(preview_frame, source="camera")
        self._show_result(result, "实时画面 / Live Frame")

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.timer.stop()
        self.camera.release()
        super().closeEvent(event)
