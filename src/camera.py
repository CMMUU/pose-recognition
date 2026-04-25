import cv2

DESIRED_FRAME_SIZES = ((1280, 720), (960, 540), (640, 480))


class CameraStream:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture: cv2.VideoCapture | None = None
        self.frame_size: tuple[int, int] | None = None

    def open(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError("无法打开摄像头")
        self._configure_capture()

    def _configure_capture(self) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for width, height in DESIRED_FRAME_SIZES:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            actual_height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.frame_size = (actual_width, actual_height)
            if actual_width >= width and actual_height >= height:
                return

    def read(self):
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError("摄像头未打开")
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("读取摄像头画面失败")
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.frame_size = None
