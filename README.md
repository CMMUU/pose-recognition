# 人脸与手部动作识别桌面 Demo / Face and Hand Action Recognition Desktop Demo

这是一个基于 Python + PySide6 + OpenCV + ONNX Runtime 的本地桌面应用，支持：
This is a local desktop application built with Python + PySide6 + OpenCV + ONNX Runtime. It supports:

- 图片上传的人脸表情识别 / Facial emotion recognition from uploaded images
- 实时摄像头人脸表情识别 / Real-time facial emotion recognition from camera input
- 手势识别（如张开手掌、握拳、点赞等） / Hand gesture recognition (such as open palm, closed fist, thumbs up, etc.)
- 摄像头下的动态手部动作识别（挥手、举手、张开/握拳切换） / Dynamic hand action recognition in camera mode (waving, hand raised, and open-close transitions)
- 界面与结果的中英双语展示 / Bilingual Chinese-English UI and result display

## 运行环境 / Environment

建议使用你已经创建好的 conda 环境：
It is recommended to use your existing conda environment:

```bash
conda activate ai
```

## 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

## 下载模型 / Download Models

### 1) 表情识别模型 / Emotion model

```bash
curl -L "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" -o models/emotion-ferplus-8.onnx
```

### 2) 手部识别模型 / Hand recognition models

下载 Qualcomm 提供的 ONNX 模型压缩包，并解压到 `models/` 目录。
Download the ONNX hand model archive provided by Qualcomm and extract it into the `models/` directory.

```bash
curl -L "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_hand_gesture/releases/v0.51.0/mediapipe_hand_gesture-onnx-float.zip" -o models/mediapipe_hand_gesture-onnx-float.zip
unzip -o models/mediapipe_hand_gesture-onnx-float.zip -d models
```

解压后应包含以下文件：
After extraction, the following files should exist:

- `models/mediapipe_hand_gesture-onnx-float/palm_detector.onnx`
- `models/mediapipe_hand_gesture-onnx-float/hand_landmark_detector.onnx`
- `models/mediapipe_hand_gesture-onnx-float/canned_gesture_classifier.onnx`

## 启动应用 / Run the App

```bash
python app.py
```

## 打包桌面安装版本 / Package Desktop Builds

### 本地打包 / Local packaging

先安装 PyInstaller：
Install PyInstaller first:

```bash
pip install pyinstaller
```

然后执行：
Then run:

```bash
pyinstaller --noconfirm app.spec
```

在 Windows 上，如需生成安装器，再执行：
On Windows, run the installer build as well:

```bash
iscc packaging\\windows\\installer.iss
```

构建完成后，PyInstaller 产物位于 `dist/FaceHandRecognition/`，安装器与其他发布文件位于 `packaging/out/`。
After the build completes, the PyInstaller output is placed in `dist/FaceHandRecognition/`, and the installer plus other release assets are placed in `packaging/out/`.

打包时会自动带上以下运行资源：
The package automatically includes these runtime assets:

- `models/emotion-ferplus-8.onnx`
- `models/mediapipe_hand_gesture-onnx-float/` 下全部模型与 `.data` 文件 / all hand model files and `.data` files under `models/mediapipe_hand_gesture-onnx-float/`
- OpenCV Haar cascade 文件 / OpenCV Haar cascade file

因此发布包内用户无需再单独下载模型。
So end users do not need to download the models separately for the packaged release.

### GitHub Actions 自动构建 / GitHub Actions builds

仓库包含 `.github/workflows/build.yml`，可在以下平台自动构建：
The repository includes `.github/workflows/build.yml` to build automatically on:

- Windows
- macOS
- Linux

触发方式：
Triggers:

- 手动触发 `workflow_dispatch` / manual `workflow_dispatch`
- 推送 `v*` 标签 / pushing a `v*` tag

默认产物：
Default outputs:

- Windows: 便携版 `.zip` + Inno Setup 安装器 `.exe` / portable `.zip` + Inno Setup installer `.exe`
- macOS: `.dmg`
- Linux: `.tar.gz` 压缩包 / `.tar.gz` archive
- 所有平台：`SHA256SUMS.txt` 校验文件 / `SHA256SUMS.txt` checksum file

自动发布行为：
Automatic release behavior:

- 手动触发时，工作流只生成并上传 Actions artifacts / manual runs only build and upload Actions artifacts
- 推送 `v*` 标签时，工作流会自动创建或更新对应 GitHub Release，并附加各平台安装包 / pushing a `v*` tag automatically creates or updates the matching GitHub Release and attaches the packaged assets

## 功能说明 / Features

### 图片模式 / Image mode

- 支持人脸表情识别 / Supports facial emotion recognition
- 支持当前手势识别 / Supports current hand gesture recognition
- 如果图片中检测到手，结果区会提示：动态动作识别需要连续摄像头帧 / If a hand is detected in an image, the app will explain that dynamic actions require continuous camera frames

### 摄像头模式 / Camera mode

- 支持实时人脸表情识别 / Supports real-time facial emotion recognition
- 支持实时手势识别 / Supports real-time hand gesture recognition
- 支持动态手部动作识别： / Supports dynamic hand action recognition:
  - 挥手 / Waving
  - 举手 / Hand Raised
  - 张开/握拳切换 / Open-Close Transition

## 当前使用的模型 / Models in Use

### 表情模型 / Emotion model

- FER+ ONNX 模型 / FER+ ONNX model
- 输入规格：`1x1x64x64` / Input shape: `1x1x64x64`
- 标签顺序 / Label order:
  - neutral
  - happiness
  - surprise
  - sadness
  - anger
  - disgust
  - fear
  - contempt

### 手部模型 / Hand models

- Qualcomm MediaPipe Hand Gesture Recognition ONNX pipeline
- 三阶段模型 / Three-stage pipeline:
  - `palm_detector.onnx`
  - `hand_landmark_detector.onnx`
  - `canned_gesture_classifier.onnx`
- 静态手势标签 / Static gesture labels:
  - `Closed_Fist`
  - `Open_Palm`
  - `Pointing_Up`
  - `Thumb_Down`
  - `Thumb_Up`
  - `Victory`
  - `ILoveYou`

## 使用说明 / Notes

- 动态手部动作识别依赖连续帧和简单时序规则，适合本地 Demo 演示，但不是严格的动作时序模型。 / Dynamic hand action recognition relies on continuous frames and lightweight temporal rules, which is suitable for a local demo but not a full temporal action model.
- 如果运行时报错模型缺失，请优先检查 `models/` 目录结构是否正确。 / If the app reports missing models, first verify the structure under `models/`.
