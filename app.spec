from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

PROJECT_ROOT = Path.cwd()
MODELS_DIR = PROJECT_ROOT / "models"
HAND_MODELS_DIR = MODELS_DIR / "mediapipe_hand_gesture-onnx-float"
HAARCASCADE_FILE = Path(__import__("cv2").data.haarcascades) / "haarcascade_frontalface_default.xml"

added_files = [
    (str(MODELS_DIR / "emotion-ferplus-8.onnx"), "models"),
    (str(HAARCASCADE_FILE), "models/haarcascades"),
]

for path in HAND_MODELS_DIR.rglob("*"):
    if path.is_file():
        destination = Path("models") / "mediapipe_hand_gesture-onnx-float" / path.relative_to(HAND_MODELS_DIR).parent
        added_files.append((str(path), str(destination).replace("\\", "/")))

added_files.extend(collect_data_files("cv2", includes=["data/*.xml"]))


a = Analysis(
    ["app.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=added_files,
    hiddenimports=["src.ui.main_window"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceHandRecognition",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceHandRecognition",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="FaceHandRecognition.app",
        icon=None,
        bundle_identifier=None,
        info_plist={
            "NSCameraUsageDescription": "Camera access is needed for live facial emotion and hand action recognition.",
        },
    )
