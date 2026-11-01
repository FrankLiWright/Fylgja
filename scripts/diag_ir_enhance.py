"""IR 暗帧增强测试：CLAHE / gamma 能否让无补光帧检出人脸。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from fylgja.engine_c import EngineC  # noqa: E402

task = str(ROOT / "models" / "face_landmarker.task")
c = EngineC(task)
c.self_check()
img = cv2.imread(str(ROOT / "ir_test_frame_raw.png"), cv2.IMREAD_GRAYSCALE)
variants = {
    "raw 灰度直通": img,
    "CLAHE 均衡": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img),
    "gamma 0.5 提亮": np.clip(((img / 255.0) ** 0.5) * 255, 0, 255).astype(np.uint8),
}
ts = 100
for name, g in variants.items():
    r = c.process(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), ts)
    ts += 100
    status = "检出" if r is not None else "未检出"
    print(f"{name}: {status}")
c.close()
