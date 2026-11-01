"""旋转角空间对照（用引擎 C 参照关键点，避开检测器）：像素角 vs 归一化角。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.engine_c import EngineC  # noqa: E402

import cv2  # noqa: E402

task = str(ROOT / "models" / "face_landmarker.task")
cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
frames = {}
idx = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if idx in (242, 318):
        frames[idx] = f
    else:
        pass
    idx += 1
cap.release()
cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
c = EngineC(task)
c.self_check()
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    r = c.process(f.copy(), 10 + int(i * 1000 / 24))
    if i in frames and r is not None:
        frames[i] = (f, np.asarray(r.landmarks))
    i += 1
cap.release()
c.close()

for i, item in frames.items():
    if isinstance(item, tuple):
        f, lm = item
    else:
        continue
    h, w = f.shape[:2]
    x_min, x_max = lm[:, 0].min(), lm[:, 0].max()
    y_min, y_max = lm[:, 1].min(), lm[:, 1].max()
    ccx, ccy = (x_min + x_max) / 2 * w, (y_min + y_max) / 2 * h
    side = 1.5 * max((x_max - x_min) * w, (y_max - y_min) * h)
    e0, e1 = lm[33], lm[263]
    dx_n, dy_n = e1[0] - e0[0], e1[1] - e0[1]
    theta_norm = float(np.degrees(np.arctan2(dy_n, dx_n)))
    theta_px = float(np.degrees(np.arctan2(dy_n * h, dx_n * w)))
    crops = []
    for ang in (theta_px, theta_norm):
        a = np.deg2rad(ang)
        s = 256.0 / side
        m = np.array([[s * np.cos(a), s * np.sin(a), 128 - s * (np.cos(a) * ccx + np.sin(a) * ccy)],
                      [-s * np.sin(a), s * np.cos(a), 128 - s * (-np.sin(a) * ccx + np.cos(a) * ccy)]])
        crops.append(cv2.warpAffine(f, m, (256, 256)))
    out = ROOT / "out" / "stage0" / f"theta_{i}.jpg"
    cv2.imwrite(str(out), np.hstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"idx={i} theta_px={theta_px:.1f} theta_norm={theta_norm:.1f} -> {out.name}")
