"""坏帧仪表：跟踪/重检路径 + 旋转差 + 关键点 RMS。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.engine_c import EngineC  # noqa: E402
from fylgja.engine_n import CropParams, EngineN  # noqa: E402
from fylgja.pose import euler_zxy  # noqa: E402

task = str(ROOT / "models" / "face_landmarker.task")
import cv2  # noqa: E402

cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
fps = 24.0

c = EngineC(task)
c.self_check()
ref = []
for i, f in enumerate(frames):
    r = c.process(f.copy(), 10 + int(i * 1000 / fps))
    ref.append(None if r is None else (np.asarray(r.transform), np.asarray(r.landmarks)))
c.close()

n = EngineN(task, device="NPU", crop=CropParams())
watch = {238, 239, 240, 241, 242, 243, 316, 318, 320, 330, 331}
for i, f in enumerate(frames):
    ts = 50010 + int(i * 1000 / fps)
    used_detector = n._prev_raw is None
    r = n.process(f.copy(), ts)
    if i in watch:
        path = "DET" if used_detector else "TRK"
        msg = f"idx={i} path={path}"
        if r is not None and ref[i] is not None:
            Tn = np.asarray(r.transform)
            d = np.array(euler_zxy(Tn[:3, :3])) - np.array(euler_zxy(ref[i][0][:3, :3]))
            h, w = f.shape[:2]
            lmrms = np.sqrt((((np.asarray(r.landmarks)[:, :2] - ref[i][1][: :2] if False else np.asarray(r.landmarks)[:, :2] - ref[i][1][:, :2]) * np.array([w, h])) ** 2).sum(axis=1).mean())
            msg += f" rot={np.sqrt((d ** 2).sum()):5.2f} lm={lmrms:5.1f}px"
        print(msg)
