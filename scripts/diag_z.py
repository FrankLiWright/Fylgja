"""z 尺度诊断：对比两引擎关键点 z 列 + z 缩放变体对旋转 RMS 的影响。"""
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

n = EngineN(task, device="NPU", crop=CropParams(smooth=False))
print("frame | z_ours mean/std | z_C mean/std | ratio(std) | z_xy_corr")
for i, f in enumerate(frames):
    r = n.process(f.copy(), 50010 + int(i * 1000 / fps))
    if i not in (0, 120, 242, 316, 331) or r is None or ref[i] is None:
        continue
    zo = np.asarray(r.landmarks)[:, 2]
    zc = ref[i][1][:, 2]
    so, sc = zo.std(), zc.std()
    corr = np.corrcoef(zo, zc)[0, 1]
    ratio = sc / so if so > 0 else float("nan")
    # 最优线性拟合 zc ≈ a*zo + b
    a, b = np.polyfit(zo, zc, 1)
    print(f"{i:5d} | {zo.mean():7.3f}/{so:6.3f} | {zc.mean():7.3f}/{sc:6.3f} | {ratio:6.3f} | {corr:6.3f} | fit: zc={a:.3f}*zo+{b:.3f}")
