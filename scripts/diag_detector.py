"""诊断：detector 原始输出分布 + 与引擎 C 关键点人脸框对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.blazeface import decode_detections, generate_anchors, nms_weighted  # noqa: E402
from fylgja.engine_c import EngineC  # noqa: E402


def letterbox(frame, size=128):
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


import cv2  # noqa: E402

cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]

# ---- 引擎 C 参照 ----
engine = EngineC(str(ROOT / "models" / "face_landmarker.task"))
engine.self_check()
res = engine.process(frame, 100)
lm = res.landmarks  # (478,3) 归一化 x,y
print("engine C landmark bbox (normalized image):",
      f"x[{lm[:, 0].min():.3f},{lm[:, 0].max():.3f}] y[{lm[:, 1].min():.3f},{lm[:, 1].max():.3f}]")
engine.close()

# ---- detector ----
import openvino as ov

core = ov.Core()
det = core.compile_model(
    core.read_model(str(ROOT / "models" / "extracted" / "face_detector.tflite")), "CPU")
canvas, scale = letterbox(frame)
x = (canvas.astype(np.float32) / 127.5 - 1.0)[None]
out = det({det.input(0): x})
arrs = [(np.asarray(v), str(k)) for k, v in out.items()]
arrs.sort(key=lambda t: t[0].size)
for v, k in arrs:
    print("output", k, v.shape, "min", v.min(), "max", v.max(), "mean", v.mean())

reg = np.concatenate([v.reshape(-1) for v, k in arrs if v.size == 896 * 16]).reshape(896, 16)
cls = np.concatenate([v.reshape(-1) for v, k in arrs if v.size == 896]).reshape(896, 1)
print("regressors abs-max per column:", np.round(np.abs(reg).max(axis=0), 2))
print("classificators raw range:", cls.min(), cls.max(), " (logits?)")

anchors = generate_anchors()
dets = nms_weighted(decode_detections(reg, cls, anchors, min_score=0.2))
print(f"\ndetector top dets (letterbox-normalized → image-normalized):")
for d in dets[:5]:
    # letterbox 归一化 → 原图归一化：canvas 128 → 原图坐标
    cx, cy, bw, bh = d.box * 128.0 / scale / np.array([w, h, w, h])
    print(f"  score={d.score:.3f} img-normalized box=({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f})")
