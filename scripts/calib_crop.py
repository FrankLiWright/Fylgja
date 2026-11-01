"""裁剪约定校准 v2：双引擎全程顺序处理（ROI 跟踪依赖帧序），密采样对比。

用法：.venv/Scripts/python.exe scripts/calib_crop.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.engine_c import EngineC  # noqa: E402
from fylgja.engine_n import CropParams, EngineN  # noqa: E402

SAMPLE_EVERY = 20  # 每 20 帧取一个对照样本


def main() -> int:
    import cv2

    task = str(ROOT / "models" / "face_landmarker.task")
    cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n_frames = len(frames)
    sample_idx = list(range(0, n_frames, SAMPLE_EVERY))
    print(f"{n_frames} 帧，样本 {len(sample_idx)} 个")

    # ---- 参照：引擎 C 全程顺序 ----
    engine_c = EngineC(task)
    engine_c.self_check()
    ref = {}
    for i, f in enumerate(frames):
        r = engine_c.process(f, 10 + i)
        if r is not None and i in sample_idx:
            ref[i] = np.asarray(r.landmarks)[:, :2]
    engine_c.close()
    print(f"引擎 C 检出样本：{len(ref)}/{len(sample_idx)}")

    best = None
    for expand, rotate in itertools.product((1.2, 1.5, 1.8), (True, False)):
        crop = CropParams(expand=expand, rotate=rotate,
                          input_range="zero_1", z_from="width")
        eng = EngineN(task, device="NPU", crop=crop)
        errs = []
        for i, f in enumerate(frames):
            r = eng.process(f, 200 + i)
            if i in ref and r is not None:
                lm = np.asarray(r.landmarks)[:, :2]
                h, w = f.shape[:2]
                px = np.linalg.norm((lm - ref[i]) * np.array([w, h]), axis=1)
                errs.append((i, float(np.sqrt((px ** 2).mean()))))
            elif i in ref:
                errs.append((i, 999.0))
        eng.close()
        if not errs:
            continue
        mean = float(np.mean([e for _, e in errs]))
        worst_i, worst = max(errs, key=lambda t: t[1])
        tag = f"expand={expand} rotate={rotate}"
        print(f"{tag:24s} mean_RMS={mean:7.3f} px  worst: idx={worst_i} ({worst:.1f} px)")
        if best is None or mean < best[0]:
            best = (mean, tag, errs)

    print(f"\n最佳: {best[1]}  mean RMS={best[0]:.3f} px")
    print("逐样本误差:", " ".join(f"{i}:{e:.1f}" for i, e in best[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
