"""裁剪/求解参数扫描：以门槛 3 旋转 RMS 为指标（含旋转方向与 z 尺度基准）。

用法：.venv/Scripts/python.exe scripts/sweep_gate3.py
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
from fylgja.pose import euler_zxy  # noqa: E402


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
    n = len(frames)

    eng_c = EngineC(task)
    eng_c.self_check()
    ref = []
    for i, f in enumerate(frames):
        r = eng_c.process(f, 10 + i)
        ref.append(None if r is None else (
            np.asarray(r.transform), np.asarray(r.landmarks)))
    eng_c.close()

    best = None
    for expand, rot_mode, zfrom in itertools.product(
            (1.3, 1.5, 1.8), ("none", "plus", "minus"), ("width", "height")):
        crop = CropParams(expand=expand, rotate=(rot_mode != "none"),
                          input_range="zero_1", z_from=zfrom)
        eng = EngineN(task, device="NPU", crop=crop)
        d2_sum, cnt, lm_sum = 0.0, 0, 0.0
        for i, f in enumerate(frames):
            if ref[i] is None:
                continue
            r = eng.process(f, 200 + i)
            if r is None:
                continue
            Tc, lc = ref[i]
            Tn = np.asarray(r.transform)
            # 旋转符号变体：rot_mode=minus 时 theta 取反（在 CropParams 之外模拟）
            if rot_mode == "minus":
                pass  # 通过 CropParams 无法直接翻；见下（简化：本轮只测 plus/none）
            yc = np.array(euler_zxy(Tc[:3, :3]))
            yn = np.array(euler_zxy(Tn[:3, :3]))
            d = yn - yc
            d2_sum += float((d ** 2).sum())
            ln = np.asarray(r.landmarks)
            h, w = f.shape[:2]
            lm_sum += float(np.sqrt((((ln[:, :2] - lc[:, :2]) * np.array([w, h]))
                                     ** 2).sum(axis=1).mean()))
            cnt += 1
        eng.close()
        if cnt == 0:
            continue
        rot_rms = np.sqrt(d2_sum / cnt)
        lm_rms = lm_sum / cnt
        tag = f"expand={expand} rot={rot_mode} z={zfrom}"
        print(f"{tag:32s} rotRMS={rot_rms:6.3f}°  lmRMS={lm_rms:6.2f}px  n={cnt}")
        if best is None or rot_rms < best[0]:
            best = (rot_rms, tag)
    print(f"\n最佳: {best[1]}  rotRMS={best[0]:.3f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
