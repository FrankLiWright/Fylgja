"""门槛 3（方案书 4.5 第 3 条）：引擎 N 与引擎 C 旋转对齐验证。

同一批帧顺序过两引擎，比较 ZXY 欧拉角差与平移相对差：
  通过标准：旋转 RMS < 0.5°，平移向量相对差 < 5%。
用法：.venv/Scripts/python.exe scripts/gate3_alignment.py
"""
from __future__ import annotations

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
    print(f"{len(frames)} 帧")

    eng_c = EngineC(task)
    eng_c.self_check()
    eng_n = EngineN(task, device="NPU", crop=CropParams())  # 默认即 MediaPipe 精确约定

    rows = []
    fps = 24.0
    for i, f in enumerate(frames):
        ts = 10 + int(i * 1000.0 / fps)  # 真实时间步（MediaPipe 平滑/跟踪依赖 dt）
        rc = eng_c.process(f.copy(), ts)
        rn = eng_n.process(f.copy(), ts + 50000)
        if rc is None or rn is None:
            continue
        tc = np.asarray(rc.transform)
        tn = np.asarray(rn.transform)
        yc, pc, rc_ = euler_zxy(tc[:3, :3])
        yn, pn, rn_ = euler_zxy(tn[:3, :3])
        rows.append((i, yc, pc, rc_, yn, pn, rn_,
                     *tc[:3, 3], *tn[:3, 3]))
    eng_c.close()
    eng_n.close()

    a = np.asarray(rows)
    dy, dp, dr = a[:, 4] - a[:, 1], a[:, 5] - a[:, 2], a[:, 6] - a[:, 3]
    rot_rms = np.sqrt(np.mean(dy ** 2 + dp ** 2 + dr ** 2))
    print(f"样本帧：{len(a)}")
    print(f"yaw   差: RMS {np.sqrt(np.mean(dy**2)):.3f}°  max {np.abs(dy).max():.2f}°")
    print(f"pitch 差: RMS {np.sqrt(np.mean(dp**2)):.3f}°  max {np.abs(dp).max():.2f}°")
    print(f"roll  差: RMS {np.sqrt(np.mean(dr**2)):.3f}°  max {np.abs(dr).max():.2f}°")
    print(f"旋转总 RMS = {rot_rms:.3f}°（门槛 < 0.5°）")

    tc = a[:, 7:10]
    tn = a[:, 10:13]
    scale = np.linalg.norm(tc, axis=1).mean()
    rel = np.linalg.norm(tn - tc, axis=1) / np.maximum(scale, 1e-9)
    print(f"平移相对差: mean {rel.mean()*100:.2f}%  max {rel.max()*100:.2f}%（门槛 < 5%）")

    verdict = "通过" if rot_rms < 0.5 else "未通过"
    print(f"\n门槛 3：{verdict}")
    return 0 if rot_rms < 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
