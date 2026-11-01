"""辅助分析：定位跳变帧 + 每秒曲线 + 真静止段抖动。"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "out" / "test" / "test_frames.csv")))

prev = None
print("--- 帧间差分>8° 的位置 ---")
for r in rows:
    cur = (float(r["filt_yaw"]), float(r["filt_pitch"]), float(r["filt_roll"]))
    if prev is not None:
        d = [abs(cur[i] - prev[i]) for i in range(3)]
        if max(d) > 8:
            print(f"t={float(r['t_s']):6.2f}s idx={r['idx']:>4} "
                  f"d=({d[0]:5.1f},{d[1]:5.1f},{d[2]:5.1f}) "
                  f"raw=({float(r['yaw']):7.1f},{float(r['pitch']):7.1f},{float(r['roll']):7.1f})")
    prev = cur

print("--- 每秒均值（滤波后） ---")
agg = defaultdict(list)
for r in rows:
    agg[int(float(r["t_s"]))].append(
        (float(r["filt_yaw"]), float(r["filt_pitch"]), float(r["filt_roll"])))
for s in sorted(agg):
    v = agg[s]
    m = [sum(x[i] for x in v) / len(v) for i in range(3)]
    print(f"{s:2d}s  yaw={m[0]:7.2f}  pitch={m[1]:7.2f}  roll={m[2]:7.2f}")

print("--- 真静止段抖动（滤波后帧间差分 std，速度<0.15°/帧） ---")
filt = np.array([[float(r["filt_yaw"]), float(r["filt_pitch"]), float(r["filt_roll"])]
                 for r in rows])
d1 = np.diff(filt, axis=0)
speed = np.abs(d1).sum(axis=1)
static = speed < 0.15
print(f"静止帧数: {int(static.sum())}/{len(speed)}")
if static.any():
    for i, ax in enumerate(("yaw", "pitch", "roll")):
        dd = d1[static, i]
        print(f"{ax}: 差分 std={np.std(dd):.4f}°  |d|最大={np.max(np.abs(dd)):.3f}°")
    # 连续静止段长度
    runs, g = [], 0
    for s in static:
        if s:
            g += 1
        elif g:
            runs.append(g); g = 0
    if g:
        runs.append(g)
    print(f"静止段: {len(runs)} 段，最长 {max(runs) if runs else 0} 帧")
