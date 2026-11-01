"""长跑测试台架（方案书 4.5 第 5 条 / 13.3）：test.mp4 循环 + 变速喂帧。

用法（仓库根目录，全部引擎 N 测试跑 NPU——用户约定）：
    .venv/Scripts/python.exe scripts/longrun_test.py --engine N --speed 1 --duration 60
    .venv/Scripts/python.exe scripts/longrun_test.py --engine C --speed 1 --duration 60   # 对照

--speed 0.5 = 慢放（每帧驻留 2 倍时间）；1 = 正常；2 = 快放（隔帧）。
输出：吞吐、推理耗时分布、**单帧进程 CPU 毫秒**（13.3 口径）、失败计数。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["N", "C"], required=True)
    ap.add_argument("--video", default=str(ROOT / "test.mp4"))
    ap.add_argument("--speed", type=float, default=1.0,
                    help="播放速率：0.5=慢放 1=正常 2=快放（隔帧）")
    ap.add_argument("--duration", type=float, default=60.0, help="墙钟秒数")
    ap.add_argument("--loop", action="store_true", default=True)
    args = ap.parse_args()

    import cv2

    task = str(ROOT / "models" / "face_landmarker.task")
    if args.engine == "C":
        from fylgja.engine_c import EngineC
        eng = EngineC(task)
    else:
        from fylgja.engine_n import CropParams, EngineN
        eng = EngineN(task, device="NPU", crop=CropParams())
    eng.self_check()

    # ---- 帧源：循环 + 变速 ----
    def frame_source():
        """产出 (frame, ts_ms)；ts 为合成播放时间线，步长 = 1000/(fps·speed)。"""
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        dt_ms = 1000.0 / (fps * args.speed)
        presented = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = cv2.VideoCapture(args.video)  # 循环
                continue
            skip = args.speed > 1.0 and (idx % int(round(args.speed)) != 0)
            reps = int(round(1.0 / args.speed)) if args.speed < 1.0 else 1
            idx += 1
            if skip:
                continue
            for _ in range(max(reps, 1)):
                yield frame, 10 + int(presented * dt_ms)  # int：MediaPipe 要求整型时间戳
                presented += 1

    src = frame_source()
    lat = []
    call_cpu = []  # 单次 engine.process 的进程 CPU 毫秒（剥离视频解码）
    n_ok = n_fail = n_none = 0
    first_err = None
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    deadline = t_wall0 + args.duration
    while time.perf_counter() < deadline:
        frame, ts = next(src)
        t0 = time.perf_counter()
        c0 = time.process_time()
        try:
            r = eng.process(frame.copy(), ts)
        except Exception as e:
            n_fail += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue
        call_cpu.append((time.process_time() - c0) * 1000.0)
        lat.append((time.perf_counter() - t0) * 1000.0)
        if r is None:
            n_none += 1
        else:
            n_ok += 1
    wall = time.perf_counter() - t_wall0
    cpu_ms = (time.process_time() - t_cpu0) * 1000.0
    eng.close()

    a = np.asarray(lat) if lat else np.zeros(1)
    cc = np.asarray(call_cpu) if call_cpu else np.zeros(1)
    total = n_ok + n_none + n_fail
    print(f"engine={args.engine} speed={args.speed} duration={wall:.1f}s")
    print(f"  帧数: {total}（检出 {n_ok} / 无人脸 {n_none} / 异常 {n_fail}）")
    if first_err:
        print(f"  首个异常: {first_err}")
    print(f"  吞吐: {total / wall:.1f} FPS（墙钟）")
    print(f"  推理延迟: p50 {np.percentile(a, 50):.2f} ms  p95 {np.percentile(a, 95):.2f} ms  max {a.max():.2f} ms")
    print(f"  引擎 CPU: {cc.mean():.2f} ms/帧（仅 process 调用）")
    print(f"  进程 CPU: {cpu_ms / max(total, 1):.2f} ms/帧（13.3 口径，含视频解码）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
