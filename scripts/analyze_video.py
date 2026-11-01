"""测试视频面部运动解析（评估姿态估计精度）。

用法（仓库根目录）：
    .venv/Scripts/python.exe scripts/analyze_video.py test.mp4

对每一帧跑引擎 C（MediaPipe FaceLandmarker VIDEO 模式，时间戳按帧号×1000/fps 单调递增），
记录原始姿态与 One Euro 滤波后姿态，输出：
  - 逐帧 CSV（out/<视频名>_frames.csv）
  - 精度报告（检出率/量程/抖动/平滑度/跳变/静止段噪声）
  - 关键帧 PNG（out/<视频名>_frame_*.png）用于人工对照曲线
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.engine_c import EngineC  # noqa: E402
from fylgja.pose import pose_from_transform  # noqa: E402
from fylgja.postprocess import PostProcessor  # noqa: E402


def analyze(video_path: Path, out_dir: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频：{video_path}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频：{video_path.name} {width}x{height} @{fps:g}fps {total} 帧")

    model = ROOT / "models" / "face_landmarker.task"
    engine = EngineC(str(model))
    engine.self_check()
    pp = PostProcessor()  # 默认 One Euro 参数

    rows = []  # (idx, t_s, detected, yaw, pitch, roll, tx, ty, tz, f_yaw, f_pitch, f_roll, infer_ms)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = int(round(idx * 1000.0 / fps)) + 10  # 单调递增；+10 避开自检时间戳
        import time as _t
        t0 = _t.perf_counter()
        res = engine.process(frame, ts_ms)
        infer_ms = (_t.perf_counter() - t0) * 1000.0

        if res is None:
            rows.append((idx, ts_ms / 1000.0, 0, *([float("nan")] * 6), *([float("nan")] * 3), infer_ms))
        else:
            raw = pose_from_transform(res.transform)
            fp = pp.process(res.transform, t=ts_ms / 1000.0)
            rows.append((
                idx, ts_ms / 1000.0, 1,
                raw.yaw, raw.pitch, raw.roll, raw.tx, raw.ty, raw.tz,
                fp.yaw, fp.pitch, fp.roll, infer_ms,
            ))
            # 关键帧：每 2 秒 + 运动极值处由后处理补拍（此处定时采样）
            if idx % int(fps * 2) == 0:
                cv2.imwrite(str(frames_dir / f"{video_path.stem}_f{idx:04d}.png"), frame)
        idx += 1
    cap.release()
    engine.close()

    # ---- 落盘 CSV ----
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{video_path.stem}_frames.csv"
    header = ("idx,t_s,detected,yaw,pitch,roll,tx,ty,tz,"
              "filt_yaw,filt_pitch,filt_roll,infer_ms")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")

    # ---- 统计报告 ----
    det = np.array([r[2] for r in rows], dtype=bool)
    raw = np.array([[r[3], r[4], r[5]] for r in rows], dtype=float)  # yaw pitch roll
    fil = np.array([[r[9], r[10], r[11]] for r in rows], dtype=float)
    infer_ms = np.array([r[12] for r in rows], dtype=float)

    report: list[str] = []
    report.append(f"# 视频面部运动解析报告：{video_path.name}")
    report.append(f"总帧数 {len(rows)}，检出 {int(det.sum())}，检出率 {det.mean() * 100:.1f}%")
    report.append(f"推理耗时：均值 {np.nanmean(infer_ms):.1f} ms，p95 {np.nanpercentile(infer_ms, 95):.1f} ms")

    axes = ("yaw", "pitch", "roll")
    if det.any():
        for i, ax in enumerate(axes):
            v = raw[det, i]
            fv = fil[det, i]
            d1 = np.diff(fv)  # 帧间差分
            report.append(
                f"{ax}: 原始范围 [{np.nanmin(v):7.2f}, {np.nanmax(v):7.2f}]° 跨度 {np.nanmax(v) - np.nanmin(v):6.2f}° | "
                f"滤波后标准差 {np.nanstd(fv):5.2f}° | 帧间差分 |d|均值 {np.nanmean(np.abs(d1)):.3f}° 最大 {np.nanmax(np.abs(d1)):.2f}°"
            )
            # 跳变检测：帧间差分超 8° 视为可疑跳变（30fps 级别人头转动 ~0.5°/帧量级）
            jumps = int(np.sum(np.abs(d1) > 8.0))
            report.append(f"{ax}: 跳变(>8°/帧) 次数 = {jumps}")
        # 静止段抖动：取 |yaw'|+|pitch'|+|roll'| 的 20 百分位以下帧视为静止
        speed = np.nansum(np.abs(np.diff(fil, axis=0)), axis=1)
        thresh = np.nanpercentile(speed, 20) if len(speed) else 0
        static = speed <= max(thresh, 1e-9)
        if static.any():
            for i, ax in enumerate(axes):
                seg = fil[1:][static, i]
                report.append(f"静止段 {ax} 抖动（帧间差分 std）= {np.nanstd(np.diff(seg)):.4f}°")

    # 检出空洞（连续未检出段）
    gaps = []
    g = 0
    for d in det:
        if not d:
            g += 1
        elif g:
            gaps.append(g)
            g = 0
    if g:
        gaps.append(g)
    report.append(f"未检出空洞：{len(gaps)} 段" + (f"，最长 {max(gaps)} 帧" if gaps else ""))

    report_path = out_dir / f"{video_path.stem}_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\n[输出] {csv_path}\n[输出] {report_path}\n[输出] 关键帧 {frames_dir}")
    return 0


if __name__ == "__main__":
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test.mp4"
    out = ROOT / "out" / video.stem
    raise SystemExit(analyze(video, out))
