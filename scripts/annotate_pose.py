"""姿态可视化标注：人脸框 + 关键点 + 头部三轴箭头 + 姿态数值。

用法：.venv/Scripts/python.exe scripts/annotate_pose.py [--engine C|N] [--both]

三轴箭头从鼻尖（landmark 1）出发：
  红 = 面部 X 轴（面部右）  绿 = 面部 Y 轴（面部上）  蓝 = 面部 Z 轴（面部前/鼻方向）
箭头方向 = 把规范坐标系三轴经位姿矩阵变换后投影到图像（度量→图像仿射拟合，
拟合残差随帧打印，残差小即投影可信）。输出 out/pose_annotated.mp4 + 关键帧 PNG。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.engine_c import EngineC  # noqa: E402
from fylgja.model_assets import extract_assets, parse_geometry_metadata  # noqa: E402
from fylgja.pose import euler_zxy  # noqa: E402

AXIS_LEN = 4.0  # cm


def fit_metric_to_image(metric: np.ndarray, img_xy: np.ndarray):
    a = np.hstack([metric, np.ones((len(metric), 1))])
    w, *_ = np.linalg.lstsq(a, img_xy, rcond=None)
    resid = float(np.sqrt((((a @ w) - img_xy) ** 2).sum(axis=1).mean()))
    return w, resid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="C", choices=["C", "N"])
    ap.add_argument("--both", action="store_true", help="叠加引擎 N 红点对照")
    ap.add_argument("--video", default=str(ROOT / "test.mp4"))
    args = ap.parse_args()

    import cv2

    assets = extract_assets(ROOT / "models" / "face_landmarker.task",
                            ROOT / "models" / "extracted")
    meta = parse_geometry_metadata(
        assets["geometry_pipeline_metadata_landmarks.binarypb"].read_bytes())
    verts = meta.canonical_vertices  # (468,3)

    task = str(ROOT / "models" / "face_landmarker.task")
    eng = EngineC(task) if args.engine == "C" else None
    if eng is None:
        from fylgja.engine_n import EngineN
        eng = EngineN(task, device="NPU")
    eng.self_check()
    eng_n = None
    if args.both:
        from fylgja.engine_n import EngineN
        eng_n = EngineN(task, device="NPU")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = ROOT / "out" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_dir / "pose_annotated.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    resid_log = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = 10 + int(idx * 1000 / fps)
        res = eng.process(frame.copy(), ts)
        vis = frame.copy()
        if res is not None:
            lm = np.asarray(res.landmarks)
            t4 = np.asarray(res.transform)
            rmat, tvec = t4[:3, :3], t4[:3, 3]
            yaw, pitch, roll = euler_zxy(rmat)

            # ---- 关键点 + 人脸框 ----
            for p in lm:
                cv2.circle(vis, (int(p[0] * w), int(p[1] * h)), 1, (0, 255, 0), -1)
            x1, y1 = int(lm[:, 0].min() * w), int(lm[:, 1].min() * h)
            x2, y2 = int(lm[:, 0].max() * w), int(lm[:, 1].max() * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # ---- 三轴箭头（规范三轴经位姿 → 度量 → 图像仿射投影）----
            metric = (rmat @ verts.T).T + tvec  # (468,3)
            # 度量→图像 仿射拟合（468 规范顶点 ↔ 前 468 关键点；余 10 个为虹膜点）
            a = np.hstack([metric, np.ones((len(metric), 1))])
            wmat, *_ = np.linalg.lstsq(a, lm[:468, :2], rcond=None)
            resid = float(np.sqrt(((a @ wmat - lm[:468, :2]) ** 2).sum(axis=1).mean()))
            resid_log.append(resid)

            def proj(p3):
                return np.array([p3[0], p3[1], p3[2], 1.0]) @ wmat

            origin = tvec  # 规范原点 → 度量
            nose_img = (int(lm[1, 0] * w), int(lm[1, 1] * h))  # 鼻尖 landmark 1
            colors = {"X": (0, 0, 255), "Y": (0, 255, 0), "Z": (255, 0, 0)}
            for name, d in (("X", (AXIS_LEN, 0, 0)), ("Y", (0, AXIS_LEN, 0)),
                            ("Z", (0, 0, AXIS_LEN))):
                tip = origin + rmat @ np.array(d)
                p0 = proj(origin)
                p1 = proj(tip)
                cv2.arrowedLine(vis,
                                (int(p0[0] * w), int(p0[1] * h)),
                                (int(p1[0] * w), int(p1[1] * h)),
                                colors[name], 3, tipLength=0.25)
            cv2.circle(vis, nose_img, 5, (255, 255, 255), -1)

            # ---- 数值 ----
            txt = (f"yaw {yaw:+7.2f}  pitch {pitch:+7.2f}  roll {roll:+7.2f} deg | "
                   f"TX {tvec[0]:+6.2f} TY {tvec[1]:+6.2f} TZ {tvec[2]:+6.2f}")
            cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2)
            cv2.putText(vis, f"resid {resid:.2f}px  engine {args.engine}",
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if eng_n is not None:
                rn = eng_n.process(frame.copy(), ts + 50000)
                if rn is not None:
                    for p in np.asarray(rn.landmarks)[::4]:
                        cv2.circle(vis, (int(p[0] * w), int(p[1] * h)), 1,
                                   (0, 0, 255), -1)

            if idx % int(fps * 2) == 0:
                cv2.imwrite(str(out_dir / f"pose_f{idx:04d}.png"), vis)
        vw.write(vis)
        idx += 1
    cap.release()
    vw.release()
    eng.close()
    if eng_n is not None:
        eng_n.close()
    if resid_log:
        print(f"仿射投影残差：mean {np.mean(resid_log):.2f}px max {np.max(resid_log):.2f}px")
    print(f"输出视频：{out_dir / 'pose_annotated.mp4'}（{idx} 帧）")
    print(f"关键帧：{out_dir}\\pose_f*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
