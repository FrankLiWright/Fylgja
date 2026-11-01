"""BlazeFace 解码校准：在 test.mp4 真实帧上画检测框，确定 x_scale/wh_scale 约定。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.blazeface import decode_detections, generate_anchors, nms_weighted  # noqa: E402


def main() -> int:
    import cv2
    import openvino as ov

    assets = ROOT / "models" / "extracted"
    core = ov.Core()
    det = core.compile_model(core.read_model(str(assets / "face_detector.tflite")), "CPU")
    anchors = generate_anchors()

    cap = cv2.VideoCapture(str(ROOT / "test.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok

    # 预处理：128×128，保持宽高比 letterbox，float 范围 [-1, 1]（face_detection.pbtxt）
    h, w = frame.shape[:2]
    scale = 128.0 / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((128, 128, 3), 0, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    x = (canvas.astype(np.float32) / 127.5 - 1.0)[None]

    out = det({det.input(0): x})
    reg = list(out.values())[0]  # Identity order by model
    # 找出哪个输出是 regressors(896,16) 哪个是 classificators(896,1)
    outs = {}
    for k, v in out.items():
        arr = np.asarray(v)
        outs[arr.shape] = arr
    reg = outs[(1, 896, 16)]
    cls = outs[(1, 896, 1)]

    for xs, ws in ((1.0, 1.0), (128.0, 128.0), (1.0, 128.0), (128.0, 1.0)):
        dets = nms_weighted(decode_detections(reg, cls, anchors, min_score=0.3,
                                              x_scale=xs, wh_scale=ws))
        print(f"x_scale={xs} wh_scale={ws}: {len(dets)} dets")
        for d in dets[:3]:
            print(f"  score={d.score:.3f} box(cxcywh)={np.round(d.box, 3)}")
            print(f"  kp0={np.round(d.keypoints[0], 3)} kp1={np.round(d.keypoints[1], 3)}")

    # 用 128/128 约定画图（若框合理）
    dets = nms_weighted(decode_detections(reg, cls, anchors, min_score=0.3,
                                          x_scale=128.0, wh_scale=128.0))
    vis = frame.copy()
    for d in dets:
        cx, cy, bw, bh = d.box * 128.0 / scale  # 归一化输入坐标 → 原图
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kx, ky in d.keypoints * 128.0 / scale:
            cv2.circle(vis, (int(kx), int(ky)), 3, (0, 0, 255), -1)
        cv2.putText(vis, f"{d.score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out_path = ROOT / "out" / "stage0" / "detector_check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"annotated -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
