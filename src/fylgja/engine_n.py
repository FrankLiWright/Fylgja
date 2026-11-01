"""引擎 N 核心（方案书 4.1/4.3）：OpenVINO 管线 + 自建变换矩阵求解器。

ROI/解码约定逐项复刻 MediaPipe Tasks FaceLandmarker（face_landmarks_detector_graph.cc）：
  - ROI：全 478 关键点外接框 → scale 1.5 → 长边方形（RectTransformation），
    旋转 = 眼线角（landmark 33→263，目标角 0，DetectionsToRects）。
  - 裁剪：旋转方形区域 → 256×256（方形 FIT 无 letterbox），RGB，[0,1]。
  - 输出：模型 256 像素坐标 /256 → 逆仿射投影回图像；z 与 x 同尺度（×边长/宽）。
  - 平滑：One Euro（min_cutoff=0.05, beta=80, d_cutoff=1.0）作用于投影后关键点
    （LandmarksSmoothing，VIDEO 模式单脸默认启用），再进求解器。
  - presence 门控：sigmoid(输出1) ≥ 0.5 才输出；ROI 失效回退全图检测（方案书 4.3 第 4 条）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blazeface import decode_detections, generate_anchors, nms_weighted
from .engine import EngineResult, PoseEngine
from .errors import Attribution, SelfCheckError
from .face_geometry import FaceGeometrySolver
from .filter import OneEuroFilterND
from .logging_setup import get_logger
from .model_assets import extract_assets, parse_geometry_metadata

log = get_logger("engine_n")

EYE_LEFT_IDX = 33    # 左眼外角（DetectionsToRects rotation_vector_start）
EYE_RIGHT_IDX = 263  # 右眼外角（rotation_vector_end）
RECT_SCALE = 1.5     # RectTransformation scale_x/scale_y
SQUARE_LONG = True
PRESENCE_THRESH = 0.5


@dataclass
class CropParams:
    """保留的微调旋钮（默认即 MediaPipe 精确约定）。"""
    input_range: str = "zero_1"   # zero_1 | minus1_1
    z_from: str = "width"         # width | height
    smooth: bool = True           # LandmarksSmoothing（VIDEO 模式默认开）
    expand: float = RECT_SCALE    # 复刻值 1.5


class EngineN(PoseEngine):
    name = "N"

    def __init__(self, model_path: str, device: str = "NPU",
                 crop: CropParams | None = None):
        import openvino as ov

        self._core = ov.Core()
        self._device = device
        if device not in self._core.available_devices:
            # 方案书 5.3：NPU 不在设备列表 → 归因驱动问题
            raise SelfCheckError(
                Attribution.NPU_DRIVER if device == "NPU" else Attribution.SYSTEM,
                f"{device} 不在 OpenVINO 设备列表（可用：{self._core.available_devices}）"
                "：NPU 驱动未安装或版本过旧（方案书 8.3）",
            )
        self.crop = crop or CropParams()
        self._anchors = generate_anchors()
        self._prev_raw: np.ndarray | None = None  # 上一帧未平滑投影关键点（ROI 外推）
        self._affine: np.ndarray | None = None
        self._crop_side_px: float = 0.0
        self._smoother = OneEuroFilterND(
            (478, 3), min_cutoff=0.05, beta=80.0, d_cutoff=1.0)

        task = Path(model_path)
        if task.suffix == ".tflite":
            task = task.with_name("face_landmarker.task")
        assets = extract_assets(task, task.parent / "extracted")
        self._meta = parse_geometry_metadata(
            assets["geometry_pipeline_metadata_landmarks.binarypb"].read_bytes())

        det = self._core.compile_model(
            self._core.read_model(str(assets["face_detector.tflite"])), "CPU")
        self._det = det
        self._det_reg = next(o for o in det.outputs if o.shape[-1] == 16)
        self._det_cls = next(o for o in det.outputs if o.shape[-1] == 1)

        lmk_model = self._core.read_model(str(assets["face_landmarks_detector.tflite"]))
        lmk_model.reshape({0: [1, 256, 256, 3]})  # NPU 静态形状（方案书 4.3 第 1 条）
        self._lmk = self._core.compile_model(lmk_model, device)
        # 异步推理队列深度 2（方案书 4.3 第 2 条）：等待 NPU 时释放 GIL，
        # 采集线程不被饿死（采集与 NPU 流水线化）。
        self._req_pool = [self._lmk.create_infer_request() for _ in range(2)]
        self._req_i = 0
        self._lmk_land = self._lmk.output(0)   # (1,1,1,1434) 关键点
        self._lmk_pres = self._lmk.output(1)   # (1,1,1,1) presence logit
        self._solvers: dict[tuple[int, int], FaceGeometrySolver] = {}
        log.info("引擎 N 就绪：landmarks@%s detector@CPU", device)

    # ------------------------------------------------------------------ 自检

    def self_check(self) -> None:
        self._det({self._det.input(0): np.zeros((1, 128, 128, 3), np.float32)})
        self._lmk({self._lmk.input(0): np.zeros((1, 256, 256, 3), np.float32)})
        self._prev_raw = None
        self._smoother.reset()

    # ------------------------------------------------------------------ 单帧

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EngineResult | None:
        h, w = frame_bgr.shape[:2]
        t_s = timestamp_ms / 1000.0

        # ---- ROI（4.3 第 4 条：关键点外推；失效才全图重检）----
        roi = self._roi_from_landmarks(self._prev_raw, h, w)
        if roi is None:
            det = self._detect(frame_bgr)
            if det is None:
                self._prev_raw = None
                return None
            cx, cy, bw, bh, kps = det
            side = self.crop.expand * max(bw * w, bh * h)
            # 像素空间算角（实测：归一化角破坏对齐，见 out/stage0/theta_*）
            theta = float(np.degrees(np.arctan2(
                (kps[3] - kps[1]) * h, (kps[2] - kps[0]) * w)))
            roi = (cx * w, cy * h, side, theta)

        ccx, ccy, side, theta = roi
        out = self._infer_crop(frame_bgr, ccx, ccy, side, theta, h, w)
        if out is None:
            self._prev_raw = None
            return None
        img_lm, presence = out

        # presence 门控（LandmarksToDetection 链同样以此为界）
        if presence < PRESENCE_THRESH:
            self._prev_raw = None
            return None

        self._prev_raw = img_lm  # 下帧 ROI 用未平滑投影点（与图一致）

        # ---- 平滑（LandmarksSmoothing，VIDEO 单脸默认开；像素空间过滤）----
        if self.crop.smooth:
            scale = np.array([w, h, w], dtype=np.float64)
            lm = self._smoother.filter(img_lm * scale, t_s) / scale
        else:
            lm = img_lm

        solver = self._solver(h, w)
        pose, _ = solver.solve(lm[:468])
        return EngineResult(landmarks=np.asarray(lm, dtype=np.float32),
                            transform=pose.astype(np.float32))

    # ------------------------------------------------------------------ 内部

    def _roi_from_landmarks(self, lm: np.ndarray, h: int, w: int):
        """全 478 关键点外接框 → scale 1.5 → 长边方形；旋转 = 眼线角 33→263。"""
        if lm is None or len(lm) < 468:
            return None
        x_min, x_max = float(lm[:, 0].min()), float(lm[:, 0].max())
        y_min, y_max = float(lm[:, 1].min()), float(lm[:, 1].max())
        ccx = (x_min + x_max) / 2 * w
        ccy = (y_min + y_max) / 2 * h
        side = self.crop.expand * max((x_max - x_min) * w, (y_max - y_min) * h)
        e0, e1 = lm[EYE_LEFT_IDX], lm[EYE_RIGHT_IDX]
        # 像素空间算角（实测校准；归一化角版本实测 RMS 10°，勿改回）
        theta = float(np.degrees(np.arctan2(
            (e1[1] - e0[1]) * h, (e1[0] - e0[0]) * w)))
        return (ccx, ccy, side, theta)

    def _detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        scale = 128.0 / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        import cv2
        resized = cv2.resize(rgb, (nw, nh))
        canvas = np.zeros((128, 128, 3), dtype=np.uint8)
        y0, x0 = (128 - nh) // 2, (128 - nw) // 2  # 居中 letterbox
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        x = (canvas.astype(np.float32) / 127.5 - 1.0)[None]
        out = self._det({self._det.input(0): x})
        reg = np.asarray(out[self._det_reg]).reshape(896, 16)
        cls = np.asarray(out[self._det_cls]).reshape(896, 1)
        dets = nms_weighted(decode_detections(reg, cls, self._anchors, min_score=0.5))
        if not dets:
            return None
        best = max(dets, key=lambda d: d.score)

        def lb2img(px, py):
            return ((px * 128.0 - x0) / scale / w, (py * 128.0 - y0) / scale / h)

        cx, cy = lb2img(best.box[0], best.box[1])
        bw = best.box[2] * 128.0 / scale / w
        bh = best.box[3] * 128.0 / scale / h
        kx, ky = lb2img(best.keypoints[:, 0], best.keypoints[:, 1])
        kps = np.stack([kx, ky], axis=1).reshape(-1)
        return cx, cy, bw, bh, kps

    def _infer_crop(self, frame_bgr, ccx, ccy, side, theta, h, w):
        """旋转方形裁剪 → 256×256 → 推理 → 投影回图像 + presence。"""
        import cv2

        self._crop_side_px = side
        self._theta_rad = float(np.deg2rad(theta))
        a = self._theta_rad  # 目检校准：该方向把面部转正
        s = 256.0 / side
        m = np.array([
            [s * np.cos(a), s * np.sin(a), 128.0 - s * (np.cos(a) * ccx + np.sin(a) * ccy)],
            [-s * np.sin(a), s * np.cos(a), 128.0 - s * (-np.sin(a) * ccx + np.cos(a) * ccy)],
        ])
        self._affine = m
        crop = cv2.warpAffine(frame_bgr, m, (256, 256), flags=cv2.INTER_LINEAR)
        x = crop[:, :, ::-1].astype(np.float32)  # BGR→RGB
        if self.crop.input_range == "minus1_1":
            x = x / 127.5 - 1.0
        else:
            x = x / 255.0
        # 双请求异步池轮转（深度 2）：start_async 后 wait，等待期释放 GIL
        req = self._req_pool[self._req_i]
        self._req_i = (self._req_i + 1) % len(self._req_pool)
        req.start_async({0: x[None]})
        req.wait()
        pts = np.asarray(req.get_output_tensor(0).data).reshape(-1)
        presence = float(1.0 / (1.0 + np.exp(
            -float(np.asarray(req.get_output_tensor(1).data).ravel()[0]))))
        lm_crop = pts.reshape(478, 3) / 256.0
        return self._crop_to_image(lm_crop, h, w), presence

    def _crop_to_image(self, lm_crop: np.ndarray, h: int, w: int) -> np.ndarray:
        import cv2

        inv = cv2.invertAffineTransform(self._affine)
        px = lm_crop[:, 0] * 256.0
        py = lm_crop[:, 1] * 256.0
        ix = inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]
        iy = inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]
        out = np.zeros((478, 3), dtype=np.float64)
        out[:, 0] = ix / w
        out[:, 1] = iy / h
        # z 尺度 = LandmarkProjection.CalculateZScale：投影单位段长
        #   = side × √(cos²θ/w² + sin²θ/h²)（随旋转角逐帧变化）
        th = self._theta_rad
        z_scale = self._crop_side_px * float(np.sqrt(
            (np.cos(th) / w) ** 2 + (np.sin(th) / h) ** 2))
        out[:, 2] = lm_crop[:, 2] * z_scale
        return out

    def _solver(self, h: int, w: int) -> FaceGeometrySolver:
        key = (h, w)
        if key not in self._solvers:
            self._solvers[key] = FaceGeometrySolver(
                self._meta.canonical_vertices, self._meta.landmark_ids,
                self._meta.weights, w, h)
        return self._solvers[key]

    def close(self) -> None:
        pass
