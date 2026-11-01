"""引擎 C：MediaPipe Tasks API 基线/兜底（CPU，零额外驱动，方案书 4.1）。"""
from __future__ import annotations

import numpy as np

from .engine import EngineResult, PoseEngine


class EngineC(PoseEngine):
    name = "C"

    def __init__(self, model_path: str):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "mediapipe 未安装：引擎 C 依赖 mediapipe 1.2.0（Python 3.11）。"
                "见 requirements.lock 与方案书 11.2。"
            ) from e
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # 上游拼写即如此（方案书 附录 A 第 7 条）
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _to_mp_image(self, frame_bgr: np.ndarray):
        # BGR -> RGB（MediaPipe 期望 RGB），并保证连续内存。
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def self_check(self) -> None:
        # 冒烟推理 + 预热（方案书 5.3 引擎 C 自检：限时 10 s，此处由上层超时包裹）。
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._landmarker.detect_for_video(self._to_mp_image(dummy), timestamp_ms=0)
        self._landmarker.detect_for_video(self._to_mp_image(dummy), timestamp_ms=1)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EngineResult | None:
        mp_img = self._to_mp_image(frame_bgr)
        res = self._landmarker.detect_for_video(mp_img, timestamp_ms)
        if not res.face_landmarks or not res.facial_transformation_matrixes:
            return None
        lm = np.asarray(
            [[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32
        )
        t = np.asarray(
            res.facial_transformation_matrixes[0], dtype=np.float32
        ).reshape(4, 4)
        return EngineResult(landmarks=lm, transform=t)

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
