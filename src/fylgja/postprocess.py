"""后处理管线（两引擎共用）：滤波 → 标定 → 回中偏移 → 轴反转。

方案书 7.3：这是两引擎精度对齐验收（13.6）成立的结构保证。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .filter import OneEuroFilter
from .pose import Pose, pose_from_transform


@dataclass
class Calibration:
    """平移标定：规范化平移量 → 厘米（方案书 3.3/4.5 第 6 条）。"""
    scale: float = 1.0  # 默认出厂系数，阶段 0 标定实验回填
    tx0: float = 0.0
    ty0: float = 0.0
    tz0: float = 0.0


class PostProcessor:
    """六个通道各自滤波；旋转不做标定，平移做标定换算。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5,
                 d_cutoff: float = 1.0, calibration: Calibration | None = None,
                 axis_invert=(False, False, False, False, False, False)):
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(6)]
        self.calibration = calibration or Calibration()
        # 顺序 (yaw, pitch, roll, tx, ty, tz)
        self.axis_invert = list(axis_invert)
        self._recenter = Pose()
        self._last_output = Pose()
        self._t: float | None = None
        self._lock = threading.Lock()  # UI 回中 与 推理线程 process 并发（5.1）

    def reset_filters(self) -> None:
        for f in self.filters:
            f.reset()

    def recenter(self) -> None:
        """回中：以当前「滤波+标定后」姿态为零点，并复位滤波器导数（方案书 7.3）。"""
        with self._lock:
            self._recenter = self._last_output.copy()
            self.reset_filters()

    def process(self, transform: np.ndarray, t: float) -> Pose:
        with self._lock:
            return self._process_locked(transform, t)

    def _process_locked(self, transform: np.ndarray, t: float) -> Pose:
        raw = pose_from_transform(transform)
        vals = [raw.yaw, raw.pitch, raw.roll, raw.tx, raw.ty, raw.tz]

        if self._t is None:
            self._t = t
        # 1) 滤波
        filtered = [f.filter(v, t) for f, v in zip(self.filters, vals)]

        # 2) 标定（仅平移）
        cal = self.calibration
        tx = filtered[3] * cal.scale + cal.tx0
        ty = filtered[4] * cal.scale + cal.ty0
        tz = filtered[5] * cal.scale + cal.tz0

        # 3) 回中偏移
        out = Pose(
            yaw=filtered[0] - self._recenter.yaw,
            pitch=filtered[1] - self._recenter.pitch,
            roll=filtered[2] - self._recenter.roll,
            tx=tx - self._recenter.tx,
            ty=ty - self._recenter.ty,
            tz=tz - self._recenter.tz,
        )

        # 4) 轴反转（滤波后、输出前，方案书 7.3）
        inv = self.axis_invert
        out.yaw = -out.yaw if inv[0] else out.yaw
        out.pitch = -out.pitch if inv[1] else out.pitch
        out.roll = -out.roll if inv[2] else out.roll
        out.tx = -out.tx if inv[3] else out.tx
        out.ty = -out.ty if inv[4] else out.ty
        out.tz = -out.tz if inv[5] else out.tz

        self._last_output = out
        return out
