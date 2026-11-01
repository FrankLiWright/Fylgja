"""姿态表示与 ZXY 欧拉分解。

OpenTrack 输出契约（方案书 3.3）：
  48 字节 = 6 个 little-endian double，顺序 TX、TY、TZ、yaw、pitch、roll
  （位置在前、角度在后；角度单位度、位置单位厘米）。

ZXY 分解约定（待阶段 0 与 MediaPipe 对齐，方案书 4.5 第 3/4 条）：
  R = Rz(roll) @ Rx(pitch) @ Ry(yaw)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Pose:
    """头部姿态。角度单位度，平移单位厘米。"""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def as_opentrack_order(self) -> tuple[float, float, float, float, float, float]:
        """按 OpenTrack 字段顺序返回 (TX, TY, TZ, yaw, pitch, roll)。"""
        return (self.tx, self.ty, self.tz, self.yaw, self.pitch, self.roll)

    def copy(self) -> "Pose":
        return Pose(self.yaw, self.pitch, self.roll, self.tx, self.ty, self.tz)


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """由 ZXY 欧拉角构造 3×3 旋转矩阵 R = Rz(roll) @ Rx(pitch) @ Ry(yaw)。"""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rz @ rx @ ry


def euler_zxy(r: np.ndarray) -> tuple[float, float, float]:
    """ZXY 分解，返回 (yaw, pitch, roll)，单位度。r 为 3×3 旋转矩阵。

    推导（R = Rz·Rx·Ry）：
      pitch = asin(R[2,1])
      roll  = atan2(-R[0,1], R[1,1])
      yaw   = atan2(-R[2,0], R[2,2])
    """
    r = np.asarray(r, dtype=float)
    pitch = math.degrees(math.asin(np.clip(r[2, 1], -1.0, 1.0)))
    roll = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
    yaw = math.degrees(math.atan2(-r[2, 0], r[2, 2]))
    return yaw, pitch, roll


def pose_from_transform(t: np.ndarray) -> Pose:
    """从 4×4 变换矩阵解出 Pose。

    平移直接取最后一列（原始度量单位，需经 postprocess 标定换算为厘米）。
    """
    t = np.asarray(t, dtype=float)
    yaw, pitch, roll = euler_zxy(t[:3, :3])
    return Pose(
        yaw=yaw, pitch=pitch, roll=roll,
        tx=float(t[0, 3]), ty=float(t[1, 3]), tz=float(t[2, 3]),
    )
