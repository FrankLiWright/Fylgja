"""MediaPipe face_geometry 流水线移植（方案书 4.3 第 3 条自建求解器）。

忠实移植 geometry_pipeline.cc / procrustes_solver.cc 的 numpy 实现
（参考：Rassibassi/mediapipeDemos face_geometry.py、geaxgx/depthai_handface 移植，
均为 MediaPipe 源码的 Apache-2.0 衍生；NOTICE 待打包阶段补充署名）。

算法（与源码注释一致）：
  1. ProjectXY：归一化屏幕坐标 → 近平面（y := 1−y 后缩放；z×x_scale）
  2. 翻手性（z→−z）后 Procrustes ① → s1（屏幕点不可反投影，先估尺度）
  3. 用 s1 反投影 → 中间度量点 → 翻手性 → Procrustes ② → s2
  4. 用 **s1×s2** 再次反投影 → 度量地标 → 翻手性
  5. Procrustes ③：规范点 ↔ 度量地标 → 位姿变换矩阵（4×4，含尺度项）

注意：屏幕空间为左手系、度量空间为右手系，跨"屏幕→度量"语义边界时必须翻手性
（z 取反），顺序为「反投影之后、Procrustes 之前」。
"""
from __future__ import annotations

import math

import numpy as np

VERTICAL_FOV_DEG = 63.0
NEAR_PLANE_CM = 1.0


def solve_weighted_orthogonal(src: np.ndarray, dst: np.ndarray,
                              w: np.ndarray) -> np.ndarray:
    """加权正交问题（含尺度）：求 4×4 使 dst ≈ s·R·src + t（procrustes_solver.cc）。

    src/dst: (3, N)；w: (N,) 非负。返回 4×4（3×3 = s·R）。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sqrt_w = np.sqrt(w)
    total_weight = float(np.sum(w))
    if total_weight <= 0:
        raise ValueError("权重和为零")

    weighted_src = src * sqrt_w[None, :]
    weighted_dst = dst * sqrt_w[None, :]

    # 仅源点集按加权质心去心（目标集去心与之代数等价）
    src_cm = np.sum(weighted_src * sqrt_w[None, :], axis=1) / total_weight
    centered_weighted_src = weighted_src - src_cm[:, None] * sqrt_w[None, :]

    design = weighted_dst @ centered_weighted_src.T
    u, _, vh = np.linalg.svd(design)
    postrotation, prerotation = u, vh
    if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0:
        postrotation[:, 2] *= -1  # 禁反射
    rotation = postrotation @ prerotation

    rotated = rotation @ centered_weighted_src
    numerator = float(np.sum(rotated * weighted_dst))
    denominator = float(np.sum(centered_weighted_src * weighted_src))
    scale = numerator / denominator if denominator > 0 else 1.0

    rotation_and_scale = scale * rotation
    diffs = weighted_dst - rotation_and_scale @ weighted_src
    translation = np.sum(diffs * sqrt_w[None, :], axis=1) / total_weight

    m = np.eye(4)
    m[:3, :3] = rotation_and_scale
    m[:3, 3] = translation
    return m


class FaceGeometrySolver:
    """ScreenToMetricSpaceConverter + 位姿 Procrustes 的完整移植。"""

    def __init__(self, canonical_vertices: np.ndarray, landmark_ids: np.ndarray,
                 weights: np.ndarray, frame_w: int, frame_h: int):
        self.canonical = np.asarray(canonical_vertices, dtype=np.float64).T  # (3, N)
        self.ids = np.asarray(landmark_ids, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.frame_w = frame_w
        self.frame_h = frame_h
        # 虚拟相机近平面（geometry_pipeline.cc view frustum）
        self.height_near = 2.0 * NEAR_PLANE_CM * math.tan(
            0.5 * math.radians(VERTICAL_FOV_DEG))
        self.width_near = frame_w * self.height_near / frame_h
        self.left = -0.5 * self.width_near
        self.bottom = -0.5 * self.height_near

    # ---- 单步变换（公式与参考实现逐行一致）----

    def _project_xy(self, lm: np.ndarray) -> np.ndarray:
        out = lm.copy()
        out[1, :] = 1.0 - out[1, :]  # y 翻转（TOP_LEFT_CORNER）
        scales = np.array([[self.width_near, self.height_near,
                            self.width_near]]).T
        offsets = np.array([[self.left, self.bottom, 0.0]]).T
        return out * scales + offsets

    @staticmethod
    def _change_handedness(lm: np.ndarray) -> np.ndarray:
        out = lm.copy()
        out[2, :] *= -1.0
        return out

    def _move_and_rescale_z(self, lm: np.ndarray, depth_offset: float,
                            scale: float) -> np.ndarray:
        out = lm.copy()
        out[2, :] = (out[2, :] - depth_offset + NEAR_PLANE_CM) / scale
        return out

    def _unproject_xy(self, lm: np.ndarray) -> np.ndarray:
        out = lm.copy()
        out[0, :] = out[0, :] * out[2, :] / NEAR_PLANE_CM
        out[1, :] = out[1, :] * out[2, :] / NEAR_PLANE_CM
        return out

    def _estimate_scale(self, lm: np.ndarray) -> float:
        m = solve_weighted_orthogonal(
            self.canonical[:, self.ids], lm[:, self.ids], self.weights)
        return float(np.linalg.norm(m[:, 0]))

    # ---- 主流程 ----

    def solve(self, landmarks_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """输入 (N,3) 归一化屏幕地标（x,y∈[0,1]，z 与 x 同尺度）。

        返回 (pose_4x4, metric_landmarks (N,3))：pose 为 canonical→metric 变换。
        """
        lm = np.asarray(landmarks_norm, dtype=np.float64).T  # (3, N)
        screen = self._project_xy(lm)
        depth_offset = float(np.mean(screen[2, :]))

        # (2)(3) 第一轮：翻手性后估尺度（屏幕点不反投影）
        inter = self._change_handedness(screen)
        s1 = self._estimate_scale(inter)

        # (4)(5) 第二轮：s1 反投影 → 中间度量 → 翻手性 → 估尺度
        inter = self._move_and_rescale_z(screen, depth_offset, s1)
        inter = self._unproject_xy(inter)
        inter = self._change_handedness(inter)
        s2 = self._estimate_scale(inter)

        # (6) 总尺度 s1×s2 最终反投影 → 度量地标
        metric = self._move_and_rescale_z(screen, depth_offset, s1 * s2)
        metric = self._unproject_xy(metric)
        metric = self._change_handedness(metric)

        # 位姿 Procrustes
        pose = solve_weighted_orthogonal(
            self.canonical[:, self.ids], metric[:, self.ids], self.weights)
        return pose, metric.T
