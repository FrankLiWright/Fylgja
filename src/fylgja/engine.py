"""PoseEngine 接口与同构输出类型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EngineResult:
    """引擎同构输出：478 关键点 + 4×4 变换矩阵。

    后处理管线不得感知引擎来源（方案书 5.2/7.2）。
    """
    landmarks: np.ndarray  # shape (478, 3)
    transform: np.ndarray  # shape (4, 4)
    detect_ms: float = 0.0  # 引擎 N：检测耗时
    npu_ms: float = 0.0     # 引擎 N：NPU 关键点耗时
    solve_ms: float = 0.0   # 引擎 N：求解器耗时
    total_ms: float = 0.0   # 总推理耗时


class PoseEngine:
    """推理引擎接口。实现：EngineC（MediaPipe）、EngineN（OpenVINO）。"""

    name = "abstract"

    def self_check(self) -> None:
        """推理引擎自检（方案书 5.3）。失败必须抛异常，不得静默返回。"""
        raise NotImplementedError

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EngineResult | None:
        """单帧推理；未检出人脸时返回 None（由上层按方案书 7.2 计数）。"""
        raise NotImplementedError

    def close(self) -> None:
        pass
