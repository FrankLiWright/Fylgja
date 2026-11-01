"""通用摄像头头部追踪软件 — 核心库。

架构见《头部追踪软件工程方案书》：
  PoseEngine (engine N / engine C) → 478 关键点 + 4×4 变换矩阵
  → 欧拉分解(ZXY) → One Euro 滤波 → 标定 → 回中偏移 → 轴反转 → PoseSink。
"""

__version__ = "0.1.0"
