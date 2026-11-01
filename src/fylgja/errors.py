"""归因错误类型（方案书 8.3 分层诊断 / 12.2 归因边界）。

每个失败必须落到确定区段，文案互不相同（13.17）：
安装包完整性 / 系统环境 / NPU 驱动 / NPU 编译 / 摄像头 / 其他。
"""
from __future__ import annotations

from enum import Enum


class Attribution(str, Enum):
    """归因区段（方案书 8.3 表格的'归因区段'列）。"""

    PACKAGE = "安装包完整性"       # 模型缺失/损坏、安装包不完整
    SYSTEM = "系统环境"           # VC++ 运行库等
    NPU_DRIVER = "NPU 驱动"       # NPU 未检测到/驱动过旧
    NPU_COMPILE = "NPU 编译"      # 编译失败
    CAMERA = "摄像头"             # 枚举不到/打开失败/被占用
    OTHER = "其他"


class SelfCheckError(RuntimeError):
    """推理引擎自检失败（方案书 5.3）：带归因区段，不得静默返回。"""

    def __init__(self, attribution: Attribution, message: str):
        super().__init__(f"[{attribution.value}] {message}")
        self.attribution = attribution


class CameraError(RuntimeError):
    """摄像头枚举/打开失败（方案书 6 步骤 3-6）。"""

    def __init__(self, message: str, attribution: Attribution = Attribution.CAMERA):
        super().__init__(f"[{attribution.value}] {message}")
        self.attribution = attribution
