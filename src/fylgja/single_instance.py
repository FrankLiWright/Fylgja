"""单实例互斥体（方案书 5.4）。

启动时创建命名互斥体；二次启动不得打开第二个管线实例，
而是激活已有窗口后退出（无 GUI 阶段：提示并退出）。
"""
from __future__ import annotations

import ctypes
import sys

ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """命名互斥体。acquire() 返回 False 表示已有实例在运行。"""

    def __init__(self, name: str = "Fylgja-single-instance"):
        self._name = name
        self._handle = None

    def acquire(self) -> bool:
        if sys.platform != "win32":  # 非 Windows（开发/测试）不设互斥
            return True
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateMutexW(None, False, f"Local\\{self._name}")
        self._handle = handle
        if handle == 0:
            return True  # 创建失败不阻塞启动；记日志由调用方决定
        return kernel32.GetLastError() != ERROR_ALREADY_EXISTS

    def activate_existing(self, window_title: str | None = None) -> bool:
        """激活已有窗口（方案书 5.4）。找到并前置成功返回 True。"""
        if sys.platform != "win32" or not window_title:
            return False
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        # 简化还原并前置（跨进程 SetForegroundWindow 有限制，尽力而为）
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        return True

    def release(self) -> None:
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None
