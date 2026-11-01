"""One Euro（1€）滤波器，每通道独立。

参考 G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012)。
NOTICE 需致谢算法出处（方案书 11.3）。
"""
from __future__ import annotations

import math


class _LowPass:
    """指数平滑，alpha 随采样间隔自适应。"""

    def __init__(self, cutoff: float):
        self.cutoff = float(cutoff)
        self._y: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._y = None
        self._last_t = None

    def filter(self, x: float, t: float) -> float:
        if self._y is None or self._last_t is None:
            self._y = x
            self._last_t = t
            return self._y
        dt = max(t - self._last_t, 1e-6)
        tau = 1.0 / (2.0 * math.pi * self.cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        self._y = alpha * x + (1.0 - alpha) * self._y
        self._last_t = t
        return self._y


class OneEuroFilter:
    """单通道 1€ 滤波器。默认参数与方案书 7.3 一致。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass(self.min_cutoff)
        self._dx = _LowPass(self.d_cutoff)
        self._prev_raw: float | None = None
        self._prev_t: float | None = None

    def reset(self) -> None:
        """复位全部状态（回中时必须调用，方案书 7.3）。"""
        self._x.reset()
        self._dx.reset()
        self._prev_raw = None
        self._prev_t = None

    def filter(self, x: float, t: float) -> float:
        if self._prev_raw is None:
            self._prev_raw = x
            self._prev_t = t
            self._x.filter(x, t)
            self._dx.filter(0.0, t)
            return x

        dt = max(t - self._prev_t, 1e-6)
        dx = (x - self._prev_raw) / dt
        edx = self._dx.filter(dx, t)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        self._x.cutoff = cutoff
        y = self._x.filter(x, t)
        self._prev_raw = x
        self._prev_t = t
        return y


class OneEuroFilterND:
    """多通道向量化 1€ 滤波（引擎 N 关键点平滑）。

    与 MediaPipe LandmarksSmoothingCalculator 同参数（VIDEO 模式默认启用）：
    min_cutoff=0.05, beta=80.0, derivate_cutoff=1.0。逐元素独立、公式同上。
    """

    def __init__(self, shape, min_cutoff: float = 0.05, beta: float = 80.0,
                 d_cutoff: float = 1.0):
        import numpy as np
        self._np = np
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._y = None
        self._dy = None
        self._prev_raw = None
        self._prev_t = None
        self._shape = tuple(shape)

    @staticmethod
    def _alpha(cutoff, dt, np):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._y = None
        self._dy = None
        self._prev_raw = None
        self._prev_t = None

    def filter(self, x, t: float):
        np = self._np
        x = np.asarray(x, dtype=np.float64)
        if self._prev_raw is None:
            self._y = x.copy()
            self._dy = np.zeros_like(x)
            self._prev_raw = x.copy()
            self._prev_t = t
            return x
        dt = max(t - self._prev_t, 1e-6)
        dx = (x - self._prev_raw) / dt
        a_d = self._alpha(self.d_cutoff, dt, np)
        self._dy = a_d * dx + (1.0 - a_d) * self._dy
        cutoff = self.min_cutoff + self.beta * np.abs(self._dy)
        a_x = self._alpha(cutoff, dt, np)
        self._y = a_x * x + (1.0 - a_x) * self._y
        self._prev_raw = x.copy()
        self._prev_t = t
        return self._y
