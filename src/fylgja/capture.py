"""采集线程 + 深度 1 最新帧信箱（方案书 5.1 / 7.1 / 6 步骤 5）。

契约要点：
- 打开摄像头按 DirectShow → Media Foundation → 默认后端顺序尝试；
  成功读取首帧后回读实际分辨率与帧率，与请求值一并记录（CameraInfo）。
- 深度 1 信箱：新帧覆盖旧帧，推理慢时丢帧不排队（方案书 5.1）。
- 时间戳由采集线程单一生成并严格单调递增（引擎 C VIDEO 模式强制要求）。
- 连续读帧失败 >= max_consecutive_failures（默认 30）时置 error 并退出，
  由上层报错（方案书 7.1；另一路"2 s 无新帧"守卫在推理线程侧）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

# 后端顺序：DirectShow → Media Foundation → 默认（方案书 6 步骤 5）
BACKEND_ORDER: tuple[str, ...] = ("dshow", "msmf", "default")

# 时间戳基准：保证大于引擎自检已消费的时间戳（0、1），避免违反单调递增契约。
_TS_BASE_MS = 1_000


class LatestFrameMailbox:
    """深度 1 信箱：只保留最新帧（方案书 5.1）。

    put 不拷贝：cap.read() 每次返回独立数组，覆写引用即可，
    已被推理线程取走的旧帧不会被改写。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._timestamp_ms: int = 0
        self._seq: int = 0

    def put(self, frame: np.ndarray, timestamp_ms: int) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp_ms = timestamp_ms
            self._seq += 1

    def get(self) -> tuple[np.ndarray | None, int, int]:
        """返回 (frame, timestamp_ms, seq)；seq 用于判断是否有新帧。"""
        with self._lock:
            return self._frame, self._timestamp_ms, self._seq


@dataclass
class CameraInfo:
    backend: str
    index: int
    requested: tuple[int, int, float]  # (width, height, fps)
    actual: tuple[int, int, float]  # (width, height, fps)，首帧后回读


class CaptureThread(threading.Thread):
    """采集线程：读帧 → 单调时间戳 → 信箱。"""

    def __init__(
        self,
        mailbox: LatestFrameMailbox,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        backend: str = "auto",
        max_consecutive_failures: int = 30,
    ) -> None:
        super().__init__(name="capture", daemon=True)
        if backend != "auto" and backend not in BACKEND_ORDER:
            raise ValueError(f"未知采集后端: {backend!r}，可选 auto|{'|'.join(BACKEND_ORDER)}")
        self._mailbox = mailbox
        self._index = camera_index
        self._requested = (width, height, fps)
        self._backend = backend
        self._max_failures = max_consecutive_failures
        self._stop_evt = threading.Event()
        self._cap = None
        self._cv2 = None
        self.info: CameraInfo | None = None
        self.frames_read = 0
        self.read_failures = 0
        self.error: str | None = None
        self._last_ts = _TS_BASE_MS

    def _open_attempts(self) -> list[str]:
        return [self._backend] if self._backend != "auto" else list(BACKEND_ORDER)

    def open(self) -> CameraInfo:
        """按后端顺序尝试打开；成功读首帧后回读实际参数（方案书 6 步骤 5）。

        失败抛 RuntimeError，由上层报错并回到摄像头选择（方案书 6 步骤 6）。
        """
        import cv2

        self._cv2 = cv2
        width, height, fps = self._requested
        tried: list[str] = []
        for name in self._open_attempts():
            backend_id = {
                "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF,
                "default": cv2.CAP_ANY,
            }[name]
            cap = cv2.VideoCapture(self._index, backend_id)
            if not cap.isOpened():
                cap.release()
                tried.append(f"{name}:open失败")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                tried.append(f"{name}:首帧超时/失败")
                continue
            # 首帧成功：回读实际采集参数（方案书 7.1）
            actual = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS)),
            )
            self._cap = cap
            self.info = CameraInfo(
                backend=name, index=self._index,
                requested=self._requested, actual=actual,
            )
            # 首帧也进信箱，时间戳走同一条生成路径
            self._put_frame(frame)
            return self.info
        raise RuntimeError(
            f"摄像头 {self._index} 打开失败（已试 {' -> '.join(self._open_attempts())}）："
            + "；".join(tried)
        )

    def _put_frame(self, frame: np.ndarray) -> int:
        self._last_ts += 1  # 严格单调递增（引擎 C VIDEO 契约）
        self._mailbox.put(frame, self._last_ts)
        return self._last_ts

    def run(self) -> None:
        if self._cap is None:
            raise RuntimeError("CaptureThread.run() 前必须先调用 open()")
        consecutive = 0
        while not self._stop_evt.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive += 1
                self.read_failures += 1
                if consecutive >= self._max_failures:
                    self.error = f"连续 {consecutive} 帧读取失败（方案书 7.1 阈值），管线已暂停"
                    break
                time.sleep(0.005)
                continue
            consecutive = 0
            self._put_frame(frame)
            self.frames_read += 1

    def stop(self) -> None:
        self._stop_evt.set()

    def close(self) -> None:
        self.stop()
        if self.is_alive():
            self.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
