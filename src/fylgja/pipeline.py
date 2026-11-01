"""推理线程（方案书 5.1 / 7.2）：信箱 → PoseEngine → 后处理 → PoseSink。

- 信箱无新帧时短暂休眠；超过 stall_timeout_s（默认 2 s，方案书 7.1）无新帧
  视为采集停滞，置 error 并退出。
- 推理异常/结果非法连续 max_consecutive_failures 次（默认 10，方案书 7.2）
  时暂停管线；运行期不得静默换引擎（方案书 7.2）。
- 每帧记录推理耗时，供状态栏与验收测量（方案书 7.2）。
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .capture import LatestFrameMailbox
from .engine import EngineResult, PoseEngine
from .postprocess import PostProcessor
from .sink import PoseSink


def _result_valid(res: EngineResult) -> bool:
    """结果非法判定（方案书 7.2）：形状不对或含非有限值。"""
    lm = np.asarray(res.landmarks)
    tr = np.asarray(res.transform)
    return (
        lm.shape == (478, 3)
        and tr.shape == (4, 4)
        and np.isfinite(lm).all()
        and np.isfinite(tr).all()
    )


class InferenceThread(threading.Thread):
    """推理线程：消费最新帧信箱，产出姿态到 PoseSink。"""

    def __init__(
        self,
        mailbox: LatestFrameMailbox,
        engine: PoseEngine,
        postprocessor: PostProcessor,
        sink: PoseSink,
        stall_timeout_s: float = 2.0,
        max_consecutive_failures: int = 10,
    ) -> None:
        super().__init__(name="inference", daemon=True)
        self._mailbox = mailbox
        self._engine = engine
        self._postprocessor = postprocessor
        self._sink = sink
        self._stall_timeout_s = stall_timeout_s
        self._max_failures = max_consecutive_failures
        self._stop_evt = threading.Event()
        self.error: str | None = None
        # 统计（方案书 7.2 / 8.2）
        self.frames_processed = 0
        self.no_face_count = 0
        self.inference_failures = 0
        self.infer_ms_last = 0.0
        self._infer_ms_sum = 0.0
        # 预览契约（8.1）：以下即"实际发送值"，图形化朝向必须以此绘制
        self.last_sent: tuple | None = None        # (tx,ty,tz,yaw,pitch,roll)
        self.last_landmarks: np.ndarray | None = None
        self.last_raw_euler: tuple | None = None   # 滤波前（8.2 状态栏对照）

    @property
    def infer_ms_avg(self) -> float:
        n = self.frames_processed + self.no_face_count + self.inference_failures
        return self._infer_ms_sum / n if n else 0.0

    def run(self) -> None:
        last_seq = 0
        last_new = time.monotonic()
        while not self._stop_evt.is_set():
            frame, ts, seq = self._mailbox.get()
            now = time.monotonic()
            if frame is None or seq == last_seq:
                if now - last_new > self._stall_timeout_s:
                    self.error = (
                        f"超过 {self._stall_timeout_s:.0f}s 无新帧（采集停滞，方案书 7.1），管线已暂停"
                    )
                    break
                self._stop_evt.wait(0.002)
                continue
            last_seq = seq
            last_new = now

            t0 = time.perf_counter()
            try:
                result = self._engine.process(frame, ts)
                self.infer_ms_last = (time.perf_counter() - t0) * 1000.0
                self._infer_ms_sum += self.infer_ms_last

                if result is None:
                    self.no_face_count += 1  # 未检出人脸（方案书 7.2 由上层计数）
                    continue
                if not _result_valid(result):
                    raise ValueError("结果形状/数值非法")

                # 后处理（滤波 → 标定 → 回中 → 轴反转，两引擎共用）
                pose = self._postprocessor.process(result.transform, t=ts / 1000.0)
                self._sink.send(pose)
                self.frames_processed += 1
                self.inference_failures = 0
                # 预览契约：记录实际发送值与关键点（8.1）
                from .pose import pose_from_transform
                raw = pose_from_transform(result.transform)
                self.last_raw_euler = (raw.yaw, raw.pitch, raw.roll)
                self.last_sent = pose.as_opentrack_order()
                self.last_landmarks = result.landmarks
            except Exception as e:  # 引擎抛出/结果非法/后处理异常：计数并按 7.2 处置
                self.inference_failures += 1
                if self.inference_failures >= self._max_failures:
                    self.error = (
                        f"推理连续失败 {self.inference_failures} 次（方案书 7.2）："
                        f"{type(e).__name__}: {e}"
                    )
                    break

    def stop(self) -> None:
        self._stop_evt.set()
