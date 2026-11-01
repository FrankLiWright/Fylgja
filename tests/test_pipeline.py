"""推理线程行为（方案书 7.2）：产出、未检出计数、失败阈值、停滞守卫。"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.capture import LatestFrameMailbox  # noqa: E402
from fylgja.engine import EngineResult  # noqa: E402
from fylgja.pipeline import InferenceThread  # noqa: E402
from fylgja.postprocess import PostProcessor  # noqa: E402
from fylgja.pose import Pose  # noqa: E402

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


class RecordingSink:
    def __init__(self):
        self.poses: list[Pose] = []

    def send(self, pose: Pose) -> None:
        self.poses.append(pose)


class FakeEngine:
    name = "fake"

    def __init__(self, result="ok", calls=None):
        self._result = result  # ok | none | invalid | raise
        self.calls = calls if calls is not None else []

    def process(self, frame, timestamp_ms):
        self.calls.append(timestamp_ms)
        if self._result == "raise":
            raise RuntimeError("boom")
        if self._result == "none":
            return None
        lm = np.zeros((478, 3), dtype=np.float32)
        if self._result == "invalid":
            return EngineResult(landmarks=lm, transform=np.full((3, 3), np.nan))
        return EngineResult(landmarks=lm, transform=np.eye(4))


def _run(thread: InferenceThread, n_frames: int, engine_calls: list, timeout=3.0) -> None:
    """喂 n_frames 帧（深度 1 信箱，必须逐帧同步）。"""
    deadline = time.monotonic() + timeout
    ts = 1001
    for _ in range(n_frames):
        before = len(engine_calls)
        thread._mailbox.put(FRAME, ts)
        ts += 1
        while len(engine_calls) == before and time.monotonic() < deadline:
            time.sleep(0.005)


class TestInferenceThread(unittest.TestCase):
    def test_process_and_send(self):
        sink = RecordingSink()
        engine = FakeEngine()
        mb = LatestFrameMailbox()
        th = InferenceThread(mb, engine, postprocessor=PostProcessor(), sink=sink,
                             stall_timeout_s=5.0)
        th.start()
        try:
            _run(th, 1, engine.calls)
            deadline = time.monotonic() + 2.0
            while not sink.poses and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            th.stop()
            th.join(2.0)
        self.assertEqual(len(sink.poses), 1)
        self.assertEqual(th.frames_processed, 1)

    def test_no_face_counted(self):
        engine = FakeEngine(result="none")
        mb = LatestFrameMailbox()
        th = InferenceThread(mb, engine, postprocessor=PostProcessor(),
                             sink=RecordingSink(), stall_timeout_s=5.0)
        th.start()
        try:
            _run(th, 1, engine.calls)
            deadline = time.monotonic() + 2.0
            while th.no_face_count == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            th.stop()
            th.join(2.0)
        self.assertEqual(th.no_face_count, 1)
        self.assertEqual(th.frames_processed, 0)

    def test_failure_threshold_stops(self):
        engine = FakeEngine(result="raise")
        mb = LatestFrameMailbox()
        th = InferenceThread(mb, engine, postprocessor=PostProcessor(),
                             sink=RecordingSink(), stall_timeout_s=5.0,
                             max_consecutive_failures=3)
        th.start()
        try:
            _run(th, 5, engine.calls)
            deadline = time.monotonic() + 2.0
            while th.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            th.stop()
            th.join(2.0)
        self.assertIsNotNone(th.error)
        self.assertIn("3", th.error)
        self.assertLessEqual(th.inference_failures, 5)

    def test_invalid_result_counted_as_failure(self):
        engine = FakeEngine(result="invalid")
        mb = LatestFrameMailbox()
        th = InferenceThread(mb, engine, postprocessor=PostProcessor(),
                             sink=RecordingSink(), stall_timeout_s=5.0,
                             max_consecutive_failures=2)
        th.start()
        try:
            _run(th, 3, engine.calls)
            deadline = time.monotonic() + 2.0
            while th.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            th.stop()
            th.join(2.0)
        self.assertIsNotNone(th.error)
        self.assertIn("非法", th.error)

    def test_stall_guard(self):
        mb = LatestFrameMailbox()
        th = InferenceThread(mb, FakeEngine(), postprocessor=PostProcessor(),
                             sink=RecordingSink(), stall_timeout_s=0.15)
        th.start()
        try:
            deadline = time.monotonic() + 2.0
            while th.error is None and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            th.stop()
            th.join(2.0)
        self.assertIsNotNone(th.error)
        self.assertIn("停滞", th.error)


if __name__ == "__main__":
    unittest.main()
