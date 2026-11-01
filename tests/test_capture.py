"""信箱语义与采集时间戳单调性（方案书 5.1）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.capture import CaptureThread, LatestFrameMailbox  # noqa: E402


class TestMailbox(unittest.TestCase):
    def test_depth1_overwrite(self):
        mb = LatestFrameMailbox()
        f1 = np.ones((2, 2, 3), dtype=np.uint8)
        f2 = np.full((2, 2, 3), 2, dtype=np.uint8)
        mb.put(f1, 1001)
        mb.put(f2, 1002)  # 新帧覆盖旧帧，不排队
        frame, ts, seq = mb.get()
        self.assertIs(frame, f2)
        self.assertEqual(ts, 1002)
        self.assertEqual(seq, 2)

    def test_seq_tracks_newness(self):
        mb = LatestFrameMailbox()
        self.assertIsNone(mb.get()[0])
        mb.put(np.zeros((1, 1, 3), dtype=np.uint8), 1)
        _, _, seq1 = mb.get()
        _, _, seq2 = mb.get()
        self.assertEqual(seq1, seq2)  # 无新帧时 seq 不变


class TestTimestamps(unittest.TestCase):
    def test_strictly_increasing(self):
        mb = LatestFrameMailbox()
        cap = CaptureThread(mb)  # 不 open，仅测时间戳生成路径
        ts_list = [cap._put_frame(np.zeros((1, 1, 3), dtype=np.uint8)) for _ in range(5)]
        self.assertEqual(ts_list, sorted(ts_list))
        self.assertEqual(len(set(ts_list)), 5)  # 严格递增
        self.assertGreater(ts_list[0], 1)  # 必须大于引擎自检已用时间戳（0、1）

    def test_backend_validation(self):
        with self.assertRaises(ValueError):
            CaptureThread(LatestFrameMailbox(), backend="gstreamer")


if __name__ == "__main__":
    unittest.main()
