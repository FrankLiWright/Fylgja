"""摄像头枚举与解析（方案书 6 步骤 3）：path 优先、丢失显式报错、空列表报错。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.camera_enum import (  # noqa: E402
    CameraDevice, enumerate_dshow_devices, resolve_camera,
)
from fylgja.errors import CameraError  # noqa: E402

FAKE = [
    CameraDevice(0, "Cam A", "@device:pnp:\\\\?\\usb#vid_a"),
    CameraDevice(1, "Cam B", "@device:sw:{bb}"),
    CameraDevice(2, "Cam C", "@device:pnp:\\\\?\\usb#vid_c"),
]


class TestResolve(unittest.TestCase):
    def test_path_match_wins_over_index(self):
        dev = resolve_camera(FAKE, path="@device:sw:{bb}", index=0)
        self.assertEqual(dev.name, "Cam B")
        self.assertEqual(dev.index, 1)

    def test_index_fallback_when_no_path(self):
        dev = resolve_camera(FAKE, path="", index=2)
        self.assertEqual(dev.name, "Cam C")

    def test_missing_path_is_explicit_error(self):
        """配置的设备丢失必须显式报错，不得静默换设备（方案书 6）。"""
        with self.assertRaises(CameraError) as ctx:
            resolve_camera(FAKE, path="@device:gone", index=0)
        self.assertIn("已丢失", str(ctx.exception))

    def test_bad_index_lists_available(self):
        with self.assertRaises(CameraError) as ctx:
            resolve_camera(FAKE, index=99)
        self.assertIn("Cam A", str(ctx.exception))

    def test_empty_enumeration(self):
        """枚举不到设备 → 指向相机隐私/占用（方案书 8.3）。"""
        with self.assertRaises(CameraError) as ctx:
            resolve_camera([], index=0)
        self.assertIn("隐私", str(ctx.exception))


class TestRealEnumeration(unittest.TestCase):
    """本机集成：DirectShow 枚举（有摄像头的机器上验证）。"""

    def test_dshow_enumeration_shape(self):
        try:
            devices = enumerate_dshow_devices()
        except CameraError:
            self.skipTest("comtypes 不可用")
        for d in devices:
            self.assertIsInstance(d, CameraDevice)
            self.assertTrue(d.name)
            self.assertTrue(d.path.startswith("@device:"))
            self.assertEqual(d.backend, "dshow")
        idx = [d.index for d in devices]
        self.assertEqual(idx, sorted(idx))  # 索引连续有序
        self.assertEqual(len(set(idx)), len(idx))


if __name__ == "__main__":
    unittest.main()
