"""空间音频增益算法（fylgja.spatial_audio.GainController 纯逻辑）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.spatial_audio import GainConfig, gains_from_yaw  # noqa: E402


class TestGainsFromYaw(unittest.TestCase):
    def test_deadzone(self):
        cfg = GainConfig(deadzone_deg=5.0)
        self.assertEqual(gains_from_yaw(0, cfg), (1.0, 1.0))
        self.assertEqual(gains_from_yaw(5, cfg), (1.0, 1.0))
        self.assertEqual(gains_from_yaw(-5, cfg), (1.0, 1.0))

    def test_fixed_mode_turn_left_attenuates_left(self):
        # yaw 正 = 头转向其左侧 → 左耳衰减（固定声场）
        cfg = GainConfig(max_db=12.0, deadzone_deg=0.0)
        gl, gr = gains_from_yaw(45, cfg)
        self.assertLess(gl, 1.0)
        self.assertEqual(gr, 1.0)

    def test_fixed_mode_turn_right_attenuates_right(self):
        cfg = GainConfig(max_db=12.0, deadzone_deg=0.0)
        gl, gr = gains_from_yaw(-45, cfg)
        self.assertEqual(gl, 1.0)
        self.assertLess(gr, 1.0)

    def test_max_attenuation_clamps(self):
        cfg = GainConfig(max_db=12.0, deadzone_deg=0.0)
        gl, _ = gains_from_yaw(90, cfg)
        self.assertAlmostEqual(gl, 10 ** (-12 / 20), places=4)
        gl2, _ = gains_from_yaw(500, cfg)  # 超范围不放大
        self.assertAlmostEqual(gl2, 10 ** (-12 / 20), places=4)

    def test_midpoint_proportional(self):
        cfg = GainConfig(max_db=10.0, deadzone_deg=10.0)
        # yaw=50：over=40，span=80 → 5 dB
        gl, _ = gains_from_yaw(50, cfg)
        self.assertAlmostEqual(gl, 10 ** (-5 / 20), places=4)

    def test_never_boosts(self):
        cfg = GainConfig()
        for yaw in range(-180, 181, 15):
            gl, gr = gains_from_yaw(yaw, cfg)
            self.assertLessEqual(gl, 1.0)
            self.assertLessEqual(gr, 1.0)


if __name__ == "__main__":
    unittest.main()
