"""后处理管线顺序回归（滤波 → 标定 → 回中 → 轴反转，方案书 7.3）。"""
import unittest

import numpy as np

from fylgja.pose import rotation_matrix
from fylgja.postprocess import Calibration, PostProcessor


def make_transform(yaw=0.0, pitch=0.0, roll=0.0, tx=0.0, ty=0.0, tz=0.0):
    t = np.eye(4)
    t[:3, :3] = rotation_matrix(yaw, pitch, roll)
    t[:3, 3] = [tx, ty, tz]
    return t


class TestPostProcess(unittest.TestCase):
    def test_axis_inversion_after_everything(self):
        # 反转载荷 pitch 通道：输入 pitch=20°，首帧直通后应输出 -20°。
        pp = PostProcessor(axis_invert=(False, True, False, False, False, False))
        out = pp.process(make_transform(pitch=20.0), t=0.0)
        self.assertAlmostEqual(out.pitch, -20.0, places=5)
        self.assertAlmostEqual(out.yaw, 0.0, places=5)

    def test_calibration_scales_translation_only(self):
        pp = PostProcessor(calibration=Calibration(scale=2.0))
        out = pp.process(make_transform(tx=1.0, ty=2.0, tz=3.0), t=0.0)
        self.assertAlmostEqual(out.tx, 2.0, places=5)
        self.assertAlmostEqual(out.ty, 4.0, places=5)
        self.assertAlmostEqual(out.tz, 6.0, places=5)
        self.assertAlmostEqual(out.yaw, 0.0, places=5)

    def test_recenter_zeroes_current_pose(self):
        pp = PostProcessor()
        t = make_transform(yaw=10.0, pitch=5.0)
        for ts in (0.0, 0.033, 0.066, 0.099, 0.132):
            pp.process(t, t=ts)
        pp.recenter()
        out = pp.process(t, t=0.165)
        self.assertAlmostEqual(out.yaw, 0.0, places=4)
        self.assertAlmostEqual(out.pitch, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
