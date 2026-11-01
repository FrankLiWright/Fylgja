"""ZXY 欧拉分解回归测试（方案书 4.5 第 4 条）。"""
import unittest

import numpy as np

from fylgja.pose import euler_zxy, pose_from_transform, rotation_matrix


def close(a, b, tol=1e-5):
    return abs(a - b) < tol


class TestEulerZxy(unittest.TestCase):
    def test_identity(self):
        y, p, r = euler_zxy(np.eye(3))
        self.assertTrue(close(y, 0) and close(p, 0) and close(r, 0))

    def test_yaw_only(self):
        r = rotation_matrix(30.0, 0.0, 0.0)
        y, p, rl = euler_zxy(r)
        self.assertTrue(close(y, 30.0))
        self.assertTrue(close(p, 0.0))
        self.assertTrue(close(rl, 0.0))

    def test_pitch_only_sign(self):
        r = rotation_matrix(0.0, -15.0, 0.0)
        y, p, rl = euler_zxy(r)
        self.assertTrue(close(p, -15.0))
        self.assertTrue(close(y, 0.0))
        self.assertTrue(close(rl, 0.0))

    def test_roll_only(self):
        r = rotation_matrix(0.0, 0.0, 25.0)
        y, p, rl = euler_zxy(r)
        self.assertTrue(close(rl, 25.0))
        self.assertTrue(close(y, 0.0))
        self.assertTrue(close(p, 0.0))

    def test_roundtrip_combined(self):
        for yaw, pitch, roll in [(10, -5, 3), (-20, 15, -8), (25, 20, -10)]:
            r = rotation_matrix(yaw, pitch, roll)
            y, p, rl = euler_zxy(r)
            self.assertTrue(close(y, yaw, 1e-4), f"yaw {yaw} != {y}")
            self.assertTrue(close(p, pitch, 1e-4), f"pitch {pitch} != {p}")
            self.assertTrue(close(rl, roll, 1e-4), f"roll {roll} != {rl}")


class TestTransformToPose(unittest.TestCase):
    def test_translation_passthrough(self):
        t = np.eye(4)
        t[:3, 3] = [1.0, -2.0, 3.0]
        pose = pose_from_transform(t)
        self.assertTrue(close(pose.tx, 1.0))
        self.assertTrue(close(pose.ty, -2.0))
        self.assertTrue(close(pose.tz, 3.0))


if __name__ == "__main__":
    unittest.main()
