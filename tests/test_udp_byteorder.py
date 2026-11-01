"""OpenTrack 输出字节序回归测试（方案书 3.3 + 4.5 第 4 条）。

依据：opentrack 官方 issue #1203 抓包实测——48 字节 = 6 个 double，
顺序 X, Y, Z, Yaw, Pitch, Roll（位置在前、角度在后）。
"""
import socket
import struct
import unittest

from fylgja.pose import Pose
from fylgja.sink import OpenTrackUdpSink


class TestPacketLayout(unittest.TestCase):
    def test_struct_order_and_endianness(self):
        pose = Pose(yaw=10.0, pitch=20.0, roll=30.0, tx=1.5, ty=-2.5, tz=3.5)
        payload = OpenTrackUdpSink.PACKET.pack(*pose.as_opentrack_order())
        self.assertEqual(len(payload), 48)
        self.assertEqual(
            struct.unpack("<6d", payload),
            (1.5, -2.5, 3.5, 10.0, 20.0, 30.0),
        )

    def test_order_matches_opentrack_issue_1203(self):
        labels = ["X", "Y", "Z", "Yaw", "Pitch", "Roll"]
        pose = Pose(yaw=10, pitch=20, roll=30, tx=1, ty=2, tz=3)
        data = OpenTrackUdpSink.PACKET.pack(*pose.as_opentrack_order())
        got = {
            label: struct.unpack("d", data[i * 8:(i + 1) * 8])[0]
            for i, label in enumerate(labels)
        }
        self.assertEqual(
            got,
            {"X": 1.0, "Y": 2.0, "Z": 3.0, "Yaw": 10.0, "Pitch": 20.0, "Roll": 30.0},
        )


class TestLoopbackSend(unittest.TestCase):
    def test_wire_bytes_end_to_end(self):
        """通过本地回环 socket 验证真实线上字节（字段顺序 + 字节序）。"""
        recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv.bind(("127.0.0.1", 0))
        port = recv.getsockname()[1]
        recv.settimeout(2.0)

        sink = OpenTrackUdpSink("127.0.0.1", port)
        pose = Pose(yaw=10.0, pitch=20.0, roll=30.0, tx=1.5, ty=-2.5, tz=3.5)
        sink.send(pose)

        data, _ = recv.recvfrom(48)
        self.assertEqual(len(data), 48)
        self.assertEqual(
            struct.unpack("<6d", data),
            (1.5, -2.5, 3.5, 10.0, 20.0, 30.0),
        )
        self.assertEqual(sink.sent_count, 1)
        recv.close()
        sink.close()


if __name__ == "__main__":
    unittest.main()
