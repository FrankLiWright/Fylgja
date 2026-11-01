"""PoseSink 接口与 OpenTrack UDP 默认实现。

输出契约（方案书 3.3，已修正字段顺序）：
  48 字节 = 6 个本机字节序（Windows x86-64 即 little-endian）double，
  顺序 TX, TY, TZ, yaw, pitch, roll。
  依据 opentrack 官方 issue #1203 抓包实测。
"""
from __future__ import annotations

import socket
import struct

from .pose import Pose


class PoseSink:
    """输出目标抽象。推理核心不得依赖具体实现（方案书 3.3）。"""

    def send(self, pose: Pose) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenTrackUdpSink(PoseSink):
    """OpenTrack「UDP over network」原生协议发送器（非连接 sendto）。"""

    PACKET = struct.Struct("<6d")  # 本机字节序 = little-endian（Windows x86-64）

    def __init__(self, host: str = "127.0.0.1", port: int = 4242):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent_count = 0
        self.last_send_epoch = 0.0
        self.error_count = 0  # 本地发送错误计数（上层按最近 60 s 滚动窗口维护）

    def send(self, pose: Pose) -> None:
        payload = self.PACKET.pack(*pose.as_opentrack_order())
        try:
            self._sock.sendto(payload, (self.host, self.port))
            self.sent_count += 1
        except OSError:
            # UDP 无连接：WSAECONNRESET/REFUSED 一律吞掉并计数，
            # 不得崩溃、不得阻塞、不得向用户断言"目标未响应"（方案书 7.4）。
            self.error_count += 1

    def close(self) -> None:
        self._sock.close()
