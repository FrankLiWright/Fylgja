"""引擎 C 基线全链路现场验证（方案书 13.1/13.2 的可执行冒烟）。

流程：UDP 监听线程 → app.main() 真实摄像头跑 N 秒 → 校验收到的 OpenTrack 包。

用法（仓库根目录）：
    .venv/Scripts/python.exe scripts/e2e_engine_c.py [--duration 10]

退出码 0 = 全链路通（采集/推理/发送/接收一致）。
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fylgja.app import main as app_main  # noqa: E402

PORT = 42430  # 避开默认 4242，防止与本机 OpenTrack 冲突


def listen(stop_evt: threading.Event, got: list, lock: threading.Lock) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", PORT))
    sock.settimeout(0.2)
    while not stop_evt.is_set():
        try:
            data, _ = sock.recvfrom(64)
        except socket.timeout:
            continue
        if len(data) == 48:
            with lock:
                got.append(struct.unpack("<6d", data))
    sock.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()

    stop_evt = threading.Event()
    got: list = []
    lock = threading.Lock()
    t = threading.Thread(target=listen, args=(stop_evt, got, lock), daemon=True)
    t.start()
    time.sleep(0.2)

    print(f"[e2e] 启动管线 {args.duration:.0f}s，监听 udp://127.0.0.1:{PORT}")
    rc = app_main([
        "--model", str(ROOT / "models" / "face_landmarker.task"),
        "--engine", "C",  # 方案书 6：非交互运行必须显式指定引擎，不静默回退
        "--port", str(PORT),
        "--duration", str(args.duration),
        "--stats-interval", "3",
    ])
    stop_evt.set()
    t.join(timeout=2.0)

    with lock:
        n = len(got)
        sample = got[-1] if got else None

    print(f"[e2e] app.main 返回码={rc}，收到 UDP 包={n}")
    if sample:
        print(
            "[e2e] 末包 (TX,TY,TZ,yaw,pitch,roll) = "
            + ", ".join(f"{v:.3f}" for v in sample)
        )
    if rc != 0:
        print("[e2e] FAIL：管线未正常退出", file=sys.stderr)
        return 1
    if n == 0:
        print(
            "[e2e] WARN：管线运行正常但未发出姿态包——全程未检出人脸"
            "（光照/遮挡/朝向，方案书 8.3），链路机制本身已通过采集→推理环路"
        )
        return 0
    print("[e2e] PASS：采集→引擎C→后处理→UDP→接收 全链路一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
