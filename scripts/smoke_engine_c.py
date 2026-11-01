"""引擎 C 冒烟测试（方案书 5.3 自检的可执行对照）。

用法（在仓库根目录）：
    .venv/Scripts/python.exe scripts/smoke_engine_c.py

验证：mediapipe 可导入 → 模型 SHA256 匹配 → 合成图像 VIDEO 模式推理 → 预热。
退出码 0 = 全部通过。
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = ROOT / "models" / "face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main() -> int:
    print("[1/4] import mediapipe ... ", end="", flush=True)
    try:
        import mediapipe  # noqa: F401
    except ImportError as e:
        print(f"FAIL: {e}")
        return 1
    print("OK")

    print("[2/4] model SHA256 ... ", end="", flush=True)
    if not MODEL.exists():
        print(f"FAIL: {MODEL} 不存在")
        return 1
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"FAIL: {digest} != {SHA256}")
        return 1
    print("OK")

    print("[3/4] EngineC.self_check() ... ", end="", flush=True)
    from fylgja.engine_c import EngineC

    t0 = time.perf_counter()
    try:
        engine = EngineC(str(MODEL))
        engine.self_check()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1
    print(f"OK ({(time.perf_counter() - t0) * 1000:.0f} ms)")

    print("[4/4] 合成帧 process() + 预热 ... ", end="", flush=True)
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    # 时间戳必须严格单调递增（引擎 C VIDEO 契约）
    ts = 1_000
    for _ in range(3):
        engine.process(frame, ts)
        ts += 33
    engine.close()
    print("OK")

    print("smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
