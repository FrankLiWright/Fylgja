"""NPU 稳态延迟探针（方案书 4.5 第 5 条的预览数据，非正式验收测量）。

对 detector / landmarks 两模型在 NPU 与 CPU 上各跑 N 次推理，输出延迟分布。
正式验收测量（60 s 端到端 + CPU 占用口径）待引擎 N 管线就绪后执行。
"""
from __future__ import annotations

import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_landmarker.task"
N_WARM, N_RUN = 5, 100


def bench(core, compiled, name: str, device: str) -> None:
    inputs = {}
    for inp in compiled.inputs:
        inputs[inp] = np.random.default_rng(0).random(list(inp.shape), dtype=np.float32)
    for _ in range(N_WARM):
        compiled(inputs)
    lat = []
    for _ in range(N_RUN):
        t0 = time.perf_counter()
        compiled(inputs)
        lat.append((time.perf_counter() - t0) * 1000)
    a = np.array(lat)
    print(f"{name} @ {device}: mean {a.mean():.2f} ms, p50 {np.percentile(a, 50):.2f}, "
          f"p95 {np.percentile(a, 95):.2f}, min {a.min():.2f}, max {a.max():.2f}")


def main() -> int:
    import openvino as ov

    core = ov.Core()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(MODEL) as z:
            for name in z.namelist():
                if name.endswith(".tflite"):
                    (Path(tmp) / Path(name).name).write_bytes(z.read(name))
        det_p = Path(tmp) / "face_detector.tflite"
        lmk_p = Path(tmp) / "face_landmarks_detector.tflite"

        for device in ("NPU", "CPU"):
            det = core.compile_model(core.read_model(str(det_p)), device)
            lmk_m = core.read_model(str(lmk_p))
            lmk_m.reshape({0: [1, 256, 256, 3]})
            lmk = core.compile_model(lmk_m, device)
            bench(core, det, "face_detector   ", device)
            bench(core, lmk, "face_landmarks  ", device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
