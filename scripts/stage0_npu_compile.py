"""阶段 0 门槛 2（方案书 4.5）：两 TFLite 子模型的 NPU 编译验证。

用法（仓库根目录）：
    .venv/Scripts/python.exe scripts/stage0_npu_compile.py

从 face_landmarker.task（zip）解出 face_detector.tflite / face_landmarks_detector.tflite，
经 OpenVINO read_model → 静态化 → NPU 编译 → 冒烟推理；记录版本、编译耗时、
缓存命中加载耗时。失败路径（方案书 4.5 第 2 条）：NPU 失败则试 iGPU，仍失败则记录证据。
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
CACHE_DIR = Path(tempfile.gettempdir()) / "Fylgja-ov-cache"


def extract_submodels(task_path: Path, dest: Path) -> dict[str, Path]:
    """face_landmarker.task 是 zip：含 detector / landmarks / blendshapes tflite。"""
    out = {}
    with zipfile.ZipFile(task_path) as z:
        for name in z.namelist():
            if name.endswith(".tflite"):
                target = dest / Path(name).name
                target.write_bytes(z.read(name))
                out[Path(name).name] = target
    return out


def try_compile(core, model_path: Path, device: str, reshape: dict | None,
                tag: str) -> bool:
    t0 = time.perf_counter()
    try:
        model = core.read_model(str(model_path))
        if reshape:
            model.reshape(reshape)
        t_read = time.perf_counter()
        compiled = core.compile_model(model, device)
        t_compile = time.perf_counter()
        # 冒烟推理
        inputs = {}
        for inp in compiled.inputs:
            shape = list(inp.shape)
            dtype = np.float32 if "f32" in str(inp.element_type) or str(inp.element_type) == "<Type: 'float32'>" else np.float32
            inputs[inp] = np.random.default_rng(0).random(shape, dtype=np.float32)
        results = compiled(inputs)
        t_infer = time.perf_counter()
        shapes_out = {str(k.get_any_name()): list(v.shape) for k, v in results.items()}
        print(f"[OK] {tag} @ {device}: read+reshape {(t_read - t0) * 1000:.0f} ms, "
              f"compile {(t_compile - t_read) * 1000:.0f} ms, "
              f"smoke infer {(t_infer - t_compile) * 1000:.1f} ms, outputs={shapes_out}")
        return True
    except Exception as e:
        print(f"[FAIL] {tag} @ {device}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    import openvino as ov

    print(f"openvino: {ov.__version__}")
    print(f"model: {MODEL} ({MODEL.stat().st_size} bytes)")
    core = ov.Core()
    print(f"devices: {core.available_devices}")
    for d in core.available_devices:
        try:
            print(f"  {d}: {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        subs = extract_submodels(MODEL, Path(tmp))
        for k, p in sorted(subs.items()):
            print(f"submodel: {k} ({p.stat().st_size} bytes)")

        det = subs.get("face_detector.tflite")
        lmk = subs.get("face_landmarks_detector.tflite")
        if not det or not lmk:
            print("缺少子模型，终止", file=sys.stderr)
            return 1

        # 编译缓存（方案书 4.3 第 5 条）：测量缓存命中后的加载耗时
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(CACHE_DIR)})

        # 关键点模型静态化 [?,256,256,3] -> [1,256,256,3]（方案书 4.3 第 1 条）
        LMK_RESHAPE = {0: [1, 256, 256, 3]}
        DET_SHAPE_HINT = None  # 检测模型 128×128 静态（附录 A）

        ok_any = False
        for device in ("NPU", "GPU"):  # NPU 失败降级 iGPU（方案书 4.5 第 2 条）
            ok_det = try_compile(core, det, device, DET_SHAPE_HINT, "face_detector")
            ok_lmk = try_compile(core, lmk, device, LMK_RESHAPE, "face_landmarks")
            if ok_det and ok_lmk:
                ok_any = True
                print(f"\n[结论] 两子模型在 {device} 编译通过（阶段 0 门槛 2 · 通过）")
                # 缓存命中测量：再编译一次（同缓存目录）
                t0 = time.perf_counter()
                try:
                    core.compile_model(str(det), device)
                    print(f"[缓存] {device} face_detector 二次加载 {(time.perf_counter() - t0) * 1000:.0f} ms")
                except Exception as e:
                    print(f"[缓存] 二次加载失败：{e}")
                t0 = time.perf_counter()
                try:
                    m = core.read_model(str(lmk)); m.reshape(LMK_RESHAPE)
                    core.compile_model(m, device)
                    print(f"[缓存] {device} face_landmarks 二次加载 {(time.perf_counter() - t0) * 1000:.0f} ms")
                except Exception as e:
                    print(f"[缓存] 二次加载失败：{e}")
                break
            print(f"[降级] {device} 未全部通过 → 尝试下一级\n")

        if not ok_any:
            print("\n[结论] NPU 与 iGPU 均未全部编译通过（阶段 0 门槛 2 · 失败）："
                  "本期仅发布引擎 C，双引擎架构保留（方案书 4.5 第 2 条）")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
