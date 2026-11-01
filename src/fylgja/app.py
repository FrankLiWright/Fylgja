"""主程序入口（方案书 5/6/7 章）：启动验证 → 双线程管线 → UDP 输出。

启动顺序：单实例 → 日志 → 崩溃钩子 → 配置加载 → 模型校验 → 引擎选择（6）
→ 摄像头打开验证（6）→ 管线运行 → 干净退出。

用法（仓库根目录）：
    .venv/Scripts/python.exe -m fylgja.app --engine C

Ctrl+C 或 --duration 到时后干净退出。交互选择加 --interactive。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .camera_enum import CameraDevice
from .capture import CaptureThread
from .config import Config, ConfigStore
from .crash import install_crash_hooks
from .engine_c import EngineC
from .errors import CameraError, SelfCheckError
from .logging_setup import get_logger, setup_logging
from .pipeline import InferenceThread
from .postprocess import Calibration, PostProcessor
from .single_instance import SingleInstance
from .sink import OpenTrackUdpSink
from .startup import open_camera, select_engine, verify_model

log = get_logger("app")

WINDOW_TITLE = "Fylgja"  # GUI 阶段用于单实例激活已有窗口
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
LOG_STATS_INTERVAL_S = 60.0  # 方案书 9.4：每 60 s 汇总一次推理/发送统计


def _cli_prompt(msg: str) -> str:
    return input(msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="通用摄像头头部追踪软件（基线链路）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="face_landmarker.task 路径")
    ap.add_argument("--engine", choices=["auto", "N", "C"], default=None,
                    help="覆盖配置中的引擎偏好（本次运行，不落盘）")
    ap.add_argument("--camera", type=int, default=None, help="摄像头索引（覆盖配置）")
    ap.add_argument("--camera-path", default=None, help="稳定设备路径（覆盖配置）")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf", "default"], default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--duration", type=float, default=0.0, help="运行秒数；0 = 直到 Ctrl+C")
    ap.add_argument("--stats-interval", type=float, default=5.0, help="终端统计打印间隔")
    ap.add_argument("--min-cutoff", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--d-cutoff", type=float, default=None)
    ap.add_argument("--interactive", action="store_true",
                    help="失败/首配时用交互提示显式选择（方案书 6）")
    ap.add_argument("--autostart", action="store_true",
                    help="无人值守自启动路径：启用 6.1 预授权回退语义")
    ap.add_argument("--no-console-log", action="store_true", help="日志只进文件")
    args = ap.parse_args(argv)

    # ---- 单实例（方案书 5.4）：二次启动激活已有窗口后退出 ----
    inst = SingleInstance()
    if not inst.acquire():
        inst.activate_existing(WINDOW_TITLE)
        print("已有实例在运行，本次退出（方案书 5.4）。", file=sys.stderr)
        return 5

    # ---- 配置（方案书 9）：加载 + CLI 覆盖（覆盖不落盘）----
    store = ConfigStore()
    cfg = store.load()
    prev = cfg.copy()
    for key in ("engine", "camera_backend", "host", "camera_path"):
        argname = {"host": "host", "engine": "engine", "camera_backend": "backend",
                   "camera_path": "camera_path"}[key]
        val = getattr(args, argname)
        if val is not None:
            setattr(cfg, "sink_host" if key == "host" else key, val)
    for key, argname in (("camera_index", "camera"), ("camera_width", "width"),
                         ("camera_height", "height"), ("camera_fps", "fps"),
                         ("sink_port", "port"), ("min_cutoff", "min_cutoff"),
                         ("beta", "beta"), ("d_cutoff", "d_cutoff")):
        val = getattr(args, argname)
        if val is not None:
            if key in ("min_cutoff", "beta", "d_cutoff"):
                setattr(cfg.filter, key, val)
            else:
                setattr(cfg, key, val)

    # ---- 日志（方案书 9.4）----
    log_file = setup_logging(cfg.log_level, console=not args.no_console_log)
    log.info("Fylgja %s 启动（配置来自 %s）", __version__, "注册表+JSON")

    # ---- 崩溃钩子（方案书 5.4）----
    engine_state = {"name": "unselected"}
    install_crash_hooks(__version__, engine_name_provider=lambda: engine_state["name"])

    # ---- 模型完整性（方案书 5.3）----
    try:
        verify_model(args.model)
    except SelfCheckError as e:
        print(f"[失败] {e}", file=sys.stderr)
        return 2

    # ---- 引擎选择（方案书 6 / 6.1）----
    def engine_factory(name: str):
        if name == "C":
            return EngineC(args.model)
        from .engine_n import EngineN
        return EngineN(args.model)

    prompt = _cli_prompt if (args.interactive and sys.stdin.isatty()) else None
    try:
        sel = select_engine(
            cfg, {"N": lambda: engine_factory("N"), "C": lambda: engine_factory("C")},
            is_autostart=args.autostart, prompt=prompt,
        )
    except SelfCheckError as e:
        log.error("启动失败：%s", e)
        print(f"[失败] {e}", file=sys.stderr)
        return 2
    engine = sel.engine
    engine_state["name"] = sel.name
    log.info("引擎选择结果：%s（来源=%s，尝试记录=%s）",
             sel.name, sel.chosen_by,
             [(a.engine, a.ok) for a in sel.attempts])
    if sel.chosen_by == "prompt":
        # 显式选择结果持久化（方案书 6）
        cfg.engine = sel.name
        store.save(cfg, prev)
        log.info("引擎选择已持久化：%s", sel.name)
    banner = sel.fallback_banner

    # ---- 摄像头（方案书 6 步骤 3-6）----
    try:
        cam = open_camera(cfg, prompt=prompt)
    except CameraError as e:
        log.error("启动失败：%s", e)
        print(f"[失败] {e}", file=sys.stderr)
        engine.close()
        return 3
    info = cam.capture.info
    print(f"[启动] 摄像头：{cam.device.index}:{cam.device.name} "
          f"后端={info.backend} 实际={info.actual[0]}x{info.actual[1]}@{info.actual[2]:g}")
    if (info.actual[0], info.actual[1]) != (info.requested[0], info.requested[1]):
        print("[提示] 实际采集分辨率与请求不一致，以实际值为准（方案书 7.1）")
    if prompt is not None and not prev.camera_path:
        # 显式选择的摄像头持久化稳定路径（下次按 path 解析）
        cfg.camera_path = cam.device.path
        cfg.camera_index = cam.device.index
        store.save(cfg, prev)
        log.info("摄像头选择已持久化：%s", cam.device.path)

    # ---- 后处理 + 输出 ----
    cal = Calibration(
        scale=cfg.calibration.scale, tx0=cfg.calibration.tx0,
        ty0=cfg.calibration.ty0, tz0=cfg.calibration.tz0,
    )
    pp = PostProcessor(
        min_cutoff=cfg.filter.min_cutoff, beta=cfg.filter.beta,
        d_cutoff=cfg.filter.d_cutoff, calibration=cal,
    )
    sink = OpenTrackUdpSink(cfg.sink_host, cfg.sink_port)
    infer = InferenceThread(cam.mailbox, engine, pp, sink)

    cam.capture.start()
    infer.start()
    print(f"[启动] 管线运行中 → udp://{cfg.sink_host}:{cfg.sink_port}（Ctrl+C 退出）")
    if banner:
        print(f"[横幅] {banner}")  # 6.1 常驻横幅；统计行会持续携带

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    last_stats = time.monotonic()
    last_log = time.monotonic()
    frames_at_last_log = 0
    exit_code = 0
    try:
        while True:
            time.sleep(0.1)
            if infer.error is not None or cam.capture.error is not None:
                err = infer.error or cam.capture.error
                log.error("管线暂停：%s", err)
                print(f"[停止] {err}", file=sys.stderr)
                exit_code = 4
                break
            if not infer.is_alive() or not cam.capture.is_alive():
                log.error("工作线程意外退出（方案书 5.4）")
                print("[停止] 工作线程意外退出", file=sys.stderr)
                exit_code = 4
                break
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                log.info("达到 --duration，正常退出")
                break
            if now - last_stats >= args.stats_interval:
                last_stats = now
                line = (
                    f"[统计] 引擎={sel.name} 帧={cam.capture.frames_read} "
                    f"处理={infer.frames_processed} 无人脸={infer.no_face_count} "
                    f"推理={infer.infer_ms_last:.1f} ms(均值 {infer.infer_ms_avg:.1f}) "
                    f"已发送={sink.sent_count}"
                )
                if sink.error_count:
                    line += f" 发送错误 ×{sink.error_count}（本机）"
                if banner:
                    line += f" | {banner}"  # 6.1：横幅不得无提示消失
                print(line)
            if now - last_log >= LOG_STATS_INTERVAL_S:
                # 方案书 9.4：每 60 s 汇总推理耗时与发送统计
                dt = now - last_log
                fps = (infer.frames_processed - frames_at_last_log) / dt if dt > 0 else 0.0
                log.info("60s 汇总：处理帧率=%.1f 推理均值=%.1f ms 已发送=%d 发送错误=%d 无人脸=%d",
                         fps, infer.infer_ms_avg, sink.sent_count,
                         sink.error_count, infer.no_face_count)
                last_log = now
                frames_at_last_log = infer.frames_processed
    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
        log.info("用户中断")
    finally:
        cam.capture.close()
        infer.stop()
        infer.join(timeout=2.0)
        engine.close()
        sink.close()
        inst.release()

    if infer.frames_processed == 0 and infer.no_face_count > 0:
        print("[提示] 全程未检出人脸：检查光照/遮挡/朝向（方案书 8.3）", file=sys.stderr)
    log.info("退出：exit_code=%d 日志=%s", exit_code, log_file)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
