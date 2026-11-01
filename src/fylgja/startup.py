"""启动验证阶段（方案书 6 / 6.1）：引擎选择 → 自检 → 摄像头枚举 → 打开验证。

原则（方案书 6）：
- 任一环节失败必须显式报错，不得自动回退或静默替换用户选择；
  唯一例外是 6.1 的预授权回退（自启动 + 开关开启 + 引擎 N 自检失败 → 引擎 C + 常驻横幅）。
- 引擎切换必须经完整的引擎自检。
- 摄像头按稳定路径解析（camera_enum），打开失败回到选择（交互模式重选，非交互报错）。
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .camera_enum import CameraDevice, enumerate_dshow_devices, resolve_camera
from .capture import CaptureThread, LatestFrameMailbox
from .config import Config
from .engine import PoseEngine
from .errors import Attribution, CameraError, SelfCheckError
from .logging_setup import get_logger

log = get_logger("startup")

SELF_CHECK_TIMEOUT_S = 10.0   # 方案书 5.3：自检冒烟限时 10 s
FALLBACK_BANNER = "当前运行于 CPU 引擎（NPU 自检失败，原因见日志）"  # 6.1 常驻横幅
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"  # 4.5 第 1 条

# prompt 回调：展示选项并返回 "N" 或 "C"（GUI/CLI 各自实现）
PromptFn = Callable[[str], str]


def verify_model(model_path: str | Path, expected_sha: str = MODEL_SHA256) -> None:
    """模型完整性校验（方案书 5.3：存在且 SHA256 匹配，失败归因安装包完整性）。"""
    p = Path(model_path)
    if not p.exists():
        raise SelfCheckError(
            Attribution.PACKAGE,
            f"模型文件缺失：{p}。安装包不完整，请重新下载并按 FIRST-RUN.md 校验 SHA256。",
        )
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest.lower() != expected_sha.lower():
        raise SelfCheckError(
            Attribution.PACKAGE,
            f"模型 SHA256 不匹配（{digest}）。安装包不完整或被修改，请重新下载。",
        )
    log.info("模型完整性校验通过：%s", p.name)


@dataclass
class EngineAttempt:
    engine: str
    ok: bool
    reason: str = ""


@dataclass
class EngineSelection:
    engine: PoseEngine
    name: str
    fallback_banner: str | None = None   # 6.1：非空 = 预授权回退已生效
    chosen_by: str = "config"            # config | prompt | fallback
    attempts: list[EngineAttempt] = field(default_factory=list)


@dataclass
class CameraSelection:
    device: CameraDevice
    capture: CaptureThread
    mailbox: LatestFrameMailbox


def _self_check(factory: Callable[[], PoseEngine], name: str) -> tuple[PoseEngine | None, str]:
    """构造 + 自检（限时）；失败返回 (None, 原因)，原因含归因前缀。"""
    try:
        eng = factory()
    except SelfCheckError as e:
        return None, str(e)
    except Exception as e:
        return None, f"[{Attribution.OTHER.value}] 构造失败 {type(e).__name__}: {e}"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(eng.self_check).result(timeout=SELF_CHECK_TIMEOUT_S)
    except FutureTimeout:
        eng.close()
        return None, f"[{Attribution.OTHER.value}] 自检超时（>{SELF_CHECK_TIMEOUT_S:.0f} s）"
    except SelfCheckError as e:
        eng.close()
        return None, str(e)
    except Exception as e:
        eng.close()
        return None, f"[{Attribution.OTHER.value}] 自检失败 {type(e).__name__}: {e}"
    log.info("引擎 %s 自检通过", name)
    return eng, ""


def select_engine(cfg: Config, factories: dict[str, Callable[[], PoseEngine]],
                  *, is_autostart: bool = False,
                  prompt: PromptFn | None = None) -> EngineSelection:
    """引擎选择（方案书 6 步骤 1-2 + 6.1 预授权回退）。"""
    attempts: list[EngineAttempt] = []

    def try_engine(name: str) -> tuple[PoseEngine | None, str]:
        eng, reason = _self_check(factories[name], name)
        attempts.append(EngineAttempt(name, eng is not None, reason))
        if not eng:
            log.warning("引擎 %s 自检失败：%s", name, reason)
        return eng, reason

    def fallback_or_raise(pref: str, reason: str, allow_prompt: bool) -> EngineSelection:
        """非自启动或开关关闭的失败路径：显式选择或显式报错，绝不静默回退。"""
        if allow_prompt and prompt is not None:
            choice = prompt(
                f"引擎 {pref} 自检失败：{reason}\n"
                f"请显式选择引擎 [N=OpenVINO NPU / C=MediaPipe CPU]："
            ).strip().upper()
            if choice not in ("N", "C"):
                raise SelfCheckError(Attribution.OTHER, f"无效选择 {choice!r}（方案书 6）")
            eng, reason2 = try_engine(choice)
            if eng is None:
                raise SelfCheckError(
                    Attribution.OTHER,
                    f"引擎 {choice} 自检失败：{reason2}（方案书 6）",
                )
            return EngineSelection(eng, choice, chosen_by="prompt", attempts=attempts)
        raise SelfCheckError(
            Attribution.OTHER,
            f"引擎 {pref} 自检失败：{reason}。"
            "未自动回退（方案书 6）：请显式指定引擎或检查安装（方案书 8.3 诊断表）。",
        )

    pref = cfg.engine  # auto | N | C

    if pref in ("N", "C"):
        eng, reason = try_engine(pref)
        if eng is not None:
            return EngineSelection(eng, pref, chosen_by="config", attempts=attempts)
        if is_autostart and cfg.autostart_fallback_c and pref == "N":
            # 6.1 预授权回退：自启动 + 开关开 + N 失败 → C + 常驻横幅
            eng2, reason2 = try_engine("C")
            if eng2 is not None:
                log.warning("6.1 预授权回退生效：%s", FALLBACK_BANNER)
                return EngineSelection(eng2, "C", fallback_banner=FALLBACK_BANNER,
                                       chosen_by="fallback", attempts=attempts)
            raise SelfCheckError(
                Attribution.OTHER,
                f"引擎 N 与 C 自检均失败：N={reason}；C={reason2}（方案书 6.1）",
            )
        return fallback_or_raise(pref, reason, allow_prompt=not is_autostart)

    # pref == "auto"：N 通过选 N；失败需显式选择（6），自启动走 6.1
    eng, reason = try_engine("N")
    if eng is not None:
        return EngineSelection(eng, "N", chosen_by="config", attempts=attempts)
    if is_autostart and cfg.autostart_fallback_c:
        eng2, reason2 = try_engine("C")
        if eng2 is not None:
            log.warning("6.1 预授权回退生效：%s", FALLBACK_BANNER)
            return EngineSelection(eng2, "C", fallback_banner=FALLBACK_BANNER,
                                   chosen_by="fallback", attempts=attempts)
        raise SelfCheckError(
            Attribution.OTHER,
            f"引擎 N 与 C 自检均失败：N={reason}；C={reason2}（方案书 6.1）",
        )
    return fallback_or_raise("N", reason, allow_prompt=True)


def open_camera(cfg: Config, *, devices: list[CameraDevice] | None = None,
                prompt: PromptFn | None = None) -> CameraSelection:
    """摄像头选择 + 打开验证（方案书 6 步骤 3-6）。

    按稳定路径解析（缺失/丢失时显式报错或交互重选）；
    打开失败回到选择（交互重选一轮，非交互报错）。
    """
    devices = devices if devices is not None else enumerate_dshow_devices()
    log.info("摄像头枚举：%s",
             "; ".join(f"{d.index}:{d.name}" for d in devices) or "(无)")

    def open_device(dev: CameraDevice) -> CameraSelection:
        mailbox = LatestFrameMailbox()
        cap = CaptureThread(
            mailbox, camera_index=dev.index,
            width=cfg.camera_width, height=cfg.camera_height, fps=cfg.camera_fps,
            backend=cfg.camera_backend,
        )
        try:
            info = cap.open()
        except RuntimeError as e:
            cap.close()
            raise CameraError(f"摄像头 {dev.index}:{dev.name} 打开失败：{e}") from e
        log.info("摄像头打开：index=%d name=%s 请求=%dx%d@%g 实际=%dx%d@%g 后端=%s",
                 dev.index, dev.name,
                 info.requested[0], info.requested[1], info.requested[2],
                 info.actual[0], info.actual[1], info.actual[2], info.backend)
        return CameraSelection(device=dev, capture=cap, mailbox=mailbox)

    try:
        dev = resolve_camera(devices, path=cfg.camera_path, index=cfg.camera_index)
    except CameraError:
        if prompt is None or not devices:
            raise
        listing = "\n".join(f"  {d.index}: {d.name} ({d.path})" for d in devices)
        choice = prompt(f"摄像头不可用，请选择：\n{listing}\n索引：").strip()
        dev = resolve_camera(devices, path="", index=int(choice))  # 显式重选

    try:
        return open_device(dev)
    except CameraError:
        if prompt is None:
            raise
        listing = "\n".join(f"  {d.index}: {d.name} ({d.path})" for d in devices)
        choice = prompt(f"打开失败（可能被占用），请重新选择：\n{listing}\n索引：").strip()
        dev = resolve_camera(devices, path="", index=int(choice))
        return open_device(dev)  # 二次失败直接抛出（回到步骤 6）
