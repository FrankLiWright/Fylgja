"""崩溃取证（方案书 5.4）：全局未处理异常钩子。

任何未捕获异常写入崩溃日志（时间、异常类型、栈、软件版本、所选引擎、
操作系统版本）后正常退出，不得静默消失。主线程走 sys.excepthook，
工作线程走 threading.excepthook（方案书 5.4 的覆盖要求）。
"""
from __future__ import annotations

import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from .logging_setup import default_log_dir, get_logger

log = get_logger("crash")

_engine_name = "unknown"
_version = "unknown"


def _write_crash_log(exc_type, exc_value, exc_tb, thread_name: str,
                     log_dir: Path | None = None) -> Path | None:
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"crash-{stamp}.log"
    try:
        path.write_text(
            f"time: {datetime.now().isoformat()}\n"
            f"thread: {thread_name}\n"
            f"exception_type: {getattr(exc_type, '__name__', exc_type)}\n"
            f"software_version: {_version}\n"
            f"selected_engine: {_engine_name}\n"
            f"os: {platform.platform()}\n"
            f"\n{''.join(traceback.format_exception(exc_type, exc_value, exc_tb))}",
            encoding="utf-8",
        )
    except OSError:
        return None
    return path


def install_crash_hooks(version: str, engine_name_provider=None,
                        log_dir: Path | str | None = None) -> None:
    """安装 sys.excepthook + threading.excepthook（方案书 5.4）。

    engine_name_provider: 可调用，返回当前所选引擎名（崩溃时尚未选定时为 unknown）。
    log_dir: 崩溃日志目录，默认 %LOCALAPPDATA%\\Fylgja\\logs\\。
    """
    global _engine_name, _version
    _version = version

    def _engine() -> str:
        if engine_name_provider is not None:
            try:
                return str(engine_name_provider())
            except Exception:
                return "unknown"
        return "unknown"

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        global _engine_name
        _engine_name = _engine()
        path = _write_crash_log(exc_type, exc_value, exc_tb, "main", log_dir)
        if path:
            log.error("未捕获异常，崩溃日志：%s", path, exc_info=(exc_type, exc_value, exc_tb))
        else:
            log.error("未捕获异常（崩溃日志写入失败）", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        global _engine_name
        _engine_name = _engine()
        path = _write_crash_log(
            args.exc_type, args.exc_value, args.exc_traceback,
            f"thread:{args.thread.name}", log_dir,
        )
        if path:
            log.error("工作线程未捕获异常，崩溃日志：%s", path)
        # 不静默消失：仍交给默认钩子输出到 stderr
        threading.__excepthook__(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook
