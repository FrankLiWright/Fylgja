"""日志层（方案书 9.4）。

- 位置：``%LOCALAPPDATA%\\<软件名>\\logs\\``（可注入便于测试）。
- 级别：ERROR / WARN / INFO / DEBUG（默认 INFO）。
- 轮转：单文件 5 MB × 3 份，总量上限 15 MB。
- 必须记录（调用方责任，本模块提供 logger）：引擎自检结果、摄像头参数、
  60 s 汇总统计、全部配置变更、全部错误与警告。
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "Fylgja"
MAX_BYTES = 5 * 1024 * 1024  # 单文件 5 MB
BACKUP_COUNT = 3             # × 3 份，总量上限 15 MB（方案书 9.4）

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"

_levels = {
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME / "logs"


def parse_level(name: str) -> int:
    return _levels.get(name.upper(), logging.INFO)


def setup_logging(level: str = "INFO", log_dir: Path | str | None = None,
                  console: bool = True) -> Path:
    """配置根 logger；返回日志文件路径。

    幂等：重复调用先移除旧 handler，避免重复输出。
    """
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "Fylgja.log"

    root = logging.getLogger(APP_NAME)
    root.setLevel(parse_level(level))
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    fmt = logging.Formatter(LOG_FORMAT)
    fh = RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    return log_file


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{APP_NAME}.{name}" if name else APP_NAME)
