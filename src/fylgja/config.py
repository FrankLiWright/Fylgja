"""配置层（方案书 9）：标量 → 系统原生存储（Windows 注册表），结构化 → JSON。

- 注册表路径与 QSettings(org="Fylgja", app="Fylgja") 兼容：
  ``HKCU\\Software\\Fylgja\\Fylgja\\<key>``，GUI 阶段换 QSettings 读同一存储。
- 每个键显式声明类型与默认值，不依赖隐式字符串转换（方案书 9.2）。
- 加载时校验取值范围；非法值回退默认值并记日志，不导致启动失败（9.3）。
- JSON 带 schema_version；未知键保留原值，缺失键补默认；回退旧版本不得拒绝加载（9.3）。
- 配置变更即时生效并落盘；save() 记录全部变更（9.4）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

from .logging_setup import get_logger

APP_NAME = "Fylgja"
REG_ROOT = rf"Software\{APP_NAME}\{APP_NAME}"  # QSettings(org, app) 兼容路径
SCHEMA_VERSION = 1

log = get_logger("config")


# ---------------------------------------------------------------- 结构化数据

@dataclass
class FilterParams:
    min_cutoff: float = 1.0
    beta: float = 0.5
    d_cutoff: float = 1.0


@dataclass
class Calibration:
    """平移标定：规范化平移量 → 厘米（方案书 3.3/4.5 第 6 条）。"""
    scale: float = 1.0
    tx0: float = 0.0
    ty0: float = 0.0
    tz0: float = 0.0
    fov_deg: float = 60.0


@dataclass
class Config:
    """全部配置项（方案书 9.1）。标量字段进注册表，结构化字段进 JSON。"""

    # --- 标量（注册表）---
    engine: str = "auto"                 # auto | N | C
    autostart_fallback_c: bool = True    # 6.1 预授权回退
    autostart: bool = False              # 开机自启动开关（10）
    first_run_completed: bool = False
    camera_backend: str = "auto"         # auto | dshow | msmf | default
    camera_index: int = 0
    camera_path: str = ""                # 稳定设备路径；空 = 用 index
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    sink_host: str = "127.0.0.1"
    sink_port: int = 4242
    log_level: str = "INFO"
    lang: str = "zh"                  # GUI 语言：zh | en
    recenter_hotkey: str = "F12"
    preview_overlay: bool = True
    # --- 结构化（JSON）---
    filter: FilterParams = field(default_factory=FilterParams)
    calibration: Calibration = field(default_factory=Calibration)
    axis_invert: tuple = (False, False, False, False, False, False)  # yaw..tz

    def copy(self) -> "Config":
        import copy as _copy
        return _copy.deepcopy(self)


# 标量键 → (类型解析器, 校验器)。显式类型，禁止隐式转换（9.2）。
def _bool(v: str) -> bool:
    if v in ("true", "True", "1"):
        return True
    if v in ("false", "False", "0"):
        return False
    raise ValueError(f"非法布尔值: {v!r}")


def _int(v: str) -> int:
    return int(v)


def _float(v: str) -> float:
    return float(v)


def _str(v: str) -> str:
    return v


_BOOLS = ("true", "false")
_ENGINES = ("auto", "N", "C")
_BACKENDS = ("auto", "dshow", "msmf", "default")
_LOG_LEVELS = ("ERROR", "WARN", "INFO", "DEBUG")

SCALARS: dict[str, tuple[Callable[[str], object], Callable[[object], bool]]] = {
    "engine": (_str, lambda v: v in _ENGINES),
    "autostart_fallback_c": (_bool, lambda v: isinstance(v, bool)),
    "autostart": (_bool, lambda v: isinstance(v, bool)),
    "first_run_completed": (_bool, lambda v: isinstance(v, bool)),
    "camera_backend": (_str, lambda v: v in _BACKENDS),
    "camera_index": (_int, lambda v: isinstance(v, int) and v >= 0),
    "camera_path": (_str, lambda v: isinstance(v, str)),
    "camera_width": (_int, lambda v: isinstance(v, int) and 0 < v <= 7680),
    "camera_height": (_int, lambda v: isinstance(v, int) and 0 < v <= 4320),
    "camera_fps": (_float, lambda v: isinstance(v, (int, float)) and 0 < v <= 240),
    "sink_host": (_str, lambda v: isinstance(v, str) and len(v) > 0),
    "sink_port": (_int, lambda v: isinstance(v, int) and 1 <= v <= 65535),
    "log_level": (_str, lambda v: v in _LOG_LEVELS),
    "lang": (_str, lambda v: v in ("zh", "en")),
    "recenter_hotkey": (_str, lambda v: isinstance(v, str)),
    "preview_overlay": (_bool, lambda v: isinstance(v, bool)),
}


# ---------------------------------------------------------------- 存储后端

class ScalarStore:
    """标量键值存储抽象（注册表或内存）。"""

    def get(self, key: str) -> str | None:
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        raise NotImplementedError

    def keys(self) -> list[str]:
        raise NotImplementedError

    def delete_all(self) -> None:
        raise NotImplementedError


class InMemoryScalarStore(ScalarStore):
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._d.get(key)

    def set(self, key: str, value: str) -> None:
        self._d[key] = value

    def keys(self) -> list[str]:
        return list(self._d)

    def delete_all(self) -> None:
        self._d.clear()


class RegistryScalarStore(ScalarStore):
    """Windows 注册表 HKCU\\Software\\<app>\\<app>（QSettings 兼容）。"""

    def __init__(self, root: str = REG_ROOT):
        self._root = root

    def _open(self, access):
        import winreg
        return winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self._root, 0, access)

    def get(self, key: str) -> str | None:
        import winreg
        try:
            with self._open(winreg.KEY_READ) as k:
                val, _ = winreg.QueryValueEx(k, key)
                return str(val)
        except FileNotFoundError:
            return None

    def set(self, key: str, value: str) -> None:
        import winreg
        with self._open(winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, key, 0, winreg.REG_SZ, str(value))

    def keys(self) -> list[str]:
        import winreg
        out: list[str] = []
        try:
            with self._open(winreg.KEY_READ) as k:
                i = 0
                while True:
                    try:
                        out.append(winreg.EnumValue(k, i)[0])
                    except OSError:
                        break
                    i += 1
        except FileNotFoundError:
            pass
        return out

    def delete_all(self) -> None:
        import winreg
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, self._root)
        except FileNotFoundError:
            pass


class JsonStructuredStore:
    """结构化 JSON 文件；读写保留未知键（9.3 迁移约束）。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            log.warning("配置 JSON 损坏，将回退默认值：%s", self.path)
            return {}

    def write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- ConfigStore

class ConfigStore:
    """load / save / reset_defaults（方案书 9.2/9.3/9.4）。"""

    def __init__(self, scalars: ScalarStore | None = None,
                 structured: JsonStructuredStore | None = None):
        self._scalars = scalars or RegistryScalarStore()
        base = Path(
            __import__("os").environ.get("LOCALAPPDATA")
            or Path.home() / "AppData" / "Local"
        ) / APP_NAME
        self._structured = structured or JsonStructuredStore(base / "config.json")

    # ---- 加载 ----
    def load(self) -> Config:
        cfg = Config()
        defaults = Config()

        # 标量：读 → 解析 → 校验；失败回默认并记日志（9.3）
        for key, (parser, validator) in SCALARS.items():
            raw = self._scalars.get(key)
            if raw is None:
                continue
            try:
                val = parser(raw)
            except (ValueError, TypeError):
                log.warning("配置项 %s 值 %r 非法，回退默认 %r", key, raw, getattr(defaults, key))
                continue
            if not validator(val):
                log.warning("配置项 %s 越界 %r，回退默认 %r", key, val, getattr(defaults, key))
                continue
            setattr(cfg, key, val)

        # 结构化：JSON；缺失补默认，未知键保留（9.3）
        data = self._structured.read()
        cfg.filter = _load_dataclass(FilterParams, data.get("filter"))
        cfg.calibration = _load_dataclass(Calibration, data.get("calibration"))
        inv = data.get("axis_invert")
        if isinstance(inv, list) and len(inv) == 6 and all(isinstance(b, bool) for b in inv):
            cfg.axis_invert = tuple(inv)
        elif inv is not None:
            log.warning("axis_invert 非法 %r，回退默认", inv)

        return cfg

    # ---- 落盘 ----
    def save(self, cfg: Config, previous: Config | None = None) -> None:
        """立即落盘；previous 非空时逐键记录变更（9.4）。"""
        for key in SCALARS:
            val = getattr(cfg, key)
            sval = "true" if val is True else "false" if val is False else str(val)
            self._scalars.set(key, sval)
            if previous is not None and getattr(previous, key) != val:
                log.info("配置变更：%s: %r -> %r", key, getattr(previous, key), val)

        data = self._structured.read()  # 先读：保留未知键（9.3）
        data["schema_version"] = SCHEMA_VERSION
        data["filter"] = {f.name: getattr(cfg.filter, f.name) for f in fields(FilterParams)}
        data["calibration"] = {f.name: getattr(cfg.calibration, f.name) for f in fields(Calibration)}
        data["axis_invert"] = list(cfg.axis_invert)
        self._structured.write(data)
        if previous is not None:
            for name in ("filter", "calibration"):
                a, b = getattr(previous, name), getattr(cfg, name)
                if a != b:
                    log.info("配置变更：%s: %r -> %r", name, a, b)
            if previous.axis_invert != cfg.axis_invert:
                log.info("配置变更：axis_invert: %r -> %r", previous.axis_invert, cfg.axis_invert)

    def reset_defaults(self) -> Config:
        """恢复默认：必须同时清注册表与 JSON（方案书 9.2）。"""
        self._scalars.delete_all()
        self._structured.write({"schema_version": SCHEMA_VERSION})
        log.info("配置已重置为默认值")
        return Config()


def _load_dataclass(cls, data):
    """从 dict 构建 dataclass：非法字段回默认并记日志，未知键忽略但保留在 JSON。"""
    obj = cls()
    if not isinstance(data, dict):
        return obj
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        try:
            cur = getattr(obj, f.name)
            if isinstance(cur, bool):
                val = bool(raw)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = str(raw)
            setattr(obj, f.name, val)
        except (ValueError, TypeError):
            log.warning("配置项 %s.%s 值 %r 非法，回退默认 %r",
                        cls.__name__, f.name, raw, getattr(obj, f.name))
    return obj
