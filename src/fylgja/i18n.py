"""GUI 双语（中/英）字符串表与切换。

简单字典方案（无需 Qt Linguist）：t(key) 取当前语言文案，
set_lang() 切换后由 GUI 整体重译。
"""
from __future__ import annotations

_lang = "zh"

LANGS = ("zh", "en")

_STRINGS = {
    "zh": {
        "app_title": "Fylgja",
        "start": "▶  开始",
        "stop": "■  停止",
        "engine": "引擎",
        "camera": "摄像头",
        "output": "输出",
        "filter": "滤波",
        "invert": "反转",
        "axes": "坐标轴",
        "landmarks": "关键点",
        "recenter": "回中 (F12)",
        "lang_btn": "EN",
        "engine_auto": "auto",
        "engine_n": "N（NPU）",
        "engine_c": "C（CPU）",
        "tray_show": "显示主窗口",
        "tray_start": "开始",
        "tray_stop": "停止",
        "tray_quit": "退出",
        "tray_min": "已最小化到托盘，追踪继续运行",
        "banner": "当前运行于 CPU 引擎（NPU 自检失败，原因见日志）",
        "err_start": "启动失败",
        "err_camera": "摄像头",
        "err_exists": "已有实例在运行（方案书 5.4）",
        "tip_axes": "显示坐标轴",
        "tip_landmarks": "显示面部关键点",
        "tip_recenter": "回中 (F12)",
        "tip_engine": "推理引擎（N=NPU，C=CPU）",
        "tip_camera": "摄像头",
        "tip_host": "输出地址（OpenTrack UDP）",
        "tip_port": "端口",
        "tip_cut": "滤波 min_cutoff",
        "tip_beta": "滤波 beta",
        "tip_stat": "yaw pitch roll | TX TY TZ | 已发送",
        "tip_lang": "切换语言 / Switch language",
        "inv_yaw": "yaw", "inv_pitch": "pitch", "inv_roll": "roll",
        "inv_tx": "tx", "inv_ty": "ty", "inv_tz": "tz",
        "spatial": "空间音频",
        "spatial_on": "启用",
        "spatial_out": "输出（耳机）",
        "spatial_max": "最大衰减",
        "spatial_dz": "死区",
        "spatial_hint": "建议使用耳机",
        "spatial_meter": "L/R 增益",
        "err_thread": "工作线程已退出，管线已停止",
        "spatial_dead": "空间音频已停止（设备断开？）",
    },
    "en": {
        "app_title": "Fylgja",
        "start": "▶  Start",
        "stop": "■  Stop",
        "engine": "Engine",
        "camera": "Camera",
        "output": "Output",
        "filter": "Filter",
        "invert": "Invert",
        "axes": "Axes",
        "landmarks": "Points",
        "recenter": "Recenter (F12)",
        "lang_btn": "中",
        "engine_auto": "auto",
        "engine_n": "N (NPU)",
        "engine_c": "C (CPU)",
        "tray_show": "Show window",
        "tray_start": "Start",
        "tray_stop": "Stop",
        "tray_quit": "Quit",
        "tray_min": "Minimized to tray — tracking keeps running",
        "banner": "Running on CPU engine (NPU self-check failed — see log)",
        "err_start": "Start failed",
        "err_camera": "Camera",
        "err_exists": "An instance is already running",
        "tip_axes": "Show coordinate axes",
        "tip_landmarks": "Show face landmarks",
        "tip_recenter": "Recenter (F12)",
        "tip_engine": "Inference engine (N=NPU, C=CPU)",
        "tip_camera": "Camera",
        "tip_host": "Output address (OpenTrack UDP)",
        "tip_port": "Port",
        "tip_cut": "Filter min_cutoff",
        "tip_beta": "Filter beta",
        "tip_stat": "yaw pitch roll | TX TY TZ | sent",
        "tip_lang": "Switch language / 切换语言",
        "inv_yaw": "yaw", "inv_pitch": "pitch", "inv_roll": "roll",
        "inv_tx": "tx", "inv_ty": "ty", "inv_tz": "tz",
        "spatial": "Spatial Audio",
        "spatial_on": "Enable",
        "spatial_out": "Output (headphones)",
        "spatial_max": "Max atten.",
        "spatial_dz": "Deadzone",
        "spatial_hint": "Headphones recommended",
        "spatial_meter": "L/R gain",
        "err_thread": "Worker thread exited — pipeline stopped",
        "spatial_dead": "Spatial audio stopped (device disconnected?)",
    },
}


def t(key: str) -> str:
    return _STRINGS.get(_lang, _STRINGS["zh"]).get(key, key)


def get_lang() -> str:
    return _lang


def set_lang(lang: str) -> None:
    global _lang
    if lang in LANGS:
        _lang = lang
