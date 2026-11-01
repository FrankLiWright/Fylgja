"""空间音频（Fylgja 附属功能）：按头部朝向实时调整左右声道增益。

链路：应用 → 系统默认输出「CABLE Input」（VB-CABLE）→ 本引擎从
「CABLE Output」捕获 → 增益（yaw 驱动 L/R 平衡，斜坡平滑）→ 渲染到真实耳机。

效果语义（仅一种）：**固定声场** —— 头向左转时左耳衰减、右耳不变，
听感上声音"留在原地"。面部追踪数据丢失时平滑回到左右平衡。

面向耳机设计（音箱场景的双声道增益不构成空间音频，仅作娱乐效果）。
虚拟声卡 VB-CABLE 为外置依赖（闭源免费，不随包分发，README 提供指引）。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .logging_setup import get_logger

log = get_logger("audio")

# ---- 常量 ----
CAPTURE_NAME_HINTS = ("CABLE Output",)
PREFERRED_RATE = 48000
BLOCK = 480                      # 10 ms @48k
RAMP_MS = 5.0                    # 跟随头部的增益斜坡（防爆音）
LOSS_RAMP_MS = 300.0             # 追踪丢失时回平衡的斜坡（平滑过渡）
TARGET_STALE_S = 0.5             # 超过该时间未收到目标 = 追踪丢失
QUEUE_BLOCKS = 6                 # 环形缓冲深度（~60 ms）


# ------------------------------------------------------------- 增益算法（纯逻辑）

@dataclass
class GainConfig:
    max_db: float = 12.0         # 最大衰减（dB）
    deadzone_deg: float = 5.0    # 死区（度）


def gains_from_yaw(yaw_deg: float, cfg: GainConfig) -> tuple[float, float]:
    """yaw（度，正=头转向其左侧）→ (gL, gR) 线性增益（≤1，防削顶）。

    固定声场：向左转 → 左耳衰减、右耳不变；向右转镜像。
    """
    y = float(yaw_deg)
    over = abs(y) - float(cfg.deadzone_deg)
    if over <= 0:
        return 1.0, 1.0
    span = max(90.0 - float(cfg.deadzone_deg), 1e-6)
    db = min(float(cfg.max_db), float(cfg.max_db) * over / span)
    att = 10.0 ** (-db / 20.0)
    return (att, 1.0) if y > 0 else (1.0, att)


# ------------------------------------------------------------- 系统音量桥

def _find_endpoint_volume(name_hint: str):
    """返回名称匹配的渲染端点音量接口（找不到返回 None）。

    VB-CABLE 驱动忽略端点音量设定——Windows 音量键调节的正是该端点，
    因此由本引擎读取并作为主增益应用，使系统音量重新生效。
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pycaw.pycaw import AudioUtilities
            for dev in AudioUtilities.GetAllDevices():
                fn = getattr(dev, "FriendlyName", "") or ""
                if name_hint in fn:
                    ep = getattr(dev, "EndpointVolume", None)
                    if ep is not None:
                        return ep
    except Exception:
        pass
    return None


# ------------------------------------------------------------- 双流引擎

class SpatialAudioEngine(threading.Thread):
    """捕获（虚拟声卡）→ 增益 → 渲染（耳机）双流引擎。

    构造即打开音频流（失败抛异常）；start() 后运行，stop() 停止。
    """

    def __init__(self, input_device: int | str | None = None,
                 output_device: int | str | None = None,
                 cfg: GainConfig | None = None):
        super().__init__(name="spatial-audio", daemon=True)
        import sounddevice as sd

        self.cfg = cfg or GainConfig()
        self._stop_evt = threading.Event()
        self._cond = threading.Condition()
        self._queue: deque[np.ndarray] = deque()
        self._targets = (1.0, 1.0)
        self._ramp_ms = RAMP_MS
        self._gains = np.array([1.0, 1.0])
        self._last_target_t = time.monotonic()
        self.blocks_in = 0
        self.blocks_out = 0
        self.underruns = 0
        self.error: str | None = None
        self._volume = _find_endpoint_volume("CABLE Input")
        self._master = 1.0
        self._vol_tick = 0
        self.global_gain = 1.0  # 用户全局增益（0..2，>1 为放大；输出限幅 ±1）

        # ---- 设备解析（失败即抛，便于 GUI 同步报错）----
        in_idx = input_device if input_device is not None else self.find_capture_device()
        if in_idx is None:
            raise RuntimeError("未找到虚拟声卡 CABLE Output（VB-CABLE 未安装？）")
        out_idx = output_device
        if out_idx is None:
            out_idx = sd.default.device[1]
        self.input_name = str(sd.query_devices(in_idx)["name"])
        self.output_name = str(sd.query_devices(out_idx)["name"])

        # ---- 打开双流（公共采样率协商）----
        in_info = sd.query_devices(in_idx)
        out_info = sd.query_devices(out_idx)
        stream_kwargs = dict(channels=2, blocksize=BLOCK, dtype="float32")
        self._in_stream = self._out_stream = None
        last_err: Exception | None = None
        for rate in (PREFERRED_RATE, 44100, int(in_info["default_samplerate"]),
                     int(out_info["default_samplerate"])):
            try:
                self._in_stream = sd.InputStream(device=in_idx, samplerate=rate,
                                                 callback=self._on_input, **stream_kwargs)
                self._out_stream = sd.OutputStream(device=out_idx, samplerate=rate,
                                                   callback=self._on_output, **stream_kwargs)
                self.rate = rate
                break
            except Exception as e:
                last_err = e
                if self._in_stream:
                    self._in_stream.close()
                self._in_stream = self._out_stream = None
        if self._in_stream is None:
            raise RuntimeError(f"无法以公共采样率打开音频流：{last_err}")

    # ---- 控制 ----
    def set_targets(self, g_l: float, g_r: float,
                    ramp_ms: float = RAMP_MS) -> None:
        with self._cond:
            self._targets = (float(g_l), float(g_r))
            self._ramp_ms = float(ramp_ms)
            self._last_target_t = time.monotonic()

    @property
    def current_gains(self) -> tuple[float, float]:
        return float(self._gains[0]), float(self._gains[1])

    @property
    def master_gain(self) -> float:
        """系统音量主增益（0..1，含静音）。"""
        return float(self._master)

    def _poll_master_volume(self) -> None:
        try:
            if self._volume is None:
                self._volume = _find_endpoint_volume("CABLE Input")  # 热重试
            if self._volume is None:
                self._master = 1.0
                return
            vol = float(self._volume.GetMasterVolumeLevelScalar())
            muted = bool(self._volume.GetMute())
            self._master = 0.0 if muted else max(0.0, min(1.0, vol))
        except Exception:
            self._master = 1.0

    def stop(self) -> None:
        self._stop_evt.set()

    # ---- 设备解析 ----
    @staticmethod
    def find_capture_device() -> int | None:
        import sounddevice as sd
        apis = sd.query_hostapis()
        best = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] < 2:
                continue
            if any(h in d["name"] for h in CAPTURE_NAME_HINTS):
                is_wasapi = "WASAPI" in apis[d["hostapi"]]["name"]
                score = (2 if is_wasapi else 0) + (1 if d["max_input_channels"] == 2 else 0)
                if best is None or score > best[0]:
                    best = (score, i)
        return best[1] if best else None

    # ---- 音频回调 ----
    def _on_input(self, indata, frames, time_info, status):
        try:
            # 追踪丢失看门狗：超时未收到目标 → 平滑回平衡
            if time.monotonic() - self._last_target_t > TARGET_STALE_S:
                self._targets = (1.0, 1.0)
                self._ramp_ms = LOSS_RAMP_MS
            block = np.asarray(indata, dtype=np.float32).copy()
            if block.ndim == 1:
                block = block[:, None]
            tgt = np.array(self._targets, dtype=np.float64)
            cur = self._gains
            step = min(1.0, (frames / self.rate) * 1000.0 / max(self._ramp_ms, 1e-3))
            nxt = cur + (tgt - cur) * step
            ramp = np.linspace(cur, nxt, frames, endpoint=False, dtype=np.float64)
            nch = min(block.shape[1], 2)
            block[:, :nch] *= ramp[:, :nch].astype(np.float32)
            self._gains = nxt
            # 系统音量：CABLE Input 端点音量 → 主增益（~100ms 轮询一次）
            self._vol_tick += 1
            if self._vol_tick % 10 == 0:
                self._poll_master_volume()
            g = self._master * float(getattr(self, "global_gain", 1.0))
            if g != 1.0:
                block *= g
                np.clip(block, -1.0, 1.0, out=block)  # 防削顶
            with self._cond:
                if len(self._queue) < QUEUE_BLOCKS:
                    self._queue.append(block)
                self._cond.notify()
            self.blocks_in += 1
        except Exception as e:  # 音频回调异常不得杀死流（稳定性）
            self.error = f"{type(e).__name__}: {e}"

    def _on_output(self, outdata, frames, time_info, status):
        try:
            with self._cond:
                if self._queue:
                    block = self._queue.popleft()
                else:
                    block = None
                    self.underruns += 1
            outdata[:] = 0.0
            if block is not None:
                n = min(frames, block.shape[0])
                outdata[:n] = block[:n]
            self.blocks_out += 1
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    # ---- 主循环 ----
    def run(self) -> None:
        try:
            self._in_stream.start()
            self._out_stream.start()
            log.info("空间音频启动：%s → %s @%d Hz",
                     self.input_name, self.output_name, self.rate)
            while not self._stop_evt.wait(0.1):
                pass
            self._out_stream.stop()
            self._in_stream.stop()
            self._out_stream.close()
            self._in_stream.close()
            log.info("空间音频停止：块 %d/%d，欠载 %d",
                     self.blocks_in, self.blocks_out, self.underruns)
        except Exception as e:
            log.error("空间音频异常：%s", e)


def list_render_devices() -> list[tuple[int, str]]:
    """输出设备清单（含全部 Host API；供 GUI 下拉）。"""
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] >= 2]
