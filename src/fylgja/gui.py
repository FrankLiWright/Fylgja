"""GUI 主入口（方案书 14 第 5 步 / 阶段 5）：Windows 11 Fluent 风格。

- 开始 / 停止：完整启动验证（引擎自检 → 摄像头打开）后运行双线程管线（5.1）。
- 预览（8.1 预览契约）：摄像头画面 + 坐标轴（**由实际发送的 6 个 double 重建**）
  + 脸部关键点（可选叠加，可关闭）+ 数值角标（13.1 一致性自检开关）。
- 托盘与后台：关闭窗口最小化到托盘继续运行；托盘菜单可 开始/停止/退出。

运行：.venv/Scripts/python.exe -m fylgja.gui
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .camera_enum import enumerate_dshow_devices
from .capture import CaptureThread, LatestFrameMailbox
from .config import Calibration, Config, ConfigStore, FilterParams
from .crash import install_crash_hooks
from .errors import CameraError, SelfCheckError
from .gui_style import build_qss, is_system_dark
from .i18n import t, get_lang, set_lang
from .logging_setup import get_logger, setup_logging
from .pipeline import InferenceThread
from .postprocess import PostProcessor
from .single_instance import SingleInstance
from .sink import OpenTrackUdpSink
from .startup import open_camera, select_engine

log = get_logger("gui")

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"
ICON_PATH = Path(__file__).resolve().parents[2] / "assets" / "fylgja_icon.png"
ICO_PATH = Path(__file__).resolve().parents[2] / "assets" / "fylgja.ico"
BANNER_PATH = Path(__file__).resolve().parents[2] / "assets" / "fylgja.png"
AUMID = "com.frankliwright.fylgja"


def app_icon():
    """窗口/托盘/任务栏图标（优先多尺寸 .ico——Windows 任务栏认 ico 生成 HICON）。"""
    from PySide6.QtGui import QIcon
    if ICO_PATH.exists():
        return QIcon(str(ICO_PATH))
    return QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon()


# ---------------------------------------------------------------- 预览部件

class PreviewWidget:
    """预览绘制混入：由 MainWindow 的 paint 驱动（QFrame 上自绘）。"""

    pass


# ---------------------------------------------------------------- 主窗口

def build_window():
    """构造 MainWindow 类（延迟导入 Qt，便于非 GUI 测试不依赖 PySide6）。"""
    from PySide6.QtCore import Qt, QTimer, QSize, Signal, QObject, QRect, QPointF
    from PySide6.QtGui import (QAction, QColor, QImage, QPainter, QPainterPath,
                               QPen, QPixmap, QFont, QIcon, QShortcut, QKeySequence,
                               QPolygonF, QLinearGradient)
    from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                                   QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                   QLineEdit, QMainWindow, QMenu, QMessageBox,
                                   QPushButton, QSizePolicy, QSlider, QSystemTrayIcon,
                                   QVBoxLayout, QWidget)

    from .model_assets import extract_assets, parse_geometry_metadata
    from .pose import Pose, rotation_matrix

    class PreviewFrame(QFrame):
        """自绘预览：摄像头帧 + 叠加层（坐标轴/关键点/数值）。"""

        def __init__(self):
            super().__init__()
            self.setObjectName("previewFrame")
            self.setMinimumSize(480, 270)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._qimage: QImage | None = None
            self._landmarks: np.ndarray | None = None
            self._sent: tuple | None = None
            self._show_axes = True
            self._show_landmarks = True
            self._axes_cache = None  # ((sent, tw, th), wmat) 投影矩阵缓存
            self._canonical: np.ndarray | None = None
            self._fps_text = ""

        def set_options(self, axes: bool, landmarks: bool):
            self._show_axes, self._show_landmarks = axes, landmarks

        def set_canonical(self, v: np.ndarray):
            self._canonical = v  # (468,3)

        def update_frame(self, frame_rgb: np.ndarray, landmarks, sent, fps_text: str):
            h, w, _ = frame_rgb.shape
            self._qimage = QImage(frame_rgb.data, w, h, 3 * w,
                                  QImage.Format.Format_RGB888).copy()
            self._landmarks = landmarks
            self._sent = sent
            self._fps_text = fps_text
            self.update()

        def clear(self):
            """停止后清空预览（回到深色空态）。"""
            self._qimage = None
            self._landmarks = None
            self._sent = None
            self._fps_text = ""
            self._axes_cache = None
            self.update()

        def paintEvent(self, ev):
            # 稳定性：绘制异常不得逃逸（否则 Qt 绘制状态损坏 → 缩放时崩溃）
            try:
                self._paint()
            except Exception as e:
                try:
                    log.error("预览绘制异常：%s", e)
                except Exception:
                    pass

        def _paint(self):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w_r, h_r = rect.width(), rect.height()
            if w_r <= 0 or h_r <= 0:
                return
            if self._qimage is None:
                p.fillRect(rect, QColor("#171717"))
                return
            iw, ih = self._qimage.width(), self._qimage.height()
            if iw <= 0 or ih <= 0:
                return
            # 直接按目标矩形绘制（避免 scaled() 逐帧分配——缩放窗口时卡死的主因）
            scale = min(w_r / iw, h_r / ih)
            tw, th = max(int(iw * scale), 1), max(int(ih * scale), 1)
            x0, y0 = (w_r - tw) // 2, (h_r - th) // 2
            p.drawImage(QRect(x0, y0, tw, th), self._qimage)

            lm = self._landmarks
            if lm is not None and self._show_landmarks:
                pts = QPolygonF()
                for pt in lm:
                    pts.append(QPointF(int(x0 + pt[0] * tw), int(y0 + pt[1] * th)))
                # drawPoints 用画笔绘制（NoPen 会导致不可见——勿改回）
                p.setPen(QPen(QColor(60, 220, 120, 190), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPoints(pts)

            sent = self._sent
            if sent is not None and self._show_axes and lm is not None and self._canonical is not None:
                self._draw_axes(p, x0, y0, tw, th, sent, lm)
            if self._fps_text:
                p.setPen(QColor("#B8B8B8"))
                p.drawText(10, y0 + th - 10, self._fps_text)

        def _draw_axes(self, p, x0, y0, tw, th, sent, lm):
            """坐标轴：由发送值重建 R，经 度量→图像 仿射投影（预览契约 8.1）。

            投影矩阵按 (发送值, 画布尺寸) 缓存——窗口缩放时避免逐帧 lstsq。
            """
            key = (tuple(float(v) for v in sent), tw, th)
            wmat = None
            if self._axes_cache is not None and self._axes_cache[0] == key:
                wmat = self._axes_cache[1]
            else:
                tx, ty, tz, yaw, pitch, roll = sent
                rmat = rotation_matrix(yaw, pitch, roll)
                metric = (rmat @ self._canonical.T).T + np.array([tx, ty, tz])
                target = lm[:468, :2] * np.array([tw, th])
                a = np.hstack([metric, np.ones((len(metric), 1))])
                try:
                    wmat, *_ = np.linalg.lstsq(a, target, rcond=None)
                except np.linalg.LinAlgError:
                    return
                if not np.isfinite(wmat).all():
                    return
                self._axes_cache = (key, wmat)

            def proj(pt3):
                return np.array([pt3[0], pt3[1], pt3[2], 1.0]) @ wmat

            tx, ty, tz, yaw, pitch, roll = sent
            rmat = rotation_matrix(yaw, pitch, roll)
            origin = np.array([tx, ty, tz])
            colors = [(230, 70, 70), (70, 200, 110), (80, 140, 255)]
            for i, color in enumerate(colors):
                tip3 = origin + rmat[:, i] * 4.0
                p0 = proj(origin)
                p1 = proj(tip3)
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                p.setPen(QPen(QColor(*color), 3))
                p.setBrush(QColor(*color))
                p.drawLine(int(x0 + p0[0]), int(y0 + p0[1]),
                           int(x0 + p1[0]), int(y0 + p1[1]))
                dx, dy = p1 - p0
                n = max(float(np.hypot(dx, dy)), 1e-6)
                ux, uy = dx / n, dy / n
                px, py = -uy, ux
                base = p1 - 0.25 * n * np.array([ux, uy])
                poly = QPolygonF()
                for q in ((p1[0], p1[1]),
                          (base[0] + 6 * px, base[1] + 6 * py),
                          (base[0] - 6 * px, base[1] - 6 * py)):
                    poly.append(QPointF(int(x0 + q[0]), int(y0 + q[1])))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(poly)
            # 鼻尖点
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255))
            np_pt = lm[1] * np.array([tw, th])
            p.drawEllipse(int(x0 + np_pt[0]) - 3, int(y0 + np_pt[1]) - 3, 6, 6)

    class MainWindow(QMainWindow):
        pipeline_event = Signal(str)

        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Fylgja · 头部追踪  v{__version__}")
            self.setWindowIcon(app_icon())
            self.setMinimumSize(920, 620)
            self._default_size_applied = False
            self.resize(1080, 680)
            self.store = ConfigStore()
            self.cfg: Config = self.store.load()
            set_lang(self.cfg.lang)
            self.engine = None
            self.sel = None
            self.capture: CaptureThread | None = None
            self.infer: InferenceThread | None = None
            self.sink = None
            self.mailbox = LatestFrameMailbox()
            self.running = False
            self._tray: QSystemTrayIcon | None = None
            self._last_frames = [0.0, 0.0]
            self._last_fps = 0.0

            self._build_ui()
            self._build_tray()
            self._load_canonical()
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._poll)
            self._timer.start(33)
            # 深色模式自动切换（跟随系统）
            self._dark = is_system_dark()
            self._theme_timer = QTimer(self)
            self._theme_timer.timeout.connect(self._sync_theme)
            self._theme_timer.start(3000)
            self._sync_theme()

        # ---------------- UI ----------------
        def showEvent(self, ev):
            super().showEvent(ev)
            if not self._default_size_applied:
                self._default_size_applied = True
                self._apply_default_size()

        def _apply_default_size(self):
            """默认窗口 = 1783×1213 屏幕像素（含标题栏），按 DPI 与边框换算客户区。"""
            try:
                dpr = self.devicePixelRatioF() or 1.0
                fg, g = self.frameGeometry(), self.geometry()
                fw = max(fg.width() - g.width(), 0)
                fh = max(fg.height() - g.height(), 0)
                self.resize(max(920, round(1783 / dpr) - fw),
                            max(620, round(1213 / dpr) - fh))
            except Exception:
                self.resize(1189, 776)

        def _sync_theme(self):
            dark = is_system_dark()
            if dark != getattr(self, "_dark", None):
                self._dark = dark
            from PySide6.QtWidgets import QApplication as QA
            QA.instance().setStyleSheet(build_qss(self._dark))

        def _bind(self, widget, key, attr="text"):
            # 必须调用 Qt setter：setattr(widget,"text",...) 只是遮蔽 Python 属性
            setter = widget.setText if attr == "text" else widget.setToolTip
            setter(t(key))
            self._tr.append((widget, key, attr))
            return widget

        def _retranslate(self):
            for widget, key, attr in self._tr:
                setter = widget.setText if attr == "text" else widget.setToolTip
                setter(t(key))
            # 引擎下拉项
            idx = self.cmbEngine.currentIndex()
            self.cmbEngine.blockSignals(True)
            self.cmbEngine.clear()
            self.cmbEngine.addItems([t("engine_auto"), t("engine_n"), t("engine_c")])
            self.cmbEngine.setCurrentIndex(idx)
            self.cmbEngine.blockSignals(False)
            self._refresh_cameras()
            self._build_tray()
            if self.banner.isVisible():
                self.banner.setText("⚠ " + t("banner"))
            self.btnLang.setText(t("lang_btn"))
            self.statBar.setToolTip(t("tip_stat"))

        def _toggle_lang(self):
            set_lang("en" if get_lang() == "zh" else "zh")
            self.cfg.lang = get_lang()
            self.store.save(self.cfg)
            self._retranslate()

        def _build_ui(self):
            self._tr = []
            root = QWidget()
            root.setObjectName("root")
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(16, 12, 16, 10)
            outer.setSpacing(10)

            head = QHBoxLayout()
            title = QLabel("Fylgja")
            title.setObjectName("title")
            head.addWidget(title)
            head.addStretch()
            self.btnLang = QPushButton(t("lang_btn"))
            self.btnLang.setFixedHeight(30)
            self.btnLang.setToolTip(t("tip_lang"))
            self.btnLang.clicked.connect(self._toggle_lang)
            self._bind(self.btnLang, "lang_btn")
            head.addWidget(self.btnLang)
            outer.addLayout(head)

            self.banner = QLabel("")
            self.banner.setObjectName("banner")
            self.banner.setVisible(False)
            outer.addWidget(self.banner)

            body = QHBoxLayout()
            body.setSpacing(12)
            outer.addLayout(body, 1)

            left = QVBoxLayout()
            left.setSpacing(8)
            self.preview = PreviewFrame()
            left.addWidget(self.preview, 1)
            tools = QHBoxLayout()
            tools.setSpacing(14)
            self.cbAxes = self._bind(QCheckBox(), "axes")
            self.cbDots = self._bind(QCheckBox(), "landmarks")
            for cb, tip in ((self.cbAxes, "tip_axes"), (self.cbDots, "tip_landmarks")):
                cb.setChecked(True)
                cb.setToolTip(t(tip))
                cb.toggled.connect(self._apply_overlay_opts)
                tools.addWidget(cb)
            tools.addStretch()
            self.btnRecenter = self._bind(QPushButton(), "recenter")
            self.btnRecenter.setToolTip(t("tip_recenter"))
            self.btnRecenter.clicked.connect(self._recenter)
            tools.addWidget(self.btnRecenter)
            left.addLayout(tools)
            body.addLayout(left, 3)

            panel = QWidget()
            panel.setObjectName("card")
            panel.setFixedWidth(272)
            pv = QVBoxLayout(panel)
            pv.setContentsMargins(16, 16, 16, 16)
            pv.setSpacing(6)

            self.btnStart = QPushButton(t("start"))
            self.btnStart.setObjectName("startBtn")
            self._bind(self.btnStart, "start")
            self.btnStart.clicked.connect(self.on_start)
            self.btnStop = QPushButton(t("stop"))
            self.btnStop.setObjectName("stopBtn")
            self._bind(self.btnStop, "stop")
            self.btnStop.setEnabled(False)
            self.btnStop.clicked.connect(self.on_stop)
            pv.addWidget(self.btnStart)
            pv.addWidget(self.btnStop)
            pv.addSpacing(6)

            self.lblEngine = self._bind(self._mini(""), "engine")
            pv.addWidget(self.lblEngine)
            self.cmbEngine = QComboBox()
            self.cmbEngine.addItems([t("engine_auto"), t("engine_n"), t("engine_c")])
            self.cmbEngine.setToolTip(t("tip_engine"))
            self.cmbEngine.currentIndexChanged.connect(self._on_engine_choice)
            pv.addWidget(self.cmbEngine)

            self.lblCamera = self._bind(self._mini(""), "camera")
            pv.addWidget(self.lblCamera)
            self.cmbCamera = QComboBox()
            self.cmbCamera.setToolTip(t("tip_camera"))
            pv.addWidget(self.cmbCamera)
            self._refresh_cameras()
            self.cmbCamera.currentIndexChanged.connect(self._on_camera_choice)

            self.lblOut = self._bind(self._mini(""), "output")
            pv.addWidget(self.lblOut)
            sinkRow = QHBoxLayout()
            sinkRow.setSpacing(6)
            self.edHost = QLineEdit(self.cfg.sink_host)
            self.edHost.setToolTip(t("tip_host"))
            self.edPort = QLineEdit(str(self.cfg.sink_port))
            self.edPort.setToolTip(t("tip_port"))
            self.edPort.setFixedWidth(58)
            sinkRow.addWidget(self.edHost, 1)
            sinkRow.addWidget(self.edPort)
            pv.addLayout(sinkRow)

            self.lblFilter = self._bind(self._mini(""), "filter")
            pv.addWidget(self.lblFilter)
            self.slCut = self._slider(1, 30, int(self.cfg.filter.min_cutoff * 10))
            self.slCut.setToolTip(t("tip_cut"))
            self.slBeta = self._slider(0, 20, int(self.cfg.filter.beta * 10))
            self.slBeta.setToolTip(t("tip_beta"))
            self.lblCut = QLabel()
            self.lblBeta = QLabel()
            self.lblCut.setObjectName("hint")
            self.lblBeta.setObjectName("hint")
            for sl, lbl, name in ((self.slCut, self.lblCut, "min_cutoff"),
                                  (self.slBeta, self.lblBeta, "beta")):
                row = QHBoxLayout()
                nm = QLabel(name)
                nm.setObjectName("hint")
                row.addWidget(nm)
                row.addStretch()
                row.addWidget(lbl)
                pv.addLayout(row)
                pv.addWidget(sl)
            self.slCut.valueChanged.connect(self._on_filter)
            self.slBeta.valueChanged.connect(self._on_filter)
            self._on_filter()

            self.lblInv = self._bind(self._mini(""), "invert")
            pv.addWidget(self.lblInv)
            invRow1 = QHBoxLayout()
            invRow1.setSpacing(10)
            invRow2 = QHBoxLayout()
            invRow2.setSpacing(10)
            self.invBoxes = []
            for i, name in enumerate(["inv_yaw", "inv_pitch", "inv_roll",
                                      "inv_tx", "inv_ty", "inv_tz"]):
                cb = self._bind(QCheckBox(), name)
                cb.setChecked(bool(self.cfg.axis_invert[i]))
                cb.toggled.connect(self._on_invert)
                self.invBoxes.append(cb)
                (invRow1 if i < 3 else invRow2).addWidget(cb)
            invRow1.addStretch()
            invRow2.addStretch()
            pv.addLayout(invRow1)
            pv.addLayout(invRow2)

            # ---- 空间音频（附属功能）----
            self.lblSpatial = self._bind(self._mini(""), "spatial")
            pv.addWidget(self.lblSpatial)
            spTop = QHBoxLayout()
            self.cbSpatialOn = self._bind(QCheckBox(), "spatial_on")
            self.cbSpatialOn.toggled.connect(self._on_spatial_toggle)
            spTop.addWidget(self.cbSpatialOn)
            spTop.addStretch()
            self.lblSpatialMeter = QLabel("L 1.00  R 1.00")
            self.lblSpatialMeter.setObjectName("hint")
            self.lblSpatialMeter.setToolTip(t("spatial_meter"))
            spTop.addWidget(self.lblSpatialMeter)
            pv.addLayout(spTop)

            self.cmbSpatialOut = QComboBox()
            self.cmbSpatialOut.setToolTip(t("spatial_out"))
            try:
                from .spatial_audio import list_render_devices
                self._render_devices = list_render_devices()
            except Exception:
                self._render_devices = []
            for idx, name in self._render_devices:
                self.cmbSpatialOut.addItem(name, idx)
            for i, (idx, name) in enumerate(self._render_devices):
                if "Headphones" in name or "耳机" in name:
                    self.cmbSpatialOut.setCurrentIndex(i)
                    break
            pv.addWidget(self.cmbSpatialOut)

            self.slSpatialMax = self._slider(0, 60, 12)
            self.slSpatialDz = self._slider(0, 30, 5)
            self.slGlobalGain = self._slider(0, 200, 100)
            self.slGlobalGain.setToolTip(t("spatial_gain"))
            self.lblSpatialMax = QLabel()
            self.lblSpatialDz = QLabel()
            self.lblGlobalGain = QLabel()
            for sl, lbl, name in ((self.slSpatialMax, self.lblSpatialMax, "spatial_max"),
                                  (self.slSpatialDz, self.lblSpatialDz, "spatial_dz"),
                                  (self.slGlobalGain, self.lblGlobalGain, "spatial_gain")):
                row = QHBoxLayout()
                nm = self._bind(QLabel(), name)
                nm.setObjectName("hint")
                row.addWidget(nm)
                row.addStretch()
                row.addWidget(lbl)
                lbl.setObjectName("hint")
                pv.addLayout(row)
                pv.addWidget(sl)
                sl.valueChanged.connect(self._on_spatial_cfg)
            self.lblSpatialHint = self._bind(QLabel(), "spatial_hint")
            self.lblSpatialHint.setObjectName("hint")
            self.lblSpatialHint.setWordWrap(True)
            pv.addWidget(self.lblSpatialHint)
            self._on_spatial_cfg()

            pv.addStretch()

            body.addWidget(panel)

            self.statBar = QLabel("")
            self.statBar.setObjectName("hint")
            self.statBar.setToolTip(t("tip_stat"))
            outer.addWidget(self.statBar)

            QShortcut(QKeySequence("F12"), self, self._recenter)
            self._apply_overlay_opts()

        def _mini(self, text):
            lbl = QLabel(text)
            lbl.setObjectName("mini")
            return lbl

        def _slider(self, lo, hi, val):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            return s

        def _apply_overlay_opts(self):
            self.preview.set_options(self.cbAxes.isChecked(),
                                     self.cbDots.isChecked())

        def _refresh_cameras(self):
            self.cmbCamera.blockSignals(True)
            self.cmbCamera.clear()
            try:
                self.devices = enumerate_dshow_devices()
            except Exception as e:
                self.devices = []
                log.warning("摄像头枚举失败：%s", e)
            for d in self.devices:
                self.cmbCamera.addItem(f"{d.index}: {d.name}", d.path)
            # 选中配置中的路径
            for i, d in enumerate(self.devices):
                if d.path and d.path == self.cfg.camera_path:
                    self.cmbCamera.setCurrentIndex(i)
                    break
            self.cmbCamera.blockSignals(False)

        def _on_engine_choice(self):
            self.cfg.engine = ["auto", "N", "C"][self.cmbEngine.currentIndex()]
            self.store.save(self.cfg)

        def _on_camera_choice(self):
            path = self.cmbCamera.currentData()
            if path:
                self.cfg.camera_path = path
                idx = self.devices[self.cmbCamera.currentIndex()].index
                self.cfg.camera_index = idx
                self.store.save(self.cfg)

        def _on_filter(self):
            mc = self.slCut.value() / 10.0
            be = self.slBeta.value() / 10.0
            self.lblCut.setText(f"{mc:.1f}")
            self.lblBeta.setText(f"{be:.1f}")
            self.cfg.filter = FilterParams(min_cutoff=mc, beta=be,
                                           d_cutoff=self.cfg.filter.d_cutoff)
            if self.infer is not None:
                for f in self.infer._postprocessor.filters:
                    f.min_cutoff = mc
                    f.beta = be
            self.store.save(self.cfg)

        def _on_invert(self):
            self.cfg.axis_invert = tuple(cb.isChecked() for cb in self.invBoxes)
            if self.infer is not None:
                self.infer._postprocessor.axis_invert = list(self.cfg.axis_invert)
            self.store.save(self.cfg)

        def _recenter(self):
            if self.infer is not None:
                self.infer._postprocessor.recenter()
                self.statBar.setText("\u2713")

        # ---------------- 空间音频 ----------------
        def _on_spatial_toggle(self, on):
            if on:
                try:
                    from .spatial_audio import GainConfig, SpatialAudioEngine
                    self.spatial = SpatialAudioEngine(
                        output_device=self.cmbSpatialOut.currentData(),
                        cfg=GainConfig(max_db=float(self.slSpatialMax.value()),
                                       deadzone_deg=float(self.slSpatialDz.value())))
                    self.spatial.start()
                    self.spatial.global_gain = self.slGlobalGain.value() / 100.0
                    self.statBar.setText(f"\u2192 {self.spatial.output_name}")
                except Exception as e:
                    self.spatial = None
                    self.cbSpatialOn.blockSignals(True)
                    self.cbSpatialOn.setChecked(False)
                    self.cbSpatialOn.blockSignals(False)
                    QMessageBox.warning(self, t("err_start"), str(e))
            else:
                sp = getattr(self, "spatial", None)
                if sp is not None:
                    sp.stop()
                    sp.join(timeout=2.0)
                    self.spatial = None
            self._on_spatial_cfg()

        def _on_spatial_cfg(self, *_):
            self.lblSpatialMax.setText(f"{self.slSpatialMax.value()} dB")
            self.lblSpatialDz.setText(f"{self.slSpatialDz.value()}\u00b0")
            self.lblGlobalGain.setText(f"{self.slGlobalGain.value()}%")
            sp = getattr(self, "spatial", None)
            if sp is not None:
                sp.cfg.max_db = float(self.slSpatialMax.value())
                sp.cfg.deadzone_deg = float(self.slSpatialDz.value())
                sp.global_gain = self.slGlobalGain.value() / 100.0

        # ---------------- 托盘 ----------------
        def _build_tray(self):
            if not QSystemTrayIcon.isSystemTrayAvailable():
                return
            menu = QMenu()
            actShow = menu.addAction(t("tray_show"))
            actShow.triggered.connect(self._show_from_tray)
            menu.addSeparator()
            actStart = menu.addAction(t("tray_start"))
            actStart.triggered.connect(self.on_start)
            actStop = menu.addAction(t("tray_stop"))
            actStop.triggered.connect(self.on_stop)
            menu.addSeparator()
            actQuit = menu.addAction(t("tray_quit"))
            actQuit.triggered.connect(self._quit)
            if getattr(self, "_tray", None) is not None:
                self._tray.setContextMenu(menu)  # 仅换菜单，不重建图标
                return
            pm = QPixmap(64, 64)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(4, 4, 56, 56, 14, 14)
            g = QLinearGradient(0, 0, 64, 64)
            g.setColorAt(0, QColor("#1A7AD1"))
            g.setColorAt(1, QColor("#0067C0"))
            p.fillPath(path, g)
            p.setPen(QColor("white"))
            p.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
            p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "HT")
            p.end()
            self._tray = QSystemTrayIcon(
                app_icon() if ICON_PATH.exists() else QIcon(pm), self)
            self._tray.setContextMenu(menu)
            self._tray.activated.connect(self._tray_activated)
            self._tray.setToolTip("Fylgja")
            self._tray.show()

        def _tray_activated(self, reason):
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                self._show_from_tray()

        def _show_from_tray(self):
            self.showNormal()
            self.raise_()
            self.activateWindow()

        def closeEvent(self, ev):
            # 关闭 → 最小化到托盘，管线继续运行（后台策略）
            if self._tray is not None and self._tray.isVisible():
                self.hide()
                self._tray.showMessage("Fylgja", t("tray_min"),
                                       QSystemTrayIcon.MessageIcon.Information, 2500)
                ev.ignore()
                return
            self._teardown()
            ev.accept()

        def _quit(self):
            self._teardown()
            from PySide6.QtWidgets import QApplication as QA
            QA.quit()

        # ---------------- 管线生命周期 ----------------
        def on_start(self):
            if self.running:
                return
            self.cfg.sink_host = self.edHost.text().strip() or "127.0.0.1"
            try:
                self.cfg.sink_port = int(self.edPort.text().strip())
            except ValueError:
                self.cfg.sink_port = 4242
            self.store.save(self.cfg)

            def prompt(msg):
                from PySide6.QtWidgets import QInputDialog
                text, ok = QInputDialog.getItem(
                    self, "引擎选择", msg.split("\n")[0],
                    ["N", "C"], 0, False)
                return text if ok else "X"

            # 1) 引擎选择 + 自检（方案书 6）
            def factory(name):
                if name == "C":
                    from .engine_c import EngineC
                    return EngineC(str(DEFAULT_MODEL))
                from .engine_n import EngineN
                return EngineN(str(DEFAULT_MODEL), device="NPU")

            try:
                verify_ok = DEFAULT_MODEL.exists()
                from .startup import verify_model
                verify_model(DEFAULT_MODEL)
                self.sel = select_engine(
                    self.cfg,
                    {"N": lambda: factory("N"), "C": lambda: factory("C")},
                    is_autostart=False, prompt=prompt)
            except (SelfCheckError, Exception) as e:
                QMessageBox.warning(self, t("err_start"), str(e))
                return
            self.engine = self.sel.engine
            if self.sel.chosen_by == "prompt":
                self.cfg.engine = self.sel.name
                self.store.save(self.cfg)
            if self.sel.fallback_banner:
                self.banner.setText("⚠ " + t("banner"))
                self.banner.setVisible(True)

            # 2) 摄像头（方案书 6 步骤 3-6）
            try:
                cam = open_camera(self.cfg,
                                  devices=getattr(self, "devices", None) or enumerate_dshow_devices())
            except (CameraError, Exception) as e:
                QMessageBox.warning(self, t("err_camera"), str(e))
                self.engine.close()
                self.engine = None
                return
            self.capture = cam.capture
            self.mailbox = cam.mailbox

            # 3) 后处理 + 输出
            cal = Calibration(scale=self.cfg.calibration.scale,
                              tx0=self.cfg.calibration.tx0,
                              ty0=self.cfg.calibration.ty0,
                              tz0=self.cfg.calibration.tz0)
            pp = PostProcessor(
                min_cutoff=self.cfg.filter.min_cutoff, beta=self.cfg.filter.beta,
                d_cutoff=self.cfg.filter.d_cutoff, calibration=cal,
                axis_invert=self.cfg.axis_invert)
            self.sink = OpenTrackUdpSink(self.cfg.sink_host, self.cfg.sink_port)
            self.infer = InferenceThread(self.mailbox, self.engine, pp, self.sink)
            self.capture.start()
            self.infer.start()
            self.running = True
            self.btnStart.setEnabled(False)
            self.btnStop.setEnabled(True)
            self.statBar.setText(f"\u2192 udp://{self.cfg.sink_host}:{self.cfg.sink_port}")
            log.info("GUI 管线启动：引擎 %s", self.sel.name)

        def on_stop(self):
            if not self.running:
                return
            self._teardown()
            self.running = False
            self.btnStart.setEnabled(True)
            self.btnStop.setEnabled(False)
            self.statBar.setText("")
            self.banner.setVisible(False)
            self.preview.clear()  # 停止后关闭预览画面（回空态）

        def _teardown(self):
            sp = getattr(self, "spatial", None)
            if sp is not None:
                sp.stop()
                sp.join(timeout=2.0)
                self.spatial = None
            if self.capture is not None:
                self.capture.close()
                self.capture = None
            if self.infer is not None:
                self.infer.stop()
                self.infer.join(timeout=2.0)
                self.infer = None
            if self.engine is not None:
                self.engine.close()
                self.engine = None
            if self.sink is not None:
                self.sink.close()
                self.sink = None
            self.running = False

        # ---------------- 预览轮询（5.1：UI 只读共享状态 + 显式拷贝）--------
        def _load_canonical(self):
            try:
                assets = extract_assets(DEFAULT_MODEL, DEFAULT_MODEL.parent / "extracted")
                meta = parse_geometry_metadata(
                    assets["geometry_pipeline_metadata_landmarks.binarypb"].read_bytes())
                self.preview.set_canonical(meta.canonical_vertices)
            except Exception as e:
                log.warning("几何元数据加载失败（坐标轴停用）：%s", e)

        def _poll(self):
            # 稳定性：轮询异常不得杀死 QTimer
            try:
                self._poll_inner()
            except Exception as e:
                log.error("预览轮询异常：%s", e)

        def _poll_inner(self):
            frame, _ts, seq = self.mailbox.get()
            if frame is None or self.infer is None:
                return
            rgb = np.ascontiguousarray(frame[:, :, ::-1])  # 显式拷贝（5.1）
            sent = self.infer.last_sent
            lm = self.infer.last_landmarks
            now = time.monotonic()
            dt = now - self._last_frames[1]
            if dt > 0.5:
                self._last_fps = 0.0
            fps_text = ""
            if self.infer is not None:
                fps_text = (f"{self._last_fps:4.1f}  {self.infer.infer_ms_last:.1f}ms  "
                            f"{self.sink.sent_count if self.sink else 0}")
            self.preview.update_frame(rgb, lm, sent, fps_text)
            # 空间音频联动：yaw 驱动增益；追踪丢失 → 平滑回平衡
            sp = getattr(self, "spatial", None)
            if sp is not None:
                if sent is not None:
                    from .spatial_audio import gains_from_yaw
                    gl, gr = gains_from_yaw(sent[3], sp.cfg)
                    sp.set_targets(gl, gr)
                else:
                    sp.set_targets(1.0, 1.0, ramp_ms=sp.LOSS_RAMP_MS)
                gl, gr = sp.current_gains
                self.lblSpatialMeter.setText(f"L {gl:.2f}  R {gr:.2f}")
            # 管线健康检查（稳定性）：出错/线程死亡 → 自动停机并提示
            if self.running and self.infer is not None:
                err = self.infer.error or (self.capture.error if self.capture else None)
                if err:
                    self.on_stop()
                    self.statBar.setText(str(err))
                elif not self.infer.is_alive() or (
                        self.capture is not None and not self.capture.is_alive()):
                    self.on_stop()
                    self.statBar.setText(t("err_thread"))
            sp = getattr(self, "spatial", None)
            if sp is not None and not sp.is_alive():
                self.spatial = None
                self.cbSpatialOn.blockSignals(True)
                self.cbSpatialOn.setChecked(False)
                self.cbSpatialOn.blockSignals(False)
                self.statBar.setText(t("spatial_dead"))
            if sent is not None:
                self.statBar.setToolTip("yaw pitch roll | TX TY TZ | 已发送")
                self.statBar.setText(
                    f"{sent[3]:+7.2f} {sent[4]:+7.2f} {sent[5]:+7.2f}    "
                    f"{sent[0]:+7.1f} {sent[1]:+7.1f} {sent[2]:+7.1f}    "
                    f"{self.sink.sent_count if self.sink else 0}"
                    + (f"  \u00d7{self.sink.error_count}" if self.sink and self.sink.error_count else ""))

        # FPS 由外部 tick 更新（简单实现：统计线程帧计数差）
        def tick_fps(self, frames: int):
            now = time.monotonic()
            dt = now - self._last_frames[0]
            if dt >= 1.0:
                self._last_fps = (frames - self._last_frames[1]) / dt
                self._last_frames[0] = now
                self._last_frames[1] = frames

    return MainWindow


def apply_app_font(app) -> None:
    """统一字体：Segoe UI + 微软雅黑回退（QSS 的 font-family 列表不可靠）。"""
    from PySide6.QtGui import QFont
    font = QFont("Segoe UI", 9)
    font.setFamilies(["Segoe UI", "Microsoft YaHei UI"])
    app.setFont(font)


def main(argv=None) -> int:
    # 任务栏身份（Windows）：设置 AppUserModelID，任务栏/快捷方式统一显示自定义图标
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(AUMID)
    except Exception:
        pass
    from PySide6.QtWidgets import QApplication

    app = QApplication(argv or sys.argv)
    app.setStyleSheet(build_qss(is_system_dark()))
    apply_app_font(app)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)

    setup_logging("INFO")
    install_crash_hooks(__version__)
    inst = SingleInstance()
    if not inst.acquire():
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(None, "Fylgja", t("err_exists"))
        return 5

    MainWindow = build_window()
    win = MainWindow()
    win.show()

    # FPS 统计驱动
    from PySide6.QtCore import QTimer
    t = QTimer()
    t.timeout.connect(lambda: win.tick_fps(
        win.infer.frames_processed if win.infer else 0))
    t.start(1000)

    rc = app.exec()
    inst.release()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
