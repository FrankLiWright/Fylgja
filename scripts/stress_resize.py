"""预览缩放压测：模拟窗口快速连续 resize 触发重绘洪水，验证不崩溃。

用法：.venv/Scripts/python.exe scripts/stress_resize.py [--rounds 300]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fylgja.gui import apply_app_font, build_window  # noqa: E402
from fylgja.gui_style import build_qss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(False))
    apply_app_font(app)
    win = build_window()()
    win.show()

    # 注入画面与姿态（含坐标轴/关键点绘制路径）
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    lm = np.random.default_rng(1).random((478, 3)).astype(np.float32)
    sent = (0.5, -0.3, -40.0, 12.0, -5.0, 3.0)
    win.preview.update_frame(frame, lm, sent, "99.9  7.7ms  42")

    t0 = time.perf_counter()
    try:
        for i in range(args.rounds):
            w = 700 + (i * 37) % 500
            h = 480 + (i * 53) % 300
            win.resize(w, h)
            app.processEvents()          # 触发 paintEvent
            if i % 50 == 0:
                win.preview.update_frame(frame, lm, sent, f"{i}")
                app.processEvents()
        # 0 尺寸边界
        win.resize(0, 0)
        app.processEvents()
        win.resize(900, 600)
        app.processEvents()
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1
    dt = time.perf_counter() - t0
    print(f"PASS: {args.rounds} 次 resize + 重绘，{dt:.2f}s（{args.rounds / dt:.0f} 次/秒）")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
