"""GUI 离屏冒烟：构建窗口、渲染一帧、截图存 PNG。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402

from fylgja.gui import apply_app_font, build_window  # noqa: E402
from fylgja.gui_style import build_qss, is_system_dark  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(is_system_dark()))
    apply_app_font(app)
    MainWindow = build_window()
    win = MainWindow()
    win.show()
    app.processEvents()
    app.processEvents()
    pm = win.grab()
    out = ROOT / "out" / "gui" / "smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.save(str(out))
    print(f"window size: {win.size().width()}x{win.size().height()}")
    print(f"saved {out}")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
