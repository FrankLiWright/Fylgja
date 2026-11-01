"""真实桌面启动 GUI：显示窗口，2 s 后抓取截图供验证，随后保持运行。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fylgja.gui import app_icon, apply_app_font, build_window  # noqa: E402
from fylgja.gui_style import build_qss, is_system_dark  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(is_system_dark()))
    apply_app_font(app)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)
    MainWindow = build_window()
    win = MainWindow()
    win.show()

    def snap():
        pm = win.grab()
        out = ROOT / "out" / "gui" / "live.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        pm.save(str(out))
        print(f"[gui_launch] screenshot -> {out}", flush=True)

    QTimer.singleShot(2500, snap)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
