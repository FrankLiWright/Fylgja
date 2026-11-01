"""Fylgja GUI 启动入口（双击运行，无命令行窗口）。

.pyw 由系统关联的 pythonw.exe 执行（无控制台）；若不是本仓库 .venv 的
Python（缺依赖），则静默转交给 .venv 的 pythonw.exe 重新拉起。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYW = ROOT / ".venv" / "Scripts" / "pythonw.exe"

if Path(sys.executable).resolve() != VENV_PYW.resolve() and VENV_PYW.exists():
    subprocess.Popen([str(VENV_PYW), str(Path(__file__).resolve())],
                     cwd=str(ROOT), close_fds=True)
    raise SystemExit(0)

sys.path.insert(0, str(ROOT / "src"))

from fylgja.gui import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
