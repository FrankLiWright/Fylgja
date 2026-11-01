"""崩溃取证与单实例（方案书 5.4）：钩子写崩溃日志、互斥体防二开。"""
from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.crash import install_crash_hooks  # noqa: E402
from fylgja.single_instance import SingleInstance  # noqa: E402


class TestCrashLog(unittest.TestCase):
    def test_sys_hook_writes_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            install_crash_hooks("0.1.0-test", engine_name_provider=lambda: "C", log_dir=tmp)
            try:
                raise ZeroDivisionError("crash-test")
            except ZeroDivisionError:
                exc = sys.exc_info()
            sys.excepthook(*exc)

            logs = list(Path(tmp).glob("crash-*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            for field in ("time:", "exception_type: ZeroDivisionError",
                          "software_version: 0.1.0-test", "selected_engine: C",
                          "os:", "crash-test"):
                self.assertIn(field, text)

    def test_thread_hook_writes_crash_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            install_crash_hooks("0.1.0-test", log_dir=tmp)

            def boom():
                raise RuntimeError("thread-crash")

            t = threading.Thread(target=boom, name="worker-test")
            t.start()
            t.join(2.0)

            logs = list(Path(tmp).glob("crash-*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertIn("thread: thread:worker-test", text)
            self.assertIn("thread-crash", text)


class TestSingleInstance(unittest.TestCase):
    def test_second_acquire_rejected(self):
        # 同进程二次 CreateMutexW 同名 → ERROR_ALREADY_EXISTS（方案书 5.4 语义）
        name = "Fylgja-test-mutex"
        first = SingleInstance(name)
        second = SingleInstance(name)
        try:
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())  # 二次启动不得开第二实例
        finally:
            first.release()
            second.release()

    def test_release_allows_reacquire(self):
        name = "Fylgja-test-mutex2"
        first = SingleInstance(name)
        self.assertTrue(first.acquire())
        first.release()
        again = SingleInstance(name)
        try:
            self.assertTrue(again.acquire())  # 退出后可重新启动
        finally:
            again.release()


if __name__ == "__main__":
    unittest.main()
