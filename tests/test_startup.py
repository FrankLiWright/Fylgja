"""启动验证流程（方案书 6/6.1）：引擎选择矩阵、预授权回退、摄像头选择回路。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.camera_enum import CameraDevice  # noqa: E402
from fylgja.config import Config  # noqa: E402
from fylgja.errors import CameraError, SelfCheckError  # noqa: E402
from fylgja.startup import FALLBACK_BANNER, open_camera, select_engine  # noqa: E402


class FakeEngine:
    def __init__(self, ok=True, log=None):
        self.ok = ok
        self.checked = 0
        self.closed = False
        if log is not None:
            log.append(self)

    def self_check(self):
        self.checked += 1
        if not self.ok:
            raise RuntimeError("self-check-boom")

    def close(self):
        self.closed = True


def factories(log=None, *, n_ok=True, c_ok=True):
    return {
        "N": lambda: FakeEngine(n_ok, log),
        "C": lambda: FakeEngine(c_ok, log),
    }


class TestSelectEngine(unittest.TestCase):
    def test_pref_c_ok(self):
        log = []
        sel = select_engine(Config(engine="C"), factories(log))
        self.assertEqual(sel.name, "C")
        self.assertIsNone(sel.fallback_banner)
        self.assertEqual(sel.chosen_by, "config")
        self.assertEqual(sel.engine.checked, 1)  # 经完整自检

    def test_pref_c_fail_is_explicit_error(self):
        """用户选择 C 失败不得静默换 N（方案书 6）。"""
        log = []
        with self.assertRaises(SelfCheckError):
            select_engine(Config(engine="C"), factories(log, c_ok=False))
        self.assertEqual(len(log), 1)  # 只试了 C

    def test_pref_n_ok(self):
        log = []
        sel = select_engine(Config(engine="N"), factories(log))
        self.assertEqual(sel.name, "N")

    def test_pref_n_fail_manual_no_prompt_raises(self):
        log = []
        with self.assertRaises(SelfCheckError):
            select_engine(Config(engine="N"), factories(log, n_ok=False))
        self.assertEqual(len(log), 1)  # 不得静默改试 C

    def test_pref_n_fail_autostart_fallback_to_c(self):
        """6.1 预授权回退：自启动 + 开关开 + N 失败 → C + 常驻横幅。"""
        log = []
        cfg = Config(engine="N", autostart_fallback_c=True)
        sel = select_engine(cfg, factories(log, n_ok=False), is_autostart=True)
        self.assertEqual(sel.name, "C")
        self.assertEqual(sel.fallback_banner, FALLBACK_BANNER)
        self.assertEqual(sel.chosen_by, "fallback")

    def test_pref_n_fail_autostart_fallback_disabled(self):
        """6.1 开关关闭：停留在失败（报错），不得回退。"""
        cfg = Config(engine="N", autostart_fallback_c=False)
        with self.assertRaises(SelfCheckError):
            select_engine(cfg, factories(n_ok=False), is_autostart=True)

    def test_auto_prefers_n(self):
        sel = select_engine(Config(engine="auto"), factories())
        self.assertEqual(sel.name, "N")

    def test_auto_fail_prompts_explicit_choice(self):
        """auto 失败须显式选择（方案书 6），选择经完整自检并持久化标识。"""
        log = []
        sel = select_engine(Config(engine="auto"), factories(log, n_ok=False),
                            prompt=lambda _msg: "C")
        self.assertEqual(sel.name, "C")
        self.assertEqual(sel.chosen_by, "prompt")
        self.assertEqual(sel.engine.checked, 1)  # 选 C 后重新完整自检
        self.assertEqual([a.engine for a in sel.attempts], ["N", "C"])

    def test_auto_fail_prompt_choice_also_fails(self):
        with self.assertRaises(SelfCheckError):
            select_engine(Config(engine="auto"), factories(n_ok=False, c_ok=False),
                          prompt=lambda _msg: "C")

    def test_invalid_prompt_answer_raises(self):
        with self.assertRaises(SelfCheckError):
            select_engine(Config(engine="auto"), factories(n_ok=False),
                          prompt=lambda _msg: "X")

    def test_engine_switch_requires_self_check(self):
        """引擎切换必须经完整引擎自检（方案书 6）：构造出的引擎都要 check。"""
        log = []
        select_engine(Config(engine="auto"), factories(log, n_ok=False),
                      prompt=lambda _msg: "N" if False else "C")
        for eng in log:
            self.assertEqual(eng.checked, 1)


class TestOpenCamera(unittest.TestCase):
    DEVICES = [
        CameraDevice(0, "Cam A", "@device:pnp:a"),
        CameraDevice(1, "Cam B", "@device:pnp:b"),
    ]

    def _cfg(self, **kw):
        cfg = Config()
        for k, v in kw.items():
            setattr(cfg, k, v)
        return cfg

    def test_resolved_by_path_and_opened(self):
        fake = mock.MagicMock()
        fake.open.return_value = mock.MagicMock()
        with mock.patch("fylgja.startup.CaptureThread", return_value=fake):
            sel = open_camera(self._cfg(camera_path="@device:pnp:b"), devices=self.DEVICES)
        self.assertEqual(sel.device.name, "Cam B")
        fake.open.assert_called_once()

    def test_missing_device_without_prompt_raises(self):
        with self.assertRaises(CameraError):
            open_camera(self._cfg(camera_path="@device:gone"), devices=self.DEVICES)

    def test_missing_device_prompts_reselect(self):
        fake = mock.MagicMock()
        fake.open.return_value = mock.MagicMock()
        with mock.patch("fylgja.startup.CaptureThread", return_value=fake):
            sel = open_camera(
                self._cfg(camera_path="@device:gone"), devices=self.DEVICES,
                prompt=lambda _msg: "0",
            )
        self.assertEqual(sel.device.name, "Cam A")  # 显式重选，非静默

    def test_open_failure_prompt_reselect(self):
        fake = mock.MagicMock()
        fake.open.side_effect = [RuntimeError("occupied"), mock.MagicMock()]
        with mock.patch("fylgja.startup.CaptureThread", return_value=fake):
            sel = open_camera(
                self._cfg(camera_path="@device:pnp:a"), devices=self.DEVICES,
                prompt=lambda _msg: "1",
            )
        self.assertEqual(sel.device.name, "Cam B")


if __name__ == "__main__":
    unittest.main()
