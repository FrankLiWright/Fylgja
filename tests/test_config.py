"""配置持久化行为（方案书 9.2/9.3/9.4）：类型解析、校验回退、未知键保留、重置。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fylgja.config import (  # noqa: E402
    Config, ConfigStore, FilterParams, InMemoryScalarStore, JsonStructuredStore,
)


def make_store(tmp: str) -> ConfigStore:
    return ConfigStore(
        scalars=InMemoryScalarStore(),
        structured=JsonStructuredStore(Path(tmp) / "config.json"),
    )


class TestRoundTrip(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            scalars = InMemoryScalarStore()
            store = ConfigStore(scalars, JsonStructuredStore(Path(tmp) / "config.json"))
            cfg = Config()
            cfg.engine = "C"
            cfg.sink_port = 5555
            cfg.camera_width = 1280
            cfg.filter = FilterParams(min_cutoff=0.8, beta=0.3, d_cutoff=1.2)
            cfg.axis_invert = (True, False, False, False, False, True)
            store.save(cfg)

            # 共享标量存储 = 模拟注册表持久化；JSON 重新打开 = 模拟重启
            store2 = ConfigStore(scalars, JsonStructuredStore(Path(tmp) / "config.json"))
            got = store2.load()
            self.assertEqual(got.engine, "C")
            self.assertEqual(got.sink_port, 5555)
            self.assertEqual(got.camera_width, 1280)
            self.assertAlmostEqual(got.filter.min_cutoff, 0.8)
            self.assertEqual(got.axis_invert, (True, False, False, False, False, True))

    def test_defaults_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = make_store(tmp).load()
            self.assertEqual(got, Config())


class TestValidation(unittest.TestCase):
    def test_illegal_scalar_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(tmp)
            store._scalars.set("sink_port", "99999")     # 越界
            store._scalars.set("engine", "X")            # 非法枚举
            store._scalars.set("camera_fps", "abc")      # 解析失败
            got = store.load()
            self.assertEqual(got.sink_port, 4242)
            self.assertEqual(got.engine, "auto")
            self.assertEqual(got.camera_fps, 30.0)

    def test_illegal_structured_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "filter": {"min_cutoff": "bad", "beta": 0.9},
                "axis_invert": [1, 0],
            }), encoding="utf-8")
            store = ConfigStore(InMemoryScalarStore(), JsonStructuredStore(path))
            got = store.load()
            self.assertEqual(got.filter.min_cutoff, 1.0)  # 回默认
            self.assertAlmostEqual(got.filter.beta, 0.9)  # 合法值保留
            self.assertEqual(got.axis_invert, (False,) * 6)  # 整体非法回默认

    def test_corrupt_json_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            store = ConfigStore(InMemoryScalarStore(), JsonStructuredStore(path))
            self.assertEqual(store.load(), Config())  # 不得启动失败（9.3）


class TestMigration(unittest.TestCase):
    def test_unknown_keys_preserved(self):
        """未知键保留原值不动（方案书 9.3）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "future_key": {"nested": True},
                "filter": {"min_cutoff": 2.0},
            }), encoding="utf-8")
            store = ConfigStore(InMemoryScalarStore(), JsonStructuredStore(path))
            cfg = store.load()
            self.assertAlmostEqual(cfg.filter.min_cutoff, 2.0)
            store.save(cfg)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["future_key"], {"nested": True})  # 未触碰

    def test_downgrade_loads_known_keys(self):
        """回退旧版本不得因新 schema 拒绝加载（9.3）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
            store = ConfigStore(InMemoryScalarStore(), JsonStructuredStore(path))
            self.assertEqual(store.load(), Config())

    def test_reset_clears_both_stores(self):
        """重置必须同时清标量与 JSON（方案书 9.2）。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store(tmp)
            cfg = Config(engine="C", sink_port=5555)
            store.save(cfg)
            store.reset_defaults()
            self.assertEqual(store.load(), Config())
            self.assertEqual(store._scalars.keys(), [])


if __name__ == "__main__":
    unittest.main()
