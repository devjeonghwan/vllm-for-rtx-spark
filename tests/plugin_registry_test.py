"""CPU regression checks for late recipes and mutable option defaults."""

import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/load_source_package.py"))["load"]()

from _rtx_spark import plugin, recipes  # noqa: E402
from _rtx_spark.kernels import cute  # noqa: E402
from _rtx_spark.optimizations import Option, resolve_options  # noqa: E402
from _rtx_spark.runtime import bootstrap  # noqa: E402
from _rtx_spark.runtime import recipe_helpers  # noqa: E402
from _rtx_spark.kernels import providers  # noqa: E402


class PluginRegistryTests(unittest.TestCase):
    def test_late_recipes_do_not_publish_helper_paths_or_reinitialize_native_bindings(self):
        parent = ROOT / ".build/plugin-registry-tests"
        parent.mkdir(parents=True, exist_ok=True)
        before = {name for name in ("torch", "vllm", "cutlass", "triton") if name in sys.modules}
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            with patch.dict(os.environ, {"VLLM_RTX_SPARK_CACHE": str(Path(temporary) / "cache")}, clear=True), \
                    patch.object(sys, "platform", "win32"), patch.object(sys, "path", list(sys.path)), \
                    patch.dict(recipes._MODELS, {}, clear=True), \
                    patch.object(bootstrap, "_initialized", False), patch.object(cute, "install") as install, \
                    patch.object(providers, "initialize_providers") as initialize_providers:
                plugin.register()
                self.assertTrue(bootstrap._initialized)
                self.assertNotIn(str(recipe_helpers()), sys.path)
                dependency_paths = list(sys.path)
                recipes.register_model("late-helper-fixture", lambda arguments, options: None,
                                       lambda arguments, options: None)
                plugin.register()
                plugin.register()
                self.assertEqual(sys.path, dependency_paths)
                self.assertNotIn(str(recipe_helpers()), sys.path)
                install.assert_called_once_with()
                self.assertEqual(initialize_providers.call_count, 3)
                self.assertNotIn("MBX_PLE_MMAP_MODE", os.environ)
                self.assertNotIn("MBX_DEPTH_CFG", os.environ)
        self.assertEqual(before, {name for name in ("torch", "vllm", "cutlass", "triton") if name in sys.modules})

    def test_mutable_defaults_are_independent_between_resolutions(self):
        default = {"layers": [1, 2]}
        option = Option("settings", dict, default)
        first = resolve_options({}, (option,))
        first["settings"]["layers"].append(3)
        first["settings"]["new"] = True
        self.assertEqual(resolve_options({}, (option,))["settings"], {"layers": [1, 2]})
        self.assertEqual(default, {"layers": [1, 2]})
        supplied = {"layers": [4]}
        self.assertIs(resolve_options({"settings": supplied}, (option,))["settings"], supplied)


if __name__ == "__main__":
    unittest.main()
