"""Recipe helpers remain isolated from unrelated packages and flat imports."""

import ast
import importlib
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(ROOT / "scripts/load_source_package.py")["load"]()
HELPERS = ROOT / "src/recipes/qwen3_8/helpers"
PACKAGE = "_rtx_spark.recipes.qwen3_8.helpers"


class HelperImportsTests(unittest.TestCase):
    def test_two_recipe_packages_can_use_the_original_reader_names_independently(self):
        parent = ROOT / ".build/helper-import-tests"
        parent.mkdir(parents=True, exist_ok=True)
        flat = {name: ModuleType(name) for name in ("loader_core", "raw_reader", "csf_codec")}
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            root = Path(temporary)
            for package in ("recipe_first", "recipe_second"):
                directory = root / package / "helpers"
                directory.mkdir(parents=True)
                (directory.parent / "__init__.py").write_bytes(b"")
                (directory / "__init__.py").write_bytes(b"")
                for name in ("loader_core.py", "raw_reader.py", "csf_codec.py"):
                    shutil.copyfile(HELPERS / name, directory / name)
            with patch.object(sys, "path", [str(root), *sys.path]), patch.dict(sys.modules, flat):
                first = importlib.import_module("recipe_first.helpers.loader_core")
                second = importlib.import_module("recipe_second.helpers.loader_core")
                self.assertEqual(first.CSFCheckpoint.__module__, "recipe_first.helpers.raw_reader")
                self.assertEqual(second.CSFCheckpoint.__module__, "recipe_second.helpers.raw_reader")
                self.assertIsNot(first.CSFCheckpoint, second.CSFCheckpoint)
                for name, poison in flat.items():
                    self.assertIs(sys.modules[name], poison)
                self.assertNotIn(str(root / "recipe_first/helpers"), sys.path)
                self.assertNotIn(str(root / "recipe_second/helpers"), sys.path)

    def test_loader_registration_uses_relative_core_despite_poisoned_global_names(self):
        registrations = []
        loader = ModuleType("vllm.model_executor.model_loader")
        default = ModuleType("vllm.model_executor.model_loader.default_loader")

        class DefaultModelLoader:
            pass

        default.DefaultModelLoader = DefaultModelLoader
        loader.register_model_loader = lambda name: lambda cls: registrations.append((name, cls)) or cls
        legacy = ModuleType("_mbx_windows_loader_candidate_v1")
        legacy.__path__ = ["another-model"]
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.model_executor": ModuleType("vllm.model_executor"),
            "vllm.model_executor.model_loader": loader,
            "vllm.model_executor.model_loader.default_loader": default,
            "_mbx_windows_loader_candidate_v1": legacy,
            **{name: ModuleType(name) for name in ("mbx_loader", "loader_core", "raw_reader", "csf_codec")},
        }
        before = {name for name in ("torch", "triton", "cutlass") if name in sys.modules}
        package = importlib.import_module(PACKAGE)
        sentinel = object()
        previous = getattr(package, "mbx_loader", sentinel)
        try:
            with patch.dict(sys.modules, modules):
                sys.modules.pop(PACKAGE + ".mbx_loader", None)
                imported = importlib.import_module(PACKAGE + ".mbx_loader")
                self.assertEqual(registrations, [("mbx", imported.MbxModelLoader)])
                self.assertTrue(issubclass(imported.MbxModelLoader, DefaultModelLoader))
                self.assertEqual(imported.MbxModelLoader.__module__, PACKAGE + ".mbx_loader")
                self.assertEqual(imported.loader_core.__name__, PACKAGE + ".loader_core")
                self.assertIs(sys.modules["_mbx_windows_loader_candidate_v1"], legacy)
                for name in ("mbx_loader", "loader_core", "raw_reader", "csf_codec"):
                    self.assertIs(sys.modules[name], modules[name])
        finally:
            if previous is sentinel:
                delattr(package, "mbx_loader")
            else:
                package.mbx_loader = previous
        self.assertEqual(before, {name for name in ("torch", "triton", "cutlass") if name in sys.modules})

    def test_patched_vllm_has_no_flat_recipe_helper_imports(self):
        helpers = {path.stem for path in HELPERS.glob("*.py") if path.stem != "__init__"}
        for patch_file in (ROOT / "patches/vllm").glob("*.patch"):
            for line in patch_file.read_text(encoding="utf-8").splitlines():
                if not line.startswith("+") or line.startswith("+++"):
                    continue
                statement = line[1:].strip()
                if not statement.startswith(("import ", "from ")):
                    continue
                try:
                    node = ast.parse(statement).body[0]
                except SyntaxError:
                    continue  # A multiline import is checked by source composition.
                if isinstance(node, ast.Import):
                    self.assertFalse(helpers.intersection(alias.name for alias in node.names), line)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, helpers, line)

    def test_flashinfer_windows_sampling_uses_the_packaged_helper(self):
        source = ROOT / "third_party/python/flashinfer/jit/sampling.py"
        function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.FunctionDef) and node.name == "gen_sampling_module")
        namespace = {"JitSpec": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        packaged = ModuleType("vllm._rtx_spark.recipes.qwen3_8.helpers.native_sampling")
        result = object()
        packaged.gen_native_sampling_spec = lambda: result
        flat = ModuleType("native_sampling")
        with patch.object(sys, "platform", "win32"), patch.dict(sys.modules, {
                packaged.__name__: packaged, "native_sampling": flat}):
            self.assertIs(namespace["gen_sampling_module"](), result)


if __name__ == "__main__":
    unittest.main()
