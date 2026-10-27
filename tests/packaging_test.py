"""Source release inventories must include new build inputs without Git staging."""

from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = runpy.run_path(ROOT / "scripts/package_release.py")


class SourceInventoryTests(unittest.TestCase):
    def test_checkout_inventory_contains_manifest_copy_inputs_vendor_and_empty_sources(self):
        entries = PACKAGER["source_entries"]()
        names = {name for path, name in entries}
        version = PACKAGER["_VERSION"]
        builder = runpy.run_path(ROOT / "scripts/build_vllm.py")
        required = {
            "src/version.py", "src/plugin.py", "src/optimizations/__init__.py",
            "patches/vllm/manifest.json",
            "src/runtime/vllm/adapter.py", "src/recipes/qwen3_8/helpers/mbx_loader.py",
            "third_party/python/openai_harmony/openai_harmony.pyd",
            "third_party/python/llguidance/_lib.pyd",
        }
        required.update(path.relative_to(ROOT).as_posix() for path, destination in builder["copy_inputs"]())
        self.assertEqual(version["manifest_path"](), ROOT / "patches/vllm/manifest.json")
        self.assertLessEqual(required, names)
        empty_sources = {path.relative_to(ROOT).as_posix()
                         for folder in ("src", "third_party", "patches")
                         for path in (ROOT / folder).rglob("__init__.py")
                         if path.is_file() and path.stat().st_size == 0}
        self.assertTrue(empty_sources, "The source fixture includes real empty package initializers")
        self.assertLessEqual(empty_sources, names)
        self.assertEqual(len(names), len(entries))
        self.assertFalse(any("__pycache__" in name or name.endswith((".pyc", ".pyo", ".log"))
                             for name in names))

    def test_new_unstaged_inputs_are_included_and_local_files_are_excluded(self):
        parent = ROOT / ".build/source-inventory-tests"
        parent.mkdir(parents=True, exist_ok=True)
        included = {
            "README.md", "sources.lock.json", ".gitignore", ".gitattributes",
            "src/recipes/new_model/__init__.py", "src/runtime/new_helper.py",
            "third_party/python/new_vendor/__init__.py", "patches/vllm/manifest.json",
            "src/recipes/new_model/helpers/__init__.py", "src/runtime/vllm/adapter.py",
            "assets/native/new.dll", "patches/vllm/new.patch", "configs/new.yaml",
            "requirements/new.txt", "scripts/new.py", "tests/new_test.py", "docs/build.md",
        }
        excluded = {
            "benchmarks/run.py", "reports/private.md", "dist/package.whl", ".git/config", ".env",
            "targets/vllm/obsolete.json",
            "src/__pycache__/runtime.pyc", "src/cached.pyc", "src/cached.pyo", "src/.git/HEAD",
            "third_party/python/old.egg-info/PKG-INFO", "third_party/.cache/download.bin",
            "docs/debug.LOG", "scripts/local-build.json", "scripts/.env", "scripts/.env.local",
            "assets/.build/temporary.bin", "src/.venv/Lib/private.py", "src/.env/private.txt",
            "tests/.pytest_cache/state", "scripts/.ruff_cache/state",
        }
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            for name in included | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"" if name.endswith("__init__.py") else b"fixture")
            with patch("subprocess.run", side_effect=AssertionError("Inventory must not run commands")), \
                    patch("subprocess.check_output", side_effect=AssertionError("Inventory must not query Git")), \
                    patch("zipfile.ZipFile", side_effect=AssertionError("Inventory must not create archives")):
                entries = PACKAGER["source_entries"](root)
            self.assertEqual({name for path, name in entries}, included)
            self.assertEqual([name for path, name in entries], sorted(included))


if __name__ == "__main__":
    unittest.main()
