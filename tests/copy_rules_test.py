"""Copy repository inputs into a prepared dependency source tree."""

import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(ROOT / "scripts/build_vllm.py")
VERSION = runpy.run_path(ROOT / "src/version.py")


class CopyRuleTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".build/copy-rules-tests"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "repository"
        (self.inputs / "src/kernels/example").mkdir(parents=True)
        (self.inputs / "src/kernels/example/__init__.py").write_bytes(b"")
        (self.inputs / "src/kernels/example/kernel.cu").write_bytes(b"// CUDA source fixture\n")
        (self.inputs / "src/runtime/vllm").mkdir(parents=True)
        (self.inputs / "src/runtime/vllm/adapter.py").write_bytes(b"# release adapter\n")
        self.manifest = {"copy": [
            {"source": "src/kernels/example", "destination": "csrc/rtx_spark/example"},
            {"source": "src/runtime/vllm/adapter.py", "destination": "vllm/adapter.py"},
        ]}

    def test_file_and_folder_rules_preserve_order_contents_and_empty_initializers(self):
        entries = BUILD["copy_inputs"](self.manifest, source_root=self.inputs)
        self.assertEqual([name for path, name in entries], [
            "csrc/rtx_spark/example/__init__.py", "csrc/rtx_spark/example/kernel.cu", "vllm/adapter.py"])
        output = self.root / "assembled"
        count = BUILD["copy_sources"](output, self.manifest, source_root=self.inputs)
        self.assertEqual(count, 3)
        for source, destination in entries:
            self.assertEqual((output / destination).read_bytes(), source.read_bytes())
        self.assertEqual((output / "csrc/rtx_spark/example/__init__.py").stat().st_size, 0)

    def test_late_existing_file_conflict_is_detected_before_any_copy(self):
        output = self.root / "assembled"
        conflict = output / "vllm/adapter.py"
        conflict.parent.mkdir(parents=True)
        conflict.write_bytes(b"existing source")
        with self.assertRaises((ValueError, RuntimeError, FileExistsError)):
            BUILD["copy_sources"](output, self.manifest, source_root=self.inputs)
        self.assertFalse((output / "csrc/rtx_spark/example").exists())
        self.assertEqual(conflict.read_bytes(), b"existing source")

    def test_overlapping_destinations_are_rejected(self):
        manifest = {"copy": [self.manifest["copy"][0],
                              {"source": "src/runtime/vllm/adapter.py",
                               "destination": "csrc/rtx_spark/example/adapter.py"}]}
        with self.assertRaises(ValueError):
            BUILD["copy_inputs"](manifest, source_root=self.inputs)

    def test_unsafe_or_missing_inputs_fail_before_assembly(self):
        for rule in ({"source": "../outside.py", "destination": "vllm/a.py"},
                     {"source": "src/runtime/vllm/adapter.py", "destination": "../outside.py"},
                     {"source": "missing.py", "destination": "vllm/a.py"}):
            with self.subTest(rule=rule), self.assertRaises((ValueError, FileNotFoundError)):
                BUILD["copy_inputs"]({"copy": [rule]}, source_root=self.inputs)

    def prepared_fixture(self):
        output = self.root / "assembled"
        BUILD["copy_sources"](output, self.manifest, source_root=self.inputs)
        for name in ("CMakeLists.txt", "vllm/__init__.py", "vllm/_version.py", "vllm/engine/arg_utils.py"):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# prepared source fixture\n")
        (output / "rtx-spark-source.json").write_text(json.dumps({
            "source_prepared": True, "commit": "fixture-pinned", "vllm_version": "0.32.0",
        }), encoding="utf-8")
        return output

    def source_globals(self):
        return patch.dict(BUILD["verify_source"].__globals__, {
            "MANIFEST": self.manifest, "ROOT": self.inputs,
            "COMMIT": "fixture-pinned", "UPSTREAM_VERSION": "0.32.0",
        })

    def test_changed_repository_input_cannot_reuse_a_prepared_source(self):
        output = self.prepared_fixture()
        with self.source_globals():
            BUILD["verify_source"](output)
            (self.inputs / "src/kernels/example/kernel.cu").write_bytes(b"// changed CUDA source\n")
            with self.assertRaisesRegex(RuntimeError, "fresh.*--source-dir"):
                BUILD["verify_source"](output)

    def test_removed_repository_input_cannot_leave_an_old_prepared_file(self):
        output = self.prepared_fixture()
        with self.source_globals():
            BUILD["verify_source"](output)
            (self.inputs / "src/kernels/example/kernel.cu").unlink()
            with self.assertRaisesRegex(RuntimeError, "fresh.*--source-dir"):
                BUILD["verify_source"](output)
            self.assertTrue((output / "csrc/rtx_spark/example/kernel.cu").is_file())

    def test_old_prepared_private_copies_cannot_duplicate_regular_source_payloads(self):
        output = self.prepared_fixture()
        legacy = output / "vllm/_rtx_spark/runtime/vllm/adapter.py"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"# obsolete copied adapter\n")
        with self.source_globals(), self.assertRaisesRegex(RuntimeError, "obsolete private copies.*fresh.*--source-dir"):
            BUILD["verify_source"](output)
        self.assertTrue(legacy.is_file())

    def test_the_dependency_manifest_copies_only_repository_vendor_sources(self):
        path = VERSION["manifest_path"]()
        self.assertEqual(path, ROOT / "patches/vllm/manifest.json")
        manifest = VERSION["load_manifest"]()
        self.assertEqual(len(manifest["patches"]), 4)
        self.assertEqual(len(manifest["copy"]), 6)
        self.assertFalse((ROOT / "targets").exists())
        inputs = BUILD["copy_inputs"]()
        self.assertEqual(len(inputs), 2041)
        self.assertTrue(all(source.is_relative_to(ROOT / "third_party/vllm-kernels")
                            for source, destination in inputs))
        self.assertFalse(any(destination.startswith("vllm/_rtx_spark/") for source, destination in inputs))


if __name__ == "__main__":
    unittest.main()
