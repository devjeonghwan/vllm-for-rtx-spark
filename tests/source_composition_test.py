"""Copy repository inputs into a prepared dependency source tree."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(ROOT / "scripts/build_vllm.py")
APPLY = runpy.run_path(ROOT / "scripts/apply_source_changes.py")["apply_source_changes"]
VERSION = runpy.run_path(ROOT / "src/version.py")


class CopyRuleTests(unittest.TestCase):
    def test_manifest_copy_roles_exclude_duplicate_private_runtime_payloads(self):
        inputs = BUILD['copy_inputs']()
        self.assertTrue(inputs)
        destinations = [destination for _, destination in inputs]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertTrue(all(source.is_relative_to(ROOT / 'third_party/vllm-kernels') for source, _ in inputs))
        self.assertFalse(any(destination.startswith('vllm/_rtx_spark/') for destination in destinations))

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


class SourceChangesTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / ".build/source-changes-tests"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def source(self, name):
        source = self.root / "prepared" / name
        source.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "--local", "core.autocrlf", "false"], check=True)
        return source

    def manifest(self, name, data):
        path = self.root / "patches" / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def patch(self, folder, name, before, after):
        (folder / name).write_bytes((
            "diff --git a/value.txt b/value.txt\n"
            "--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n"
            f"-{before}\n+{after}\n").encode())

    def test_explicit_library_manifests_apply_ordered_patches_and_repository_copies(self):
        first = self.source("first")
        second = self.source("second")
        (first / "value.txt").write_bytes(b"one\n")
        owned = self.root / "src/kernels/example/native"
        owned.mkdir(parents=True)
        (owned / "kernel.cu").write_bytes(b"// CUDA source fixture; never compiled\n")
        (owned / "empty.h").write_bytes(b"")
        manifest = self.manifest("first", {"patches": ["first.patch", "second.patch"], "copy": [
            {"source": "src/kernels/example/native", "destination": "csrc/example"}]})
        self.patch(manifest.parent, "first.patch", "one", "two")
        self.patch(manifest.parent, "second.patch", "two", "three")
        other = self.manifest("second", {"copy": [
            {"source": "src/kernels/example/native/empty.h", "destination": "include/empty.h"}]})
        self.manifest("unselected", {"patches": ["must-not-be-read.patch"]})
        result = APPLY(first, manifest, source_root=self.root)
        self.assertEqual(result, {"patches_applied": 2, "copied_files": 2})
        self.assertEqual((first / "value.txt").read_bytes(), b"three\n")
        self.assertEqual((first / "csrc/example/kernel.cu").read_bytes(), (owned / "kernel.cu").read_bytes())
        self.assertFalse((second / "include").exists())
        self.assertEqual(APPLY(second, other, source_root=self.root), {"patches_applied": 0, "copied_files": 1})
        self.assertEqual((second / "include/empty.h").read_bytes(), b"")

    def test_missing_later_patch_fails_before_applying_the_first(self):
        source = self.source("missing")
        (source / "value.txt").write_bytes(b"before\n")
        manifest = self.manifest("missing", {"patches": ["valid.patch", "missing.patch"]})
        self.patch(manifest.parent, "valid.patch", "before", "after")
        with self.assertRaises(FileNotFoundError):
            APPLY(source, manifest, source_root=self.root)
        self.assertEqual((source / "value.txt").read_bytes(), b"before\n")

    def test_applicator_requires_an_isolated_prepared_tree(self):
        source = self.root / "not-a-repository"
        source.mkdir()
        manifest = self.manifest("plain", {"patches": [], "copy": []})
        with self.assertRaisesRegex(ValueError, "own Git root"):
            APPLY(source, manifest, source_root=self.root)

    def test_local_upstream_crlf_settings_cannot_change_patch_inputs(self):
        upstream = self.source("upstream")

        def git(*arguments):
            return subprocess.check_output(["git", "-C", str(upstream), *arguments])

        before = b"cmake_minimum_required(VERSION 3.26)\nset(FIXTURE before)\n"
        (upstream / "CMakeLists.txt").write_bytes(before)
        for name in ("vllm/__init__.py", "vllm/_version.py", "vllm/engine/arg_utils.py"):
            path = upstream / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# source fixture\n")
        git("add", "--all")
        git("-c", "user.name=Source fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Source fixture")
        commit = git("rev-parse", "HEAD").decode().strip()
        git("config", "--local", "core.autocrlf", "true")
        git("config", "--local", "core.eol", "crlf")
        # Prove that the unscoped archive reproduces the build failure's CRLF
        # input before exercising the builder's normal prepare path.
        archive = git("archive", "--format=tar", commit)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            self.assertEqual(bundle.extractfile("CMakeLists.txt").read(), before.replace(b"\n", b"\r\n"))

        repository = self.root / "repository"
        manifest = self.manifest("archive", {
            "vllm": {"repository": str(upstream), "commit": commit, "version": "0.32.0"},
            "patches": ["fixture.patch"], "copy": [],
        })
        (manifest.parent / "fixture.patch").write_bytes(
            b"diff --git a/CMakeLists.txt b/CMakeLists.txt\n"
            b"--- a/CMakeLists.txt\n+++ b/CMakeLists.txt\n@@ -1,2 +1,2 @@\n"
            b" cmake_minimum_required(VERSION 3.26)\n-set(FIXTURE before)\n+set(FIXTURE after)\n")
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        prepared = self.root / "assembled"
        with patch.dict(BUILD["prepare"].__globals__, {
            "MANIFEST": spec, "ROOT": repository, "COMMIT": commit, "UPSTREAM_VERSION": "0.32.0",
            "_VERSION": {"manifest_path": lambda: manifest},
        }):
            result = BUILD["prepare"](prepared, upstream)
        self.assertTrue(result["source_prepared"])
        self.assertEqual((prepared / "CMakeLists.txt").read_bytes(), before.replace(b"before", b"after"))
        self.assertEqual(git("config", "--local", "core.autocrlf").strip(), b"true")
        self.assertEqual(git("config", "--local", "core.eol").strip(), b"crlf")


if __name__ == "__main__":
    unittest.main()
