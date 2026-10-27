"""The same applicator must prepare explicit, independent library trees."""
import io
import json
from pathlib import Path
import runpy
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
APPLY = runpy.run_path(ROOT / "scripts/apply_source_changes.py")["apply_source_changes"]
BUILD = runpy.run_path(ROOT / "scripts/build_vllm.py")


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
