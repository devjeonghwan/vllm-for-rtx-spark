"""External input preparation contracts; synthetic archives, no network or GPU."""
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import runpy
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = runpy.run_path(ROOT / "scripts/prepare_dependencies.py")
NAMESPACE = TOOLS["lock"].__globals__


@contextmanager
def fixture():
    parent = ROOT / ".build/dependency-tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as folder:
        root = Path(folder)
        (root / "cache").mkdir()
        (root / "assets/python/native_module").mkdir(parents=True)
        (root / "assets/python/native_module/extension.pyd").write_bytes(b"preserved native fixture")
        with patch.dict(NAMESPACE, ROOT=root), patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            yield root


def archive_spec(path, kind):
    return {"version": "1.0", "filename": path.name, "kind": kind,
            "url": "https://example.invalid/" + path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def set_lock(root, entries):
    (root / "sources.lock.json").write_text(json.dumps({"dependencies": entries}), encoding="utf-8")


def runtime_archive(root):
    archive = root / "cache/flashinfer.whl"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("flashinfer/core.py", "value = 'original'\n")
        stream.writestr("flashinfer/data.json", '{"setting": 1}')
        stream.writestr("flashinfer/kernel.cu", "build-only source")
    spec = archive_spec(archive, "bdist_wheel")
    spec.update(runtime=True, modules=["flashinfer"], patches=[], native_patches=[])
    set_lock(root, {"flashinfer-python": spec})
    return spec


class DependencyPreparationTests(unittest.TestCase):
    def test_verified_offline_runtime_preserves_native_payload_and_cache(self):
        with fixture() as root:
            runtime_archive(root)
            out = root / "runtime"
            TOOLS["prepare_runtime"](output=out, cache=root / "cache", offline=True)
            self.assertEqual((out / "native_module/extension.pyd").read_bytes(), b"preserved native fixture")
            self.assertEqual((out / "flashinfer/core.py").read_text(), "value = 'original'\n")
            self.assertFalse((out / "flashinfer/kernel.cu").exists())
            self.assertEqual(TOOLS["prepare_runtime"](output=out, cache=root / "cache", offline=True), out)

    def test_archive_digest_failure_happens_before_output_creation(self):
        with fixture() as root:
            runtime_archive(root)
            (root / "cache/flashinfer.whl").write_bytes(b"corrupt cached archive")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                TOOLS["prepare_runtime"](output=root / "runtime", cache=root / "cache", offline=True)
            self.assertFalse((root / "runtime").exists())

    def test_offline_cache_miss_never_downloads(self):
        with fixture() as root:
            spec = runtime_archive(root)
            with self.assertRaisesRegex(FileNotFoundError, "not cached"):
                TOOLS["fetch"](spec, root / "absent", offline=True)

    def test_changed_or_incomplete_outputs_are_preserved_and_rejected(self):
        with fixture() as root:
            runtime_archive(root)
            out = root / "runtime"
            TOOLS["prepare_runtime"](output=out, cache=root / "cache", offline=True)
            changed = out / "flashinfer/core.py"
            changed.write_text("local experiment\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                TOOLS["prepare_runtime"](output=out, cache=root / "cache", offline=True)
            self.assertEqual(changed.read_text(), "local experiment\n")
            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                TOOLS["prepare_runtime"](output=incomplete, cache=root / "cache", offline=True)

    def test_patch_application_is_independent_of_surrounding_git_settings(self):
        with fixture() as root:
            spec = runtime_archive(root)
            (root / "fix.patch").write_bytes(b"--- a/flashinfer/core.py\n+++ b/flashinfer/core.py\n@@ -1 +1 @@\n-value = 'original'\n+value = 'patched'\n")
            spec["patches"] = ["fix.patch"]
            set_lock(root, {"flashinfer-python": spec})
            TOOLS["prepare_runtime"](output=root / "runtime", cache=root / "cache", offline=True)
            self.assertEqual((root / "runtime/flashinfer/core.py").read_bytes(), b"value = 'patched'\n")

    def test_record_is_regenerated_for_only_shipped_package_files(self):
        with fixture() as root:
            spec = runtime_archive(root)
            archive = root / "cache/library.whl"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("library/__init__.py", "# original code\n")
                stream.writestr("library/locale.mo", "excluded locale")
                stream.writestr("library-1.0.dist-info/METADATA", "Name: library\nVersion: 1.0\n")
                stream.writestr("library-1.0.dist-info/RECORD", "outdated RECORD\n")
                stream.writestr("library-1.0.dist-info/LICENSE", "original license")
            library = dict(archive_spec(archive, "bdist_wheel"), runtime=True,
                           modules=["library"], dist_info="library-1.0.dist-info")
            set_lock(root, {"flashinfer-python": spec, "library": library})
            TOOLS["prepare_runtime"](output=root / "runtime", cache=root / "cache", offline=True)
            record = (root / "runtime/library-1.0.dist-info/RECORD").read_text()
            self.assertIn("library/__init__.py,sha256=", record)
            self.assertIn("library-1.0.dist-info/LICENSE,sha256=", record)
            self.assertNotIn("locale.mo", record)
            self.assertEqual((root / "runtime/library-1.0.dist-info/LICENSE").read_text(), "original license")

    def test_native_build_layout_preserves_only_the_original_input_selection(self):
        with fixture() as root:
            archive = root / "cache/native.whl"
            entries = {
                "cccl/library/detail.inl": b"original CCCL implementation",
                "cccl/tools/check.py": b"original CCCL tool",
                "include/flashinfer/math.cuh": b"original shared header",
                "csrc/sampling.cu": b"original sampler",
                "csrc/kda/tool.py": b"original native source tool",
                "csrc/unused.jinja": b"excluded JIT template",
                "cutlass/include/generator.inl": b"excluded generator",
                "cutlass/tools/unused.py": b"excluded generator",
            }
            with zipfile.ZipFile(archive, "w") as stream:
                for name, raw in entries.items():
                    stream.writestr("flashinfer/data/" + name, raw)
            spec = dict(archive_spec(archive, "bdist_wheel"), native_patches=[])
            set_lock(root, {"flashinfer-python": spec})
            out = TOOLS["prepare_flashinfer_native"](output=root / "native", cache=root / "cache", offline=True)
            names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(names, {"cccl/library/detail.inl", "cccl/tools/check.py", "vendor/include/flashinfer/math.cuh",
                                     "native-include/flashinfer/math.cuh", "vendor/csrc/sampling.cu", "vendor/csrc/kda/tool.py"})
            self.assertEqual((out / "native-include/flashinfer/math.cuh").read_bytes(), entries["include/flashinfer/math.cuh"])

    def test_source_extraction_preserves_original_bytes(self):
        with fixture() as root:
            archive = root / "cache/original.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                entry = tarfile.TarInfo("library-1.0/source.rs")
                raw = b"// original source\r\n"
                entry.size = len(raw)
                stream.addfile(entry, io.BytesIO(raw))
            set_lock(root, {"library": archive_spec(archive, "sdist")})
            out = TOOLS["prepare_source"]("library", output=root / "source", cache=root / "cache", offline=True)
            self.assertEqual((out / "source.rs").read_bytes(), raw)
            self.assertEqual(TOOLS["prepare_source"]("library", output=out, cache=root / "cache", offline=True), out)

    def test_archive_paths_and_links_cannot_escape_output(self):
        with fixture() as root:
            for name in ("../escape", "/absolute", "C:/drive", "a\\windows"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    TOOLS["member"](root, name)
            archive = root / "cache/linked.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                entry = tarfile.TarInfo("library/link")
                entry.type, entry.linkname = tarfile.SYMTYPE, "../../escape"
                stream.addfile(entry)
            set_lock(root, {"library": archive_spec(archive, "sdist")})
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                TOOLS["prepare_source"]("library", output=root / "source", cache=root / "cache", offline=True)
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
