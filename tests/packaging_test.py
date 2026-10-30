"""Release packaging includes one checked wheel and installation files only."""

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = runpy.run_path(ROOT / "scripts/package_release.py")


class InstallationPackagingTests(unittest.TestCase):
    @contextmanager
    def prepared_release(self):
        parent = ROOT / ".build/installation-packaging-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            output = root / "dist"
            output.mkdir()
            files = {
                "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "sources.lock.json",
                "configs/model/gpu.yaml", "requirements/constraints.txt", "scripts/install.ps1",
                "patches/vllm/manifest.json",
                # Developer sources and private results are not installation inputs.
                "src/version.py", "third_party/vendor/source.cpp", "tests/private_test.py",
                "benchmarks/results.json", "research/private.txt",
            }
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            wheel = output / (PACKAGER["CORE_WHEEL_PREFIX"] + "cp313-cp313-win_arm64.whl")
            wheel.write_bytes(b"wheel fixture; checked by a mocked validator")
            version = dict(PACKAGER["_VERSION"], manifest_path=lambda: root / "patches/vllm/manifest.json")
            namespace = PACKAGER["main"].__globals__
            captures = []
            with patch.dict(namespace, ROOT=root, _VERSION=version,
                            archive=lambda path, entries: captures.append((path, entries))), \
                    patch("subprocess.run") as checker, \
                    patch("zipfile.ZipFile", side_effect=AssertionError("CPU contracts must not create ZIPs")), \
                    patch("sys.argv", ["package_release.py", "--output", str(output)]):
                yield root, output, wheel, checker, captures

    def test_packages_only_installation_zip_and_reports_existing_wheel(self):
        with self.prepared_release() as (root, output, wheel, checker, captures):
            build_info = output / "build-info.json"
            build_info.write_text("{}", encoding="utf-8")
            old_source = output / f"vllm-for-rtx-spark-{PACKAGER['VERSION']}-source.zip"
            old_source.write_bytes(b"preserve old release evidence")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(PACKAGER["main"](), 0)
            checker.assert_called_once_with(
                [PACKAGER["sys"].executable, str(root / "scripts/check_wheels.py"), str(wheel)],
                cwd=root, check=True,
            )
            self.assertEqual(len(captures), 1)
            destination, entries = captures[0]
            self.assertEqual(destination.name, f"vllm-for-rtx-spark-{PACKAGER['VERSION']}-win_arm64.zip")
            names = [name for _, name in entries]
            self.assertEqual(set(names), {
                "wheels/" + wheel.name, "manifest.json", "README.md", "LICENSE",
                "THIRD_PARTY_NOTICES.md", "sources.lock.json", "build-info.json",
                "configs/model/gpu.yaml", "requirements/constraints.txt", "scripts/install.ps1",
            })
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(path.is_file() for path, _ in entries))
            self.assertEqual(json.loads(stdout.getvalue())["artifacts"], [destination.name, wheel.name])
            self.assertEqual(old_source.read_bytes(), b"preserve old release evidence")
            self.assertEqual(wheel.read_bytes(), b"wheel fixture; checked by a mocked validator")

    def test_requires_exactly_one_matching_wheel_before_packaging(self):
        with self.prepared_release() as (_, output, wheel, checker, captures):
            (output / "unrelated.whl").write_bytes(b"extra fixture")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                PACKAGER["main"]()
            checker.assert_not_called()
            self.assertEqual(captures, [])
            self.assertTrue(wheel.is_file())

    def test_preserves_existing_installation_archive(self):
        with self.prepared_release() as (_, output, _, checker, captures):
            existing = output / f"vllm-for-rtx-spark-{PACKAGER['VERSION']}-win_arm64.zip"
            existing.write_bytes(b"existing installation fixture")
            with self.assertRaises(FileExistsError):
                PACKAGER["main"]()
            checker.assert_not_called()
            self.assertEqual(captures, [])
            self.assertEqual(existing.read_bytes(), b"existing installation fixture")


if __name__ == "__main__":
    unittest.main()
