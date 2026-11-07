"""CMake patch timing and Windows ARM64 build selection, without compilation."""
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from support import ROOT, source_path


BUILD = runpy.run_path(ROOT / "scripts/build_vllm.py")


class BuildFlowTests(unittest.TestCase):
    def temporary(self):
        parent = ROOT / ".build/build-flow-tests"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def command(self, *args, check=True):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True)
        if check and result.returncode:
            self.fail(result.stdout + result.stderr)
        return result

    def test_fetchcontent_patches_before_configure_and_rejects_conflicts(self):
        if not shutil.which("cmake"):
            self.skipTest("CMake is required for the dependency lifecycle contract")
        helper = source_path("cmake/patches/apply_windows_kernel_patch.cmake")
        root = self.temporary()
        upstream = root / "upstream source"
        upstream.mkdir()
        (upstream / "value.txt").write_bytes(b"original\n")
        (upstream / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.26)
project(DependencyFixture LANGUAGES NONE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/value.txt" value)
if(NOT value STREQUAL "patched\\n")
  message(FATAL_ERROR "Dependency was configured before its patch")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/configured.txt" "${value}")
''', encoding="utf-8", newline="\n")
        self.command("git", "init", "--quiet", upstream)
        self.command("git", "-C", upstream, "config", "--local", "core.autocrlf", "false")
        self.command("git", "-C", upstream, "config", "--local", "core.eol", "lf")
        self.command("git", "-C", upstream, "add", ".")
        self.command("git", "-C", upstream, "-c", "user.name=CPU fixture",
                     "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                     "commit", "--quiet", "-m", "Fixture")
        changes = root / "changes.patch"
        changes.write_bytes(b"--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-original\n+patched\n")
        project = root / "project"
        project.mkdir()
        (project / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.26)
project(BuildFixture LANGUAGES NONE)
include(FetchContent)
FetchContent_Declare(example
  SOURCE_DIR "{upstream.as_posix()}"
  PATCH_COMMAND "${{CMAKE_COMMAND}}" "-DSOURCE_DIR=<SOURCE_DIR>"
    "-DPATCH_FILE={changes.as_posix()}" -P "{helper.as_posix()}")
FetchContent_MakeAvailable(example)
''', encoding="utf-8", newline="\n")
        build = root / "build"
        self.command("cmake", "-S", project, "-B", build, "-G", "Ninja")
        self.assertEqual((build / "_deps/example-build/configured.txt").read_text(), "patched\n")
        source = upstream
        self.command("cmake", "-DSOURCE_DIR=" + str(source), "-DPATCH_FILE=" + str(changes), "-P", helper)
        self.command("cmake", "-S", project, "-B", build, "-G", "Ninja")
        changes.write_bytes(b"--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-missing\n+replacement\n")
        failed = self.command("cmake", "-DSOURCE_DIR=" + str(source), "-DPATCH_FILE=" + str(changes), "-P", helper, check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((source / "value.txt").read_bytes(), b"patched\n")

    def test_only_selected_arm64_extensions_are_compiled(self):
        root = self.temporary()
        calls = []
        fake_torch = SimpleNamespace(__version__=BUILD["MANIFEST"]["platform"]["torch"])
        with patch.dict(BUILD["build_native"].__globals__, run=lambda args, **kwargs: calls.append((args, kwargs))), \
                patch("platform.machine", return_value="ARM64"), \
                patch("shutil.which", return_value="tool"), \
                patch.object(sys, "platform", "win32"), \
                patch.object(sys, "version_info", (3, 13, 0)), \
                patch.dict(sys.modules, torch=fake_torch):
            BUILD["build_native"](root / "source", root / "build", 2)
        configure, settings = calls[0]
        self.assertIn("-DVLLM_BUILD_EXTERNAL_KERNELS=OFF", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=121a-real", configure)
        self.assertFalse(any("\\" in value for value in configure
                             if str(value).startswith(("-DVLLM_PYTHON_EXECUTABLE=", "-DCMAKE_INSTALL_PREFIX="))))
        self.assertEqual(settings["env"]["TORCH_CUDA_ARCH_LIST"], "12.1a")
        compiled = [args[-1] for args, _ in calls if args[1] == "--build"]
        self.assertEqual(compiled, ["_C_stable_libtorch", "_moe_C_stable_libtorch"])
        installed = [args[-1] for args, _ in calls if args[1] == "--install"]
        self.assertEqual(installed, compiled + BUILD["MANIFEST"]["build"]["source_components"])


if __name__ == "__main__":
    unittest.main()
