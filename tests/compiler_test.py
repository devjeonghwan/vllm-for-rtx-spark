"""CPU checks for the isolated build compiler's actual process architecture."""
import ast
import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "src/kernels/cute/compiler"
SPEC = importlib.util.spec_from_file_location("compiler_process_arch", COMPILER / "process_arch.py")
process_arch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process_arch)


def pe(machine):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    return data


class CompilerProcessTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / ".build/compiler-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.executable = Path(self.temporary.name) / "python.exe"

    def tearDown(self):
        self.temporary.cleanup()

    def test_x64_process_on_arm64_host_is_admitted(self):
        self.executable.write_bytes(pe(0x8664))
        with patch.object(process_arch.sys, "executable", str(self.executable)), \
                patch.object(process_arch.sys, "platform", "win32"), \
                patch("platform.machine", return_value="ARM64"), \
                patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "ARM64"}):
            self.assertEqual(process_arch.require_windows_x64_process(), 0x8664)

    def test_host_environment_cannot_admit_arm64_process(self):
        self.executable.write_bytes(pe(0xAA64))
        with patch.object(process_arch.sys, "executable", str(self.executable)), \
                patch.object(process_arch.sys, "platform", "win32"), \
                patch("platform.machine", return_value="AMD64"), \
                patch.dict(os.environ, {"PROCESSOR_ARCHITECTURE": "AMD64"}):
            with self.assertRaisesRegex(RuntimeError, "0xaa64"):
                process_arch.require_windows_x64_process()

    def test_other_pe_architectures_are_rejected(self):
        for machine in (0x014C, 0xA641, 0):
            with self.subTest(machine=machine):
                self.executable.write_bytes(pe(machine))
                with patch.object(process_arch.sys, "executable", str(self.executable)), \
                        patch.object(process_arch.sys, "platform", "win32"):
                    with self.assertRaisesRegex(RuntimeError, "requires a Windows x64"):
                        process_arch.require_windows_x64_process()

    def test_malformed_executables_fail_closed(self):
        bad_offset = pe(0x8664)
        struct.pack_into("<I", bad_offset, 60, 10000)
        bad_signature = pe(0x8664)
        bad_signature[64:68] = b"nope"
        for data in (b"", b"\x7fELF" + bytes(128), bad_offset, bad_signature):
            with self.subTest(data=bytes(data[:4])):
                self.executable.write_bytes(data)
                with patch.object(process_arch.sys, "executable", str(self.executable)), \
                        patch.object(process_arch.sys, "platform", "win32"):
                    with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
                        process_arch.require_windows_x64_process()

    def test_non_windows_process_is_rejected(self):
        with patch.object(process_arch.sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "requires a Windows x64"):
                process_arch.require_windows_x64_process()

    def test_all_compiler_entrypoints_use_shared_process_guard(self):
        for name in ("launch.py", "b12x/compile_base.py", "b12x/compile_fast.py",
                     "hc/compile.py", "skinny/compile.py", "gdn/compile.py"):
            with self.subTest(name=name):
                tree = ast.parse((COMPILER / name).read_text(encoding="utf-8"))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                self.assertTrue(any(isinstance(node.func, ast.Subscript)
                                    and isinstance(node.func.slice, ast.Constant)
                                    and node.func.slice.value == "require_windows_x64_process"
                                    for node in calls))
                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        self.assertNotIn("platform.machine", ast.unparse(node.test))


if __name__ == "__main__":
    unittest.main()
