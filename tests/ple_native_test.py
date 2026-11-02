"""CPU byte and lifetime regression tests for the bundled Windows PLE reader."""

from concurrent.futures import ThreadPoolExecutor
import builtins
import ctypes
import errno
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PAGE_BYTES = 4096
GUARD_BYTES = 16


def native_windows_arm64():
    if os.name != "nt":
        return False
    try:
        with Path(sys.executable).open("rb") as source:
            source.seek(60)
            source.seek(struct.unpack("<I", source.read(4))[0])
            return source.read(6) == b"PE\0\0\x64\xaa"
    except (OSError, struct.error):
        return False


def payload(row, columns, salt):
    # The first eight bytes identify the row and file independently of the reader.
    identity = (row | (salt << 32)).to_bytes(8, "little")
    rest = bytes((row * 17 + column * 29 + salt) & 255 for column in range(columns))
    return (identity + rest)[:columns]


@unittest.skipUnless(native_windows_arm64(), "requires native Windows ARM64")
class PleNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ctypes.CDLL(str(ROOT / "assets/native/mbx_ple_nvme.dll"), winmode=0x100 | 0x1000)
        functions = {
            "mbx_t_fd": [ctypes.c_char_p],
            "mbx_t_open": [ctypes.c_int],
            "mbx_t_close": [ctypes.c_int],
            "mbx_t_ok2": [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int],
            "mbx_t_read": [ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
        }
        for name, arguments in functions.items():
            function = getattr(cls.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int

    def setUp(self):
        parent = ROOT / ".build/ple-native-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name)
        self.descriptors = set()

    def tearDown(self):
        try:
            for descriptor in tuple(self.descriptors):
                self.assertEqual(self.library.mbx_t_close(descriptor), 0)
                self.descriptors.remove(descriptor)
        finally:
            self.temporary.cleanup()

    def fixture(self, name="테이블.flat", salt=17, cp=80, cs=10):
        rows = 2053
        stride = max(cp + cs, 96)
        rows_per_page = PAGE_BYTES // stride
        storage = bytearray([0xA5]) * (((rows + rows_per_page - 1) // rows_per_page) * PAGE_BYTES)
        for row in range(rows):
            offset = row // rows_per_page * PAGE_BYTES + row % rows_per_page * stride
            storage[offset:offset + cp] = payload(row, cp, salt)
            storage[offset + cp:offset + cp + cs] = payload(row, cs, salt + 37)
        path = self.directory / name
        path.write_bytes(storage)
        self.assertEqual(self.library.mbx_t_ok2(str(path).encode("utf-8"), rows, cp, cs), 1)
        descriptor = self.library.mbx_t_fd(str(path).encode("utf-8"))
        self.assertGreater(descriptor, 0)
        self.descriptors.add(descriptor)
        return {"path": path, "fd": descriptor, "rows": rows, "cp": cp, "cs": cs,
                "salt": salt, "rows_per_page": rows_per_page, "file_bytes": len(storage)}

    def read(self, fixture, rows, qd=256, expected_code=0):
        rows = tuple(rows)
        self.assertEqual(self.library.mbx_t_open(qd), 0)  # Queue depth is thread local.
        indices = (ctypes.c_int64 * len(rows))(*rows)
        outputs = []
        for columns in (fixture["cp"], fixture["cs"]):
            output = (ctypes.c_ubyte * (len(rows) * columns + 2 * GUARD_BYTES))()
            ctypes.memset(output, 0xCD, len(output))
            outputs.append(output)
        code = self.library.mbx_t_read(
            fixture["fd"], indices, len(rows),
            ctypes.byref(outputs[0], GUARD_BYTES), ctypes.byref(outputs[1], GUARD_BYTES),
            fixture["cp"], fixture["cs"], 0,
        )
        self.assertEqual(code, expected_code)
        for index, (output, columns) in enumerate(zip(outputs, (fixture["cp"], fixture["cs"]))):
            self.assertEqual(bytes(output[:GUARD_BYTES]), b"\xcd" * GUARD_BYTES)
            self.assertEqual(bytes(output[-GUARD_BYTES:]), b"\xcd" * GUARD_BYTES)
            actual = ctypes.string_at(ctypes.byref(output, GUARD_BYTES), len(rows) * columns)
            if expected_code == 0:
                salt = fixture["salt"] + (37 if index else 0)
                expected = b"".join(payload(row, columns, salt) for row in rows)
                self.assertEqual(actual, expected)
            else:
                self.assertEqual(actual, b"\xcd" * (len(rows) * columns))

    def close(self, fixture):
        self.assertEqual(self.library.mbx_t_close(fixture["fd"]), 0)
        self.descriptors.remove(fixture["fd"])
        self.read(fixture, [0], expected_code=-errno.EBADF)
        self.assertEqual(self.library.mbx_t_close(fixture["fd"]), 0)
        # Both the primary and helper read handles must be released on Windows.
        fixture["path"].unlink()

    def test_default_paths_queue_transitions_and_exact_bytes(self):
        for cp, cs in ((80, 10), (120, 24)):
            with self.subTest(cp=cp, cs=cs):
                fixture = self.fixture(name=f"테이블-{cp}-{cs}.flat", cp=cp, cs=cs)
                cases = ((256, 0), (256, 1), (256, 511), (256, 512), (256, 513),
                         (256, 64), (256, 728), (256, 64), (768, 200), (256, 728),
                         (768, 768), (256, 513), (16, 769), (1, 65), (256, 1023))
                for qd, count in cases:
                    with self.subTest(qd=qd, count=count):
                        self.read(fixture, ((index * 17 + 31) % fixture["rows"] for index in range(count)), qd)
                self.read(fixture, [0, 41, 42, 2049, 2052, 0, 42])
                self.read(fixture, [42] * 513)
                self.read(fixture, reversed(range(769)))
                self.read(fixture, [0, 41, -1], expected_code=-errno.EINVAL)
                beyond = fixture["file_bytes"] // PAGE_BYTES * fixture["rows_per_page"]
                self.read(fixture, [beyond], expected_code=-errno.EINVAL)
                self.read(fixture, [0, 42, 2052])
                self.close(fixture)

    def test_same_file_and_two_file_concurrent_readers_close_after_join(self):
        first = self.fixture(name="first.flat", salt=17)
        second = self.fixture(name="second.flat", salt=29)

        def consume(fixture, offset):
            for count in (511, 512, 513, 728):
                self.read(fixture, ((index * 17 + offset) % fixture["rows"] for index in range(count)))

        for pair in ((first, first), (first, second)):
            with self.subTest(same_file=pair[0] is pair[1]):
                with ThreadPoolExecutor(max_workers=2) as workers:
                    futures = [workers.submit(consume, fixture, 31 + index * 71)
                               for index, fixture in enumerate(pair)]
                    for future in futures:
                        future.result(timeout=15)
        self.close(first)
        self.read(second, [2052, 42, 0])
        self.close(second)


class PleCompletionBoundaryTests(unittest.TestCase):
    """No Torch, native DLL, CUDA, or OS I/O execution is required here."""

    def setUp(self):
        source = ROOT / "src/runtime/ple/__init__.py"
        spec = importlib.util.spec_from_file_location("_ple_completion_fixture", source)
        self.boundary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.boundary)

    def completion_prefixes(self):
        """Execute the four post-event guards and legacy post-copy guard.

        Extract only the completion prefix from the public patch, so importing
        vLLM/Torch is unnecessary. The appended publish sentinel represents the
        output assignment that follows each guard in the patched method.
        """
        patch_text = (ROOT / "patches/vllm/0003-recipe-runtime.patch").read_text(encoding="utf-8")
        fragments = []
        for relative, event in (("vllm/v1/worker/gpu/async_utils.py", "copy_event"),
                                ("vllm/v1/worker/gpu_model_runner.py", "async_copy_ready_event")):
            block = patch_text.split(f"diff --git a/{relative} ", 1)[1].split("\ndiff --git ", 1)[0]
            postimage = [line[1:] for line in block.splitlines()
                         if line.startswith((" ", "+")) and not line.startswith("+++")]
            starts = [i for i, line in enumerate(postimage) if line.strip() == f"self.{event}.synchronize()"]
            self.assertEqual(len(starts), 2)  # Generation and pooling output.
            fragments.extend(postimage[index:index + 5] for index in starts)
            if event == "async_copy_ready_event":
                guards = [i for i, line in enumerate(postimage) if line.strip() == 'if os.name == "nt":']
                self.assertEqual(len(guards), 3)
                index = guards[-1]
                # Its source hunk is after the existing blocking sampled-token
                # transfer/parsing and still inside the synchronous branch.
                self.assertEqual(postimage[index - 1].strip(), ")")
                self.assertEqual(postimage[index + 4].strip(), "else:")
                fragments.append(["        self.blocking_transfer()"] + [line[4:] for line in postimage[index:index + 4]])
        self.assertEqual(len(fragments), 5)
        functions = []
        for fragment in fragments:
            self.assertEqual(fragment[1].strip(), 'if os.name == "nt":')
            self.assertEqual(fragment[2].strip(), 'from vllm._rtx_spark.runtime.ple import raise_if_async_failed')
            self.assertEqual(fragment[4].strip(), 'raise_if_async_failed()')
            function_source = "def complete(self):\n" + textwrap.indent(textwrap.dedent("\n".join(fragment)), "    ") + "\n    self.publish()\n"
            imports = dict(vars(builtins))
            original_import = imports["__import__"]
            def import_boundary(name, *args, **kwargs):
                if name == "vllm._rtx_spark.runtime.ple":
                    return self.boundary
                return original_import(name, *args, **kwargs)
            imports["__import__"] = import_boundary
            namespace = {"__builtins__": imports, "os": SimpleNamespace(name="nt")}
            exec(compile(function_source, "<actual-public-PLE-completion-prefix>", "exec"), namespace)
            functions.append(namespace["complete"])
        return functions

    def fixture(self, fail_at_completion=False, status=1):
        events = []
        owner = SimpleNamespace(failed=0)
        def failed():
            events.append("failed_query")
            return owner.failed
        owner.mbx_pa_failed = failed
        self.boundary._LIBRARIES["MBX_PLE_ASYNC_LIB"] = ("already-loaded-fixture.dll", owner)
        def synchronize():
            events.append("existing_copy_event_complete")
            if fail_at_completion:
                owner.failed = status
                events.append("native_failure_is_now_completed")
        output = SimpleNamespace(copy_event=SimpleNamespace(synchronize=synchronize),
                                 async_copy_ready_event=SimpleNamespace(synchronize=synchronize),
                                 blocking_transfer=synchronize,
                                 publish=lambda: events.append("publish"))
        return owner, output, events

    def test_delayed_read_failure_is_rejected_after_existing_event_before_publish(self):
        for complete in self.completion_prefixes():
            for status in (1, 2):
                with self.subTest(status=status):
                    owner, output, events = self.fixture(fail_at_completion=True, status=status)
                    self.assertEqual(owner.mbx_pa_failed(), 0)  # The old early guard missed it.
                    with patch.object(ctypes, "CDLL", side_effect=AssertionError("must not load a DLL")):
                        with self.assertRaisesRegex(RuntimeError, "stop and restart this worker"):
                            complete(output)
                    self.assertNotIn("publish", events)
                    self.assertEqual(events.count("existing_copy_event_complete"), 1)
                    self.assertGreater(events.index("native_failure_is_now_completed"), events.index("existing_copy_event_complete"))
                    self.assertEqual(events[-1], "failed_query")
                    self.assertEqual(owner.failed, status)  # Fault is never acknowledged/reset.

    def test_normal_completion_queries_only_the_already_loaded_owner(self):
        for complete in self.completion_prefixes():
            _, output, events = self.fixture()
            with patch.object(ctypes, "CDLL", side_effect=AssertionError("must not load a DLL")):
                complete(output)
            self.assertEqual(events, ["existing_copy_event_complete", "failed_query", "publish"])

    def test_uninitialized_boundary_is_noop_and_never_loads_a_library(self):
        self.boundary._LIBRARIES.clear()
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("must not load a DLL")):
            self.boundary.raise_if_async_failed()
            for complete in self.completion_prefixes():
                events = []
                output = SimpleNamespace(copy_event=SimpleNamespace(synchronize=lambda: events.append("complete")),
                                         async_copy_ready_event=SimpleNamespace(synchronize=lambda: events.append("complete")),
                                         blocking_transfer=lambda: events.append("complete"),
                                         publish=lambda: events.append("publish"))
                complete(output)
                self.assertEqual(events, ["complete", "publish"])

    def test_sticky_failure_blocks_later_outputs_without_dropping_native_owner(self):
        owner, _, _ = self.fixture(fail_at_completion=False)
        owner.failed = 1
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "native status 1"):
                self.boundary.raise_if_async_failed()
        self.assertIs(self.boundary._LIBRARIES["MBX_PLE_ASYNC_LIB"][1], owner)


if __name__ == "__main__":
    unittest.main()
