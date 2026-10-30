"""CPU byte and lifetime regression tests for the bundled Windows PLE reader."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
