"""Build-only compiler process checks, independent of the Windows host CPU."""
from pathlib import Path
import struct
import sys

AMD64 = 0x8664


def executable_machine(path):
    """Read the actual executable's PE Machine field without loading it."""
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("Compiler executable is not a Windows PE image")
        offset = struct.unpack_from("<I", header, 60)[0]
        if offset < 64 or offset + 6 > path.stat().st_size:
            raise ValueError("Compiler executable has an invalid PE header offset")
        handle.seek(offset)
        signature = handle.read(6)
        if signature[:4] != b"PE\0\0":
            raise ValueError("Compiler executable has an invalid PE signature")
        return struct.unpack_from("<H", signature, 4)[0]


def require_windows_x64_process():
    """Require the authorized x64 build process, including on ARM64 Windows.

    platform.machine() and PROCESSOR_ARCHITECTURE can identify the host rather
    than the Python process under emulation; neither is an admission check.
    This helper is never used to admit an inference process.
    """
    if sys.platform != "win32":
        raise RuntimeError("This compiler provider requires a Windows x64 build process")
    try:
        machine = executable_machine(sys.executable)
    except (OSError, ValueError) as issue:
        raise RuntimeError("Cannot verify the Windows x64 compiler process") from issue
    if machine != AMD64:
        raise RuntimeError("This compiler provider requires a Windows x64 build process; "
                           f"Python PE Machine is 0x{machine:04x}")
    return machine
