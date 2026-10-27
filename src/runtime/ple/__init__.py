"""OS boundary for the original recipe PLE Python forward and C ABI.

No embedding/dequantization/hash/checkpoint/Graph policy is implemented here.
The two original C ABIs must first be built as native Windows ARM64 DLLs.
This source candidate intentionally fails if that native bridge is absent.
"""

import ctypes
import os
import platform
import struct
from pathlib import Path

_LIBRARIES = {}
_ATTACHED_LAYERS = []
_FAILED_OWNERS = []


def library(variable):
    """Load only an explicitly selected ARM64 PE DLL, never a Linux host ELF."""
    if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("This bridge is only the native Windows ARM64 OS boundary")
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("Original PLE C ABI requires64-bit pointers")
    configured = os.environ.get(variable)
    if not configured:
        raise RuntimeError(variable + " must select the native original-C-ABI Windows DLL")
    path = Path(configured)
    if not path.is_absolute() or path.suffix.lower() != ".dll":
        raise ValueError("Use an absolute native DLL path for " + variable)
    with path.open("rb") as source:
        header = source.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("Original Linux HOST ELF is not a Windows module")
        source.seek(struct.unpack_from("<I", header, 60)[0])
        pe = source.read(6)
    if pe[:4] != b"PE\0\0" or len(pe) != 6 or struct.unpack_from("<H", pe, 4)[0] != 0xAA64:
        raise ValueError("The PLE bridge must be an ARM64 PE DLL")
    resolved = str(path.resolve())
    previous = _LIBRARIES.get(variable)
    if previous is not None:
        if previous[0] != resolved:
            raise RuntimeError("An original PLE library cannot be rebound while owned")
        return previous[1]
    result = ctypes.CDLL(resolved, winmode=0x100 | 0x1000)
    if variable == "MBX_PLE_ASYNC_LIB":
        fixture = result.mbx_pa_cpu_fixture_mode
        fixture.argtypes, fixture.restype = [], ctypes.c_int
        if fixture() != 0:
            raise RuntimeError("The CPU-only flag fixture DLL is forbidden for actual inference")
    _LIBRARIES[variable] = (resolved, result)
    return result


def attach_nvme_table(layer, tab, table_class, hooks, keep):
    """Reuse original setup/post/wait hooks without publishing a CPU VA as CUDA.

    The original Linux implementation creates a CUDA DLPack span over mapped
    CPU file virtual addresses. That assumption is not made for Windows.
    NVMe+async setup consumes only the original table geometry/file offsets.
    """
    if (
        os.environ.get("MBX_PLE_MMAP_MODE", "mmap").lower() != "nvme"
        or os.environ.get("MBX_PLE_ASYNC", "0") != "1"
    ):
        raise RuntimeError("Windows original PLE requires native NVMe+async C ABI; direct mapped-file CUDA VA is unavailable")
    table = table_class()
    table.tab = tab
    table.keep, table.maps = [], []
    for field in ("S", "ns", "cp", "cs", "rows"):
        setattr(table, field, tab[field])
    table.windows_metadata_only = True
    layer._mbx_tab = table
    keep.append(table)
    _ATTACHED_LAYERS.append(layer)
    # These are the original recipe hook objects, not replacement Python math.
    for hook in hooks:
        hook(layer)
    asynchronous = getattr(layer, "_mbx_pa", None)
    if asynchronous is None or getattr(asynchronous, "lib", None) is None:
        raise RuntimeError("Original native async initialization failed; Windows cannot fall back to Linux UVA")


def shutdown_after_graphs_drained(*, graphs_reset, streams_drained):
    """Caller must supply actual original-manager reset/drain evidence first.

    No flag is cleared or completion fabricated here. Failed native resources,
    original layer buffers and DLLs remain strongly owned for explicit retry.
    """
    if graphs_reset is not True or streams_drained is not True:
        raise RuntimeError("Reset original Graphs and drain all actual streams before closing PLE")
    asynchronous = _LIBRARIES.get("MBX_PLE_ASYNC_LIB")
    reader = _LIBRARIES.get("MBX_PLE_LIB")
    try:
        if asynchronous is not None:
            shutdown = asynchronous[1].mbx_pa_shutdown
            shutdown.argtypes, shutdown.restype = [], ctypes.c_int
            result = shutdown()
            if result:
                raise RuntimeError(f"Original async worker remains owned: shutdown result {result}")
        if reader is not None:
            close = reader[1].mbx_t_close
            close.argtypes, close.restype = [ctypes.c_int], ctypes.c_int
            for descriptor in {getattr(layer, "_mbx_t_fd", None) for layer in _ATTACHED_LAYERS} - {None}:
                result = close(descriptor)
                if result:
                    raise RuntimeError(f"Original table I/O remains owned: close result {result}")
        _ATTACHED_LAYERS.clear()
        _FAILED_OWNERS.clear()
        return {"native_worker_stopped": True, "native_files_closed": True}
    except BaseException:
        _FAILED_OWNERS.append((dict(_LIBRARIES), tuple(_ATTACHED_LAYERS)))
        raise
