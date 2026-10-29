"""Lazy native ARM64 HC upmix ABI; no CUDA work or DLL load at import."""

import ctypes
from pathlib import Path
import platform
import struct
import sys
import threading

_library = None
_launch = None
_devices = {}
_lock = threading.Lock()
_selection = None


def asset_path():
    from ...runtime import artifact

    path, record = artifact("hc_upmix")
    if record.get("abi") != 2:
        raise RuntimeError("Unsupported RTX Spark HC native ABI")
    return Path(path).resolve()


def available_device(tensor):
    """Called only during real custom-op execution, never by fake/schema registration."""
    if sys.platform != "win32" or platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    import torch

    device = tensor.device.index
    if device not in _devices:
        _devices[device] = torch.cuda.get_device_capability(tensor.device) == (12, 1)
    return _devices[device]


def aligned(*tensors):
    return all(tensor.data_ptr() % 16 == 0 for tensor in tensors)


def _load():
    global _library, _launch
    if _library is not None:
        return
    with _lock:
        if _library is not None:
            return
        path = asset_path()
        raw = path.read_bytes()
        if len(raw) < 64 or raw[:2] != b"MZ":
            raise RuntimeError("Expected RTX Spark HC ARM64 PE")
        offset = struct.unpack_from("<I", raw, 60)[0]
        if raw[offset:offset + 6] != b"PE\0\0\x64\xaa":
            raise RuntimeError("Expected RTX Spark HC ARM64 PE")
        library = ctypes.CDLL(str(path))
        abi = library.rtx_hc_upmix_abi
        abi.argtypes, abi.restype = [], ctypes.c_int
        if abi() != 2:
            raise RuntimeError("RTX Spark HC DLL ABI mismatch")
        launch = library.rtx_hc_upmix_bf16
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        # Both stay reachable for process lifetime, including every captured Graph.
        _launch, _library = launch, library


def invoke(xn, lora, weight):
    global _selection
    import torch

    _load()
    output = xn.new_empty((xn.shape[0], 2560))
    stream = torch.cuda.current_stream(xn.device)
    error = _launch(lora.data_ptr(), weight.data_ptr(), xn.data_ptr(), output.data_ptr(),
                    xn.shape[0], lora.stride(0), stream.cuda_stream)
    if error:
        raise RuntimeError(f"RTX Spark HC CUDA launch error {error}")
    if _selection is None:
        _selection = {"rows": xn.shape[0], "lora_stride": lora.stride(0),
                      "activation_dtype": str(xn.dtype), "weight_dtype": str(weight.dtype)}
        from vllm.logger import init_logger

        init_logger("vllm._rtx_spark.kernels.hc_upmix").info(
            "RTX Spark HC upmix selected: %s", _selection)
    return output


def status():
    return {"loaded": _library is not None, "checked_devices": len(_devices), "abi": 2,
            "first_selection": _selection}
