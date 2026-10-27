"""Canonical native ARM64 loader for ORIGINAL CuTe-exported host COFF DLLs.

Only preloading does admission/initialization. invoke passes original header
argument addresses to the untouched host function, including its stream value.
No model tensor, extra GPU allocation, Graph executor, or TMA encoding is owned.
"""
import ctypes as C
import os
import platform
import struct
import sys
import threading
from pathlib import Path

_modules = {}
_failed_modules = []
_lock = threading.RLock()


def _pe_machine(path):
    with Path(path).open("rb") as f:
        f.seek(0x3C)
        offset = struct.unpack("<I", f.read(4))[0]
        f.seek(offset)
        if f.read(4) != b"PE\0\0":
            raise ValueError("Not a Windows PE: " + str(path))
        return struct.unpack("<H", f.read(2))[0]


def _runtime_directory(torch):
    """Prefer the matching native runtime shipped with the installed Torch."""
    candidates = [Path(torch.__file__).resolve().parent / "lib"]
    for name in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]) / "bin" / "arm64")
    candidates.append(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/arm64"))
    for directory in candidates:
        library = directory / "cudart64_13.dll"
        if library.is_file() and _pe_machine(library) == 0xAA64:
            return directory
    raise RuntimeError("A native ARM64 CUDA 13 runtime is missing from Torch and CUDA_PATH")


def runtime_version_supported(version):
    """The CUDA 13.4 build ABI accepts a compatible CUDA 13.x runtime."""
    return isinstance(version, int) and 13040 <= version < 14000


class CoffModule:
    """One original cudaLibrary handle/prefix/device, plus cached host entries."""
    def __init__(self, dll, *, init_symbol, load_symbol, stream):
        self.path = Path(dll).resolve()
        self.init_symbol, self.load_symbol = init_symbol, load_symbol
        self.device = stream.device.index
        self.handle = C.c_void_p()
        self.closed = False
        self.failure = None
        self.entries = {}
        self.dll = None
        self.native_DLL_directory = None
        self.cuda_runtime = None
        if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
            raise RuntimeError("Final host execution requires native Windows ARM64")
        if _pe_machine(sys.executable) != 0xAA64 or _pe_machine(self.path) != 0xAA64:
            raise RuntimeError("Python and original-host DLL must both be ARM64 PE")
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            raise RuntimeError("Borrow an already initialized original Torch CUDA context")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Original AOT libraries must be preloaded before CUDA Graph capture")
        native_bin = _runtime_directory(torch)
        self.native_DLL_directory = os.add_dll_directory(str(native_bin))
        self.cuda_runtime = C.WinDLL(str(native_bin / "cudart64_13.dll"), winmode=0x1100)
        self.cuda_runtime.cudaRuntimeGetVersion.argtypes = [C.POINTER(C.c_int)]
        self.cuda_runtime.cudaRuntimeGetVersion.restype = C.c_int
        runtime_version = C.c_int()
        if (self.cuda_runtime.cudaRuntimeGetVersion(C.byref(runtime_version)) != 0
                or not runtime_version_supported(runtime_version.value)):
            raise RuntimeError("This kernel pack requires a CUDA 13.x runtime version 13.4 or newer")
        self.dll = C.WinDLL(str(self.path), winmode=0x1100)
        self.dll.original_cute_runtime_abi.argtypes = [C.POINTER(C.c_uint64), C.c_size_t]
        self.dll.original_cute_runtime_abi.restype = C.c_int
        self.dll.original_cute_runtime_get_device.argtypes = [C.POINTER(C.c_int)]
        self.dll.original_cute_runtime_get_device.restype = C.c_int
        self.dll.original_cute_runtime_unload.argtypes = [C.c_void_p]
        self.dll.original_cute_runtime_unload.restype = C.c_int
        self.dll.original_cute_runtime_error_string.argtypes = [C.c_int]
        self.dll.original_cute_runtime_error_string.restype = C.c_char_p
        abi = (C.c_uint64 * 6)()
        if self.dll.original_cute_runtime_abi(abi, 6) != 0 or list(abi) != [1, 8, 56, 72, 13040, 0xAA64]:
            raise RuntimeError("Native CUDA runtime/shim ABI differs from the built source")
        actual_device = C.c_int()
        self._check(self.dll.original_cute_runtime_get_device(C.byref(actual_device)), "original current-device query")
        if actual_device.value != self.device:
            raise RuntimeError("Original current device differs from caller stream")
        self.init = getattr(self.dll, init_symbol)
        self.init.argtypes, self.init.restype = [C.POINTER(C.c_void_p)], C.c_int
        self.load = getattr(self.dll, load_symbol)
        self.load.argtypes, self.load.restype = [C.POINTER(C.c_void_p)], C.c_int
        try:
            # These two functions are the ORIGINAL emitted COFF functions.
            self._check(self.init(C.byref(self.handle)), init_symbol)
            self._check(self.load(C.byref(self.handle)), load_symbol)
        except BaseException as error:
            self.failure = error
            _failed_modules.append(self)
            raise

    def _check(self, value, label):
        code = int(value)
        if code:
            text = self.dll.original_cute_runtime_error_string(code)
            raise RuntimeError(f"{label} returned{code}: {text!r}")

    def bind(self, symbol, argument_count):
        """Bind the generated header's void(void**,int32) entry before capture."""
        if type(argument_count) is not int or argument_count < 0:
            raise ValueError("Original input argument count must be nonnegative")
        existing = self.entries.get(symbol)
        if existing is not None:
            if existing[1] != argument_count:
                raise ValueError("Cached original host signature changed")
            return self
        torch = sys.modules.get("torch")
        if torch is None or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind original host functions during prewarm, never capture")
        function = getattr(self.dll, symbol)
        function.argtypes, function.restype = [C.POINTER(C.c_void_p), C.c_int32], None
        self.entries[symbol] = (function, argument_count)
        return self

    def invoke(self, symbol, raw_argument_addresses):
        """Hot path: original COFF does all original TMA/PDL/launch work.

        Caller retains its original ctypes arguments through this synchronous
        host call. CUDA copies kernel argument values during launch; this code
        retains no tensor/CPU argument/descriptor after return. The raw stream
        inside the original argument array may differ from the preload stream.
        """
        if self.closed or self.failure is not None:
            raise RuntimeError("Original native host module is closed or failed")
        function, count = self.entries[symbol]
        if len(raw_argument_addresses) != count:
            raise ValueError("Argument count differs from original generated header")
        result = C.c_int32()
        arguments = (C.c_void_p * (count + 1))(*raw_argument_addresses, C.addressof(result))
        function(arguments, count + 1)
        self._check(result.value, symbol)
        return int(result.value)

    def close(self, *, graphs_reset, streams_drained):
        if self.closed:
            return
        if graphs_reset is not True or streams_drained is not True:
            raise RuntimeError("Keep original code libraries until real Graph reset and all consumer streams drain")
        try:
            if self.handle.value:
                self._check(self.dll.original_cute_runtime_unload(self.handle), "original CUDA library unload")
        except BaseException as error:
            self.failure = error
            if not any(item is self for item in _failed_modules):
                _failed_modules.append(self)
            raise
        self.closed = True
        self.handle.value = None
        self.entries.clear()
        # Do not manually FreeLibrary: cached bound functions/other prefixes may
        # share the same PE module. The OS and ctypes retain native code safely.


def load_module(dll, *, init_symbol, load_symbol, stream, invoke_symbols=None):
    """Prewarm-only cache lookup/first load, one canonical module instance."""
    key = (str(Path(dll).resolve()).casefold(), init_symbol, stream.device.index)
    with _lock:
        module = _modules.get(key)
        if module is None:
            module = CoffModule(dll, init_symbol=init_symbol, load_symbol=load_symbol, stream=stream)
            _modules[key] = module
        elif module.load_symbol != load_symbol or module.closed or module.failure is not None:
            raise RuntimeError("Cached original library identity changed or previous load failed")
        for symbol, count in (invoke_symbols or {}).items():
            module.bind(symbol, count)
        return module


def close_after_graphs_drained(*, graphs_reset, streams_drained):
    """Called once by the ORIGINAL runner cleanup after its final device sync."""
    if graphs_reset is not True or streams_drained is not True:
        raise RuntimeError("Original Graph/stream cleanup proof is required")
    with _lock:
        owners = list({id(m): m for m in [*_modules.values(), *_failed_modules]}.values())
        closed, errors = 0, []
        for module in owners:
            try:
                module.close(graphs_reset=True, streams_drained=True)
                closed += 1
            except BaseException as error:
                errors.append(repr(error))
        if errors:
            raise RuntimeError("Original CUDA library cleanup failed; owners retained: " + repr(errors))
        _modules.clear()
        _failed_modules.clear()
        return {"original_CUDA_libraries_closed": closed, "graphs_reset": True,
                "streams_drained": True, "extra_GPU_buffers_owned": 0}
