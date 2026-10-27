"""Early, CPU-only setup shared by the API process and spawned workers."""

from __future__ import annotations

import os
import sys

_initialized = False


def initialize():
    """Make packaged original helpers importable before vLLM loads models.

    This deliberately imports neither Torch nor vLLM. The patched vLLM calls
    it before importing platform/configuration modules, including in children.
    """
    global _initialized
    if _initialized:
        return
    if sys.platform != "win32":
        return
    from . import cache_root, recipe_helpers, resources_root, vendor_root

    root = resources_root()
    cache_root().mkdir(parents=True, exist_ok=True)
    for directory in (vendor_root(), recipe_helpers()):
        if not directory.is_dir():
            raise RuntimeError("RTX Spark runtime dependency is missing: " + str(directory))
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    defaults = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "USE_LIBUV": "0",
        "VLLM_USE_RUST_FRONTEND": "0",
        "MBX_LOADER_PREPARE": "0",
        "MBX_PLE_MMAP": "1",
        "MBX_PLE_MMAP_MODE": "nvme",
        "MBX_PLE_ASYNC": "1",
        "MBX_PLE_ASYNC_MAX": "32768",
        "MBX_PLE_NVME_THREADS": "4",
        "MBX_PLE_QD": "16",
        "MBX_PLE_ASYNC_PREFETCH": "1",
        "MBX_PLE_MMAP_PREWARM": "0",
        "MBX_LOADER_EMPTY_CACHE": "1",
        "MBX_PLE_PREFILL_CKPT": "1",
        "MBX_DEPTH_SYNCFREE": "1",
        "VLLM_GDN_DECODE_KERNEL": "cuda",
        "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        "MBX_CSF_VISUAL_BF16": "0",
        "MBX_OPT_QUANT_HC_DOWN": "0",
        "TORCH_CUDA_ARCH_LIST": "12.1a",
        "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    paths = {
        "MBX_PLE_LIB": root / "native" / "mbx_ple_nvme.dll",
        "MBX_PLE_ASYNC_LIB": root / "native" / "mbx_ple_async.dll",
        "MBX_DEPTH_LIB": root / "native" / "mbx_mtp.dll",
        "MBX_DEPTH_CFG": root / "recipes" / "qwen3_8" / "mtp-depth.json",
        "MBX_DEPTH_STATS": cache_root() / "depth-stats.json",
        "MBX_DEPTH_STEPTIME": cache_root() / "depth-steptime.json",
        "MBX_PLE_NVME_DIR": cache_root() / "ple-nvme",
        "FLASHINFER_WORKSPACE_BASE": cache_root() / "flashinfer",
    }
    for name, value in paths.items():
        os.environ.setdefault(name, str(value))
    from ..kernels.cute import install

    install()
    _initialized = True
