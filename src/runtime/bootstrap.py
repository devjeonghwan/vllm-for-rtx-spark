"""Early, CPU-only setup shared by the API process and spawned workers."""

from __future__ import annotations

import os
import sys

_initialized = False


def publish_dependency_paths():
    """Expose bundled third-party bindings without publishing recipe helpers."""
    from . import vendor_root

    for directory in (vendor_root(),):
        if not directory.is_dir():
            raise RuntimeError("RTX Spark runtime dependency is missing: " + str(directory))
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)


def initialize():
    """Initialize packaged dependencies before vLLM loads models.

    This deliberately imports neither Torch nor vLLM. The patched vLLM calls
    it before importing platform/configuration modules, including in children.
    """
    global _initialized
    if sys.platform != "win32":
        return
    publish_dependency_paths()
    from ..kernels.providers import initialize_providers

    initialize_providers()
    if _initialized:
        return
    from . import cache_root

    cache_root().mkdir(parents=True, exist_ok=True)
    defaults = {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "USE_LIBUV": "0",
        "VLLM_USE_RUST_FRONTEND": "0",
        "TORCH_CUDA_ARCH_LIST": "12.1a",
        "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    paths = {
        "FLASHINFER_WORKSPACE_BASE": cache_root() / "flashinfer",
    }
    for name, value in paths.items():
        os.environ.setdefault(name, str(value))
    from ..kernels.cute import install

    install()
    _initialized = True
