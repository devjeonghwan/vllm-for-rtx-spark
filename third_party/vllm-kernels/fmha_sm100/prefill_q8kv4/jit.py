# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Op-local JIT for the SM100 Q8KV4 sparse prefill attention kernel."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import sysconfig
import time
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import jinja2
from torch.utils import cpp_extension

from .. import _jit_cache
from ..decode_q8kv4._build_utils import cuda_home as _cuda_home
from ..decode_q8kv4._build_utils import cutlass_root as _cutlass_root
from ..decode_q8kv4._build_utils import cutlass_version as _cutlass_version
from ..decode_q8kv4._build_utils import require_cuda_version
from ..decode_q8kv4._build_utils import supports_qmul4
from ..decode_q8kv4._build_utils import target_arch as _resolve_target_arch

logger = logging.getLogger(__name__)

# Offline builds (build.py) select the target architecture with this variable.
FMHA_SM100_PREFILL_Q8KV4_ARCH = "FMHA_SM100_PREFILL_Q8KV4_ARCH"
# Set to 1 to compile the packed-FP16 dequant path even when the toolchain accepts QMUL4: the path
# SM107 takes, so the two can be compared on one GPU.
DISABLE_QMUL4_ENV = "FMHA_SM100_PREFILL_Q8KV4_DISABLE_QMUL4"
_SUPPORTED_ARCHES = ("100a", "103a", "107a")
_QMUL4_DEQUANT = "qmul4"
_FP16_DEQUANT = "fp16"

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_API = _CSRC / "api"
_INCLUDE = _CSRC / "include"
_TEMPLATES = _CSRC / "templates"
# Largest block-scale staging shift (an E4M3 exponent offset, detail::kBlockScaleShift); each
# shift is its own extension so caches whose products already fit pay nothing.
MAX_BLOCK_SCALE_SHIFT = 7


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version(
        (13, 4),
        component="Q8KV4 prefill attention",
    )


@lru_cache(maxsize=None)
def _target_arch(device=None) -> str:
    arch = _resolve_target_arch(
        device,
        component="Q8KV4 prefill attention",
        supported_arches=_SUPPORTED_ARCHES,
        env_var=FMHA_SM100_PREFILL_Q8KV4_ARCH,
    )
    if arch == "107a" and _cutlass_version() < (4, 8):
        raise RuntimeError(
            "Q8KV4 prefill attention on SM107 requires CUTLASS 4.8 or newer; set "
            f"CUTLASS_ROOT (found {_cutlass_version()} at {_cutlass_root()})"
        )
    return arch


def _qmul4_disabled() -> bool:
    value = os.environ.get(DISABLE_QMUL4_ENV, "").strip().lower() or "0"
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{DISABLE_QMUL4_ENV} must be 0 or 1, got {value!r}")
    return value in ("1", "true")


@cache
def _dequant_mode(arch: str) -> str:
    """QMUL4 where the toolchain accepts it for ``arch``, the packed-FP16 path otherwise."""

    probe_root = _cache_root() / "capability_probes"
    if _qmul4_disabled() or not supports_qmul4(arch, probe_root=probe_root):
        return _FP16_DEQUANT
    return _QMUL4_DEQUANT


def _cache_root() -> Path:
    """Share the package JIT cache location; fall back to the torch extension dir."""
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR") or os.environ.get(
        "TORCH_EXTENSIONS_DIR"
    )
    root = (
        Path(explicit).expanduser()
        if explicit
        else Path.home() / ".cache/minfer/fmha_sm100"
    )
    return root / "prefill_q8kv4"


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


@lru_cache(maxsize=1)
def _namespace() -> _jit_cache.Namespace:
    """The cache namespace of the selected toolchain (see ``fmha_sm100._jit_cache``)."""
    return _jit_cache.Namespace(_cache_root(), _cuda_home())


def _import_extension(module_name: str, library: Path):
    """Import a built extension the way ``cpp_extension.load`` does, without running ninja."""
    spec = importlib.util.spec_from_file_location(module_name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class JitSpec:
    """One compile-time configuration: target architecture, block-scale shift, dequant path."""

    target_arch: str
    block_scale_shift: int = 0
    dequant_mode: str = _QMUL4_DEQUANT
    variant_name: str = "prefill_attention_q8kv4"
    q_heads_per_kv: int = 16
    head_dim: int = 128
    page_size: int = 128
    q_stages: int = 3
    score_stages: int = 2

    @property
    def uri(self) -> str:
        return (
            f"{self.variant_name}_shift{self.block_scale_shift}_{self.dequant_mode}_"
            f"{self.target_arch}"
        )

    @property
    def module_name(self) -> str:
        return f"_fmha_sm100_{self.uri}"

    def _template_params(self) -> dict:
        return {
            "q_heads_per_kv": self.q_heads_per_kv,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "q_stages": self.q_stages,
            "score_stages": self.score_stages,
        }

    def _load_arguments(self, cache_dir: Path) -> dict:
        shift_define = f"-DFMHA_SM100_PREFILL_Q8KV4_BLOCK_SCALE_SHIFT={self.block_scale_shift}"
        has_qmul4 = int(self.dequant_mode == _QMUL4_DEQUANT)
        return {
            "sources": [
                str(_API / "prefill_attention_api.cpp"),
                str(_API / "prefill_attention_binding.cpp"),
                str(cache_dir / "prefill_attention_inst.cu"),
            ],
            "extra_include_paths": [
                str(path)
                for path in (
                    _API,
                    _INCLUDE,
                    _cutlass_root() / "include",
                    _cutlass_root() / "tools/util/include",
                )
            ],
            "extra_cflags": ["-O3", "-DNDEBUG", "-std=c++20", shift_define],
            "extra_cuda_cflags": [
                "-O3",
                "-DNDEBUG",
                "-lineinfo",
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-static-global-template-stub=false",
                "-Xptxas=-O3",
                # An explicit gencode keeps torch from adding its own arch list, which does not
                # know SM107.
                f"-gencode=arch=compute_{self.target_arch},code=sm_{self.target_arch}",
                shift_define,
                f"-DFMHA_SM100_PREFILL_Q8KV4_HAS_QMUL4={has_qmul4}",
            ],
            "extra_ldflags": ["-lcuda", f"-Wl,-rpath,{_cuda_home() / 'lib64'}"],
        }

    def recipe(self) -> _jit_cache.Recipe:
        """The extension's cache recipe. The key holds the torch and Python builds the
        extension links against, so their headers are left out of the recorded inputs."""
        import torch

        arguments = self._load_arguments(Path("<build>"))
        return _jit_cache.Recipe(
            _namespace(),
            self.uri,
            {
                "load": arguments,
                "target_arch": self.target_arch,
                "params": self._template_params(),
                "torch": torch.__version__,
                "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "python": sys.version,
            },
            templates=(_TEMPLATES / "prefill_attention_inst.cu.jinja",),
            covered=(
                os.path.realpath(Path(torch.__file__).parent) + os.sep,
                os.path.realpath(sysconfig.get_paths()["include"]) + os.sep,
            ),
        )

    def build_and_load(self):
        if not 0 <= self.block_scale_shift <= MAX_BLOCK_SCALE_SHIFT:
            raise ValueError(f"block_scale_shift must be in [0, {MAX_BLOCK_SCALE_SHIFT}]")
        recipe = self.recipe()
        library = recipe.lookup()
        if library is not None:
            return _import_extension(self.module_name, library)
        built = {}

        def builder(cache_dir: Path) -> Path:
            built["module"] = self._compile(cache_dir)
            return cache_dir / f"{self.module_name}.so"

        library = recipe.build(builder)
        # cpp_extension.load already imported what this process built; a library another
        # process published while this one waited for the lock is imported here.
        return built.get("module") or _import_extension(self.module_name, library)

    def _compile(self, cache_dir: Path):
        template = jinja2.Template(
            (_TEMPLATES / "prefill_attention_inst.cu.jinja").read_text(encoding="utf-8")
        )
        _write_text_if_changed(
            cache_dir / "prefill_attention_inst.cu", template.render(**self._template_params())
        )
        started_at = time.time()
        previous_cuda_home = cpp_extension.CUDA_HOME
        cpp_extension.CUDA_HOME = str(_cuda_home())
        try:
            extension = cpp_extension.load(
                name=self.module_name,
                **self._load_arguments(cache_dir),
                build_directory=str(cache_dir),
                verbose=False,
                with_cuda=True,
            )
        finally:
            cpp_extension.CUDA_HOME = previous_cuda_home
        logger.info(
            "Compiled fmha_sm100.prefill_q8kv4 in %.1fs",
            time.time() - started_at,
        )
        return extension


def gen_jit_spec(device=None, block_scale_shift: int = 0) -> JitSpec:
    """Return the compile-time configuration for a device and block-scale shift."""

    arch = _target_arch(device)
    return JitSpec(
        target_arch=arch,
        block_scale_shift=int(block_scale_shift),
        dequant_mode=_dequant_mode(arch),
    )


@lru_cache(maxsize=None)
def _load_extension_for_arch(arch: str, block_scale_shift: int = 0):
    return JitSpec(
        target_arch=arch,
        block_scale_shift=block_scale_shift,
        dequant_mode=_dequant_mode(arch),
    ).build_and_load()


def load_extension(device=None, block_scale_shift: int = 0):
    """Build and load the extension for the tensor architecture and block-scale shift."""

    return _load_extension_for_arch(_target_arch(device), int(block_scale_shift))


__all__ = [
    "DISABLE_QMUL4_ENV",
    "FMHA_SM100_PREFILL_Q8KV4_ARCH",
    "MAX_BLOCK_SCALE_SHIFT",
    "JitSpec",
    "gen_jit_spec",
    "load_extension",
]
