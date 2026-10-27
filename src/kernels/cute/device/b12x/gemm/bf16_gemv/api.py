"""Public surface for gemm.bf16_gemv (docs in the op ``__init__``)."""

from __future__ import annotations

import os

import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import SMALL_M_MAX
from ._kernel import bf16_gemv_small_n  # noqa: F401  (registers the op; alias)
from ._kernel import (
    precompile_bf16_gemv_small_n as precompile,
)

# Routing hints for integrations choosing only narrow decode projections.
# mm itself supports broad geometries and owns tensor-core/SIMT dispatch;
# these hints are not support limits or required integration-side policy.
SMALL_N_GEMV_MAX_OUT = 1024
SMALL_N_GEMV_MIN_IN = 1024


def is_disabled() -> bool:
    """True when ``B12X_DISABLE_BF16_GEMV`` turns the routing off
    entirely (debug isolation switch)."""
    return os.environ.get("B12X_DISABLE_BF16_GEMV", "").lower() in (
        "1",
        "true",
        "yes",
    )


def mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Native ``x @ weight.T + bias`` with BF16 or FP32 operands.

    Accumulation and bias addition precede the final output cast. BF16/BF16
    broad multi-row inputs use tensor cores; FP32 operands remain unrounded
    in the SIMT path. Live rows and strides are runtime arguments; unsupported
    inputs fail rather than switching compute providers. ``out`` selects
    allocation-free serving and supports padded row strides.
    """
    if out is None:
        return torch.ops.b12x.bf16_gemv_small_n(x, weight, bias, output_dtype)
    if output_dtype is not None and out.dtype != output_dtype:
        raise ValueError("output_dtype disagrees with caller-owned out")
    torch.ops.b12x.bf16_gemv_small_n_out(x, weight, out, bias)
    return out


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, unless disabled
    via ``B12X_DISABLE_BF16_GEMV``."""
    if is_disabled():
        return False
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "mm",
    "bf16_gemv_small_n",
    "precompile",
    "is_supported",
    "is_disabled",
    "SMALL_M_MAX",
    "SMALL_N_GEMV_MAX_OUT",
    "SMALL_N_GEMV_MIN_IN",
]
