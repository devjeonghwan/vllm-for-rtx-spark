"""Compatibility alias for tensor-scaled FP8 ``blockscaled`` calls.

New code should use ``b12x.gemm.blockscaled.pack_weight``, ``.mm``, and
``.prewarm``. This module retains its original names for compatibility.

Example:
    from b12x.gemm import blockscaled

    packed = blockscaled.pack_weight(
        weight_fp8,
        input_scale * weight_scale,
    )
    output = blockscaled.mm(input_fp8, packed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="tensor_fp8_linear",
    group="gemm",
    api_style="oneshot",
    entry_points=("Weight", "mm", "pack_weight", "prewarm", "is_supported"),
    dtypes=("fp8_e4m3", "bf16", "fp16"),
    recipes=("tensor_fp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="1bc4f82",
        paths=("b12x/gemm/tensor_fp8_linear",),
    ),
    test_path="tests/gemm/test_tensor_fp8_linear.py",
    since="1.0.1",
)

if TYPE_CHECKING:
    from .api import Weight, is_supported, mm, pack_weight, prewarm  # noqa: F401

install_lazy_api(globals(), META)
