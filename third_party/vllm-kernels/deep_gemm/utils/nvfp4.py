"""Reference preparation helpers for NVFP4 MegaMoE."""
from typing import Tuple

import torch

from .math import _quantize_to_fp4_e2m1, cast_back_from_fp4


def per_token_cast_to_nvfp4(x: torch.Tensor, global_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantizer: packed E2M1 and packed per-16 E4M3 scale bytes.

    ``x ~= dequantize(packed) * block_scale * global_scale``. The caller chooses
    the positive tensor-wide dequantization scale (typically tensor amax/2688).
    This PyTorch helper is intended for preparation/testing, not timed inference.
    """
    assert x.dim() == 2 and x.size(1) % 64 == 0
    assert global_scale > 0 and global_scale < float('inf')
    m, n = x.shape
    blocks = x.float().reshape(m, n // 16, 16)
    sf = (blocks.abs().amax(-1) / (6.0 * global_scale)).clamp(2 ** -9, 448).to(torch.float8_e4m3fn)
    codes = _quantize_to_fp4_e2m1(blocks / (sf.float().unsqueeze(-1) * global_scale)).reshape(m, n // 2, 2)
    packed = (codes[..., 0] & 15) | ((codes[..., 1] & 15) << 4)
    return packed.contiguous(), sf.view(torch.int32).contiguous()


def cast_back_from_nvfp4(packed: torch.Tensor, sf: torch.Tensor, global_scale: float = 1.0) -> torch.Tensor:
    """Dequantize row-major, int32-packed per-16 E4M3 scales."""
    scales = sf.contiguous().view(torch.float8_e4m3fn).float() * global_scale
    return cast_back_from_fp4(packed, scales, gran_k=16)
