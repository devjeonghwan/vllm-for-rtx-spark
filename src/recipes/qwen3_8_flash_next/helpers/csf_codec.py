"""Lossless codec-0 NVFP4 scale decoding, independent of Linux libraries.

Public format provenance (Apache-2.0): local-inference-lab/b12x commit
2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88, _lib/quant/nvfp4_csf.py
and tests/quantization/test_nvfp4_csf.py. Checkpoint layout is 16-row
slabs: 16 base E4M3 bytes, followed by row-major low/high offset nibbles.
Each little-endian U32 exception stores a 24-bit logical index and byte value.
Only original scale bytes are reconstructed; no floating-point quantization.
"""
from __future__ import annotations

import struct


def _validate(fixed: bytes, exceptions: bytes, rows: int, columns: int):
    if rows <= 0 or rows % 16 or columns <= 0 or columns % 2:
        raise ValueError('Codec-0 requires 16-row slabs and an even scale width')
    if rows * columns > 1 << 24:
        raise ValueError('Scale plane exceeds codec-0 24-bit index capacity')
    slab_bytes = 16 * (1 + columns // 2)
    if len(fixed) != rows // 16 * slab_bytes or len(exceptions) % 4:
        raise ValueError('Compressed scale tensor extent mismatch')
    for slab in range(rows // 16):
        if max(fixed[slab * slab_bytes:slab * slab_bytes + 16]) > 240:
            raise ValueError('Invalid codec-0 base byte: exceeds 240')
    previous = -1
    for (word,) in struct.iter_unpack('<I', exceptions):
        position = word & 0xffffff
        if not previous < position < rows * columns:
            raise ValueError('Exceptions must be sorted, unique and in range')
        previous = position
    return slab_bytes


def decode_csf_byte_window(fixed: bytes, exceptions: bytes, rows: int, columns: int, *, codec=0, layout=0) -> bytearray:
    """Reconstruct one caller-selected expert projection's E4M3 bytes.

    Output size is rows*columns, not model-wide scale or BF16 weight storage.
    Other CSF codecs/layouts are unsupported and must not be silently guessed.
    """
    if codec != 0 or layout != 0:
        raise ValueError('Only original checkpoint codec-0/layout-0 is supported')
    slab_bytes = _validate(fixed, exceptions, rows, columns)
    result = bytearray(rows * columns)
    for row in range(rows):
        slab = row // 16 * slab_bytes
        within = row % 16
        base = fixed[slab + within]
        packed_start = slab + 16 + within * (columns // 2)
        out = row * columns
        for pair in range(columns // 2):
            code = fixed[packed_start + pair]
            result[out + pair * 2] = base + (code & 15)
            result[out + pair * 2 + 1] = base + (code >> 4)
    for (word,) in struct.iter_unpack('<I', exceptions):
        result[word & 0xffffff] = word >> 24
    return result


def decode_csf_torch(fixed: bytes, exceptions: bytes, rows: int, columns: int, device='cpu', *, codec=0, layout=0):
    """Independent vectorized decoder using native PyTorch integer operations."""
    if codec != 0 or layout != 0:
        raise ValueError('Only original checkpoint codec-0/layout-0 is supported')
    import torch
    slab_bytes = _validate(fixed, exceptions, rows, columns)
    records = torch.frombuffer(bytearray(fixed), dtype=torch.uint8).to(device).reshape(rows // 16, slab_bytes)
    bases = records[:, :16].reshape(rows, 1)
    packed = records[:, 16:].reshape(rows, columns // 2)
    offsets = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, columns)
    decoded = bases + offsets
    if exceptions:
        words = torch.frombuffer(bytearray(exceptions), dtype=torch.int32).to(device=device, dtype=torch.int64) & 0xffffffff
        decoded.reshape(-1)[words & 0xffffff] = (words >> 24).to(torch.uint8)
    return decoded
