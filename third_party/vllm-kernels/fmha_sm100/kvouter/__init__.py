# Copyright (c) 2026 Fireworks AI
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 sparse-attention kernels."""

from .interface import (
    BLOCK_SIZE,
    HEAD_DIM,
    can_run_sparse_kvouter,
    kvouter_attention,
)

__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "can_run_sparse_kvouter",
    "kvouter_attention",
]
