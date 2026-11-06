"""Generic dense FP8 provider choice; each model recipe opts into the declaration."""

from . import Option


FUSED_FP8 = Option("fused_fp8", bool, False)
