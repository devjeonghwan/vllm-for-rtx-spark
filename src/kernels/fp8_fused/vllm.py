"""Pinned vLLM scaled-MM provider. Selection is scoped to the current engine."""

import platform
import sys

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)

from . import enabled


class RtxSparkFusedChannelWiseFP8ScaledMMLinearKernel(ChannelWiseTorchFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        if sys.platform != "win32" or platform.machine().lower() not in ("arm64", "aarch64"):
            return False, "requires native Windows ARM64"
        try:
            config = get_current_vllm_config()
        except AssertionError:
            return False, "requires an engine configuration"
        if not enabled(config):
            return False, "rtx_spark.fused_fp8 is disabled"
        if compute_capability is None:
            from vllm.platforms import current_platform
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability is not None else None
        if compute_capability != 121:
            return False, "validated device family is sm_121"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if (config.activation_quant_key != kFp8DynamicTokenSym
                or config.weight_quant_key != kFp8StaticChannelSym):
            return False, "requires dynamic token FP8 and static channel FP8 with FP32 scales"
        if config.out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False, "unsupported output dtype"
        return True, None

    def __init__(self, config, layer_param_names):
        super().__init__(config, layer_param_names)
        # Register before Dynamo tracing. No tensor admission/query occurs in
        # the compiled forward; the opaque implementation owns those checks.
        from .ops import scaled_mm
        self._rtx_spark_scaled_mm = scaled_mm

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        return self._rtx_spark_scaled_mm(A, B, As, Bs, bias, out_dtype, output_shape)
