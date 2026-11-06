"""Text-only RedHatAI Gemma 4 12B FP8 Dynamic recipe; no model replacement."""

from collections.abc import Mapping

from ...optimizations.compilation import BREAKABLE_CUDAGRAPH
from ...optimizations.fp8 import FUSED_FP8


MODEL_ID = "RedHatAI/gemma-4-12B-it-FP8-Dynamic"
MODEL_REVISION = "2083adceb2a3df0465e6021e08a7196d21dbbb44"
ARCHITECTURES = ("Gemma4UnifiedForConditionalGeneration",)
# The pinned vLLM already implements this architecture.
MODEL_CLASSES = {}
OPTIONS = (BREAKABLE_CUDAGRAPH, FUSED_FP8)


def matches(arguments):
    from ...runtime.vllm import get_adapter

    return "gemma-4-12b-it-fp8-dynamic" in get_adapter().model_name(arguments).lower()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _validate_execution(arguments):
    parallel = getattr(arguments, "parallel_config", arguments)
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise NotImplementedError("Gemma4-12B-FP8 on RTX Spark is currently tested with " + name + "=1")
    if getattr(arguments, "speculative_config", None) is not None or getattr(arguments, "use_replayssm", False):
        raise NotImplementedError("Gemma4-12B-FP8 recipe currently supports ordinary decoding without speculation")
    load = getattr(arguments, "load_config", arguments)
    if getattr(load, "load_format", None) == "mbx":
        raise ValueError("Gemma4-12B-FP8 uses the standard safetensors loader; mbx is Qwen-specific")
    model = getattr(arguments, "model_config", None)
    mm = getattr(model, "multimodal_config", None)
    text_only = getattr(arguments, "language_model_only", getattr(mm, "language_model_only", False))
    if not text_only:
        raise NotImplementedError("Gemma4-12B-FP8 recipe currently requires language-model-only=true")
    if getattr(arguments, "enable_mm_embeds", getattr(mm, "enable_mm_embeds", False)):
        raise NotImplementedError("Gemma4-12B-FP8 recipe does not support multimodal embeddings")


def _validate_checkpoint(model):
    hf = getattr(model, "hf_config", None)
    if hf is None:
        raise ValueError("Gemma4-12B-FP8 requires actual checkpoint metadata")
    if _field(hf, "model_type") != "gemma4_unified" or not set(ARCHITECTURES).intersection(
            _field(hf, "architectures", ()) or ()):
        raise ValueError("Gemma4-12B-FP8 requires Gemma4UnifiedForConditionalGeneration metadata")
    text = _field(hf, "text_config")
    if (_field(text, "hidden_size"), _field(text, "num_hidden_layers"), _field(text, "enable_moe_block")) != (
            3840, 48, False):
        raise ValueError("Gemma4-12B-FP8 recipe supports the 12B dense checkpoint")
    quant = _field(hf, "quantization_config", {})
    if _field(quant, "quant_method") != "compressed-tensors" or _field(quant, "format") != "float-quantized":
        raise ValueError("Gemma4-12B-FP8 requires compressed-tensors float-quantized metadata")
    groups = _field(quant, "config_groups", {})
    if not isinstance(groups, Mapping) or len(groups) != 1:
        raise ValueError("Gemma4-12B-FP8 requires one FP8 Linear quantization group")
    group = next(iter(groups.values()))
    if _field(group, "targets") != ["Linear"]:
        raise ValueError("Gemma4-12B-FP8 requires the checkpoint's Linear quantization targets")
    for name, strategy, dynamic in (("weights", "channel", False), ("input_activations", "token", True)):
        scheme = _field(group, name, {})
        actual = tuple(_field(scheme, key) for key in ("type", "num_bits", "strategy", "dynamic", "symmetric"))
        if actual != ("float", 8, strategy, dynamic, True) or _field(scheme, "group_size") is not None:
            raise ValueError("Gemma4-12B-FP8 requires static per-channel FP8 weights and dynamic per-token FP8 activations")


def prepare(arguments, options):
    """Validate declared scope; generic startup owns dependencies and providers."""
    _validate_execution(arguments)
    from ...runtime.bootstrap import initialize

    initialize()


def validate(config, options):
    _validate_execution(config)
    _validate_checkpoint(config.model_config)
