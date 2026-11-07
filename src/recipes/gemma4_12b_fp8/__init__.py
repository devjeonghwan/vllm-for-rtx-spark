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
    # Recipe identity is an API boundary, not a size/quantization whitelist.
    # The upstream loader and each selected provider validate actual formats.
    hf = getattr(model, "hf_config", None)
    if hf is None:
        raise ValueError("Gemma4-12B-FP8 requires actual checkpoint metadata")
    if _field(hf, "model_type") != "gemma4_unified" or not set(ARCHITECTURES).intersection(
            _field(hf, "architectures", ()) or ()):
        raise ValueError("Gemma4-12B-FP8 requires Gemma4UnifiedForConditionalGeneration metadata")


def prepare(arguments, options):
    """Validate declared scope; generic startup owns dependencies and providers."""
    _validate_execution(arguments)
    from ...runtime.bootstrap import initialize

    initialize()


def validate(config, options):
    _validate_execution(config)
    _validate_checkpoint(config.model_config)
