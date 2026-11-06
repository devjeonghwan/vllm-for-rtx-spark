"""Acquire the Qwen MTP policy only for its prepared model instance."""


def get_depth_helper(config):
    """Keep Qwen's native policy out of ordinary/non-MTP model startup.

    The recipe selection is serialized in VllmConfig. Do not select a recipe
    again from a filename or an inherited MBX environment variable. Instances
    retain the returned original helper, so hot paths do no registry lookup.
    """
    from .vllm import get_adapter

    adapter = get_adapter()
    if adapter.integration_options(config).get("model") != "qwen3.8-flash-next":
        return None
    if "Qwen4ExpForConditionalGeneration" not in adapter.architecture_names(config):
        return None
    speculative = getattr(config, "speculative_config", None)
    method = (speculative.get("method") if isinstance(speculative, dict)
              else getattr(speculative, "method", None))
    if method != "mtp":
        return None  # In particular, fixed-block DFlash must not borrow MTP state.
    from ..recipes.qwen3_8_flash_next.helpers import mbx_depth

    return mbx_depth
