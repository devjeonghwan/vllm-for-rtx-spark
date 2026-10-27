"""The vLLM 0.32 EngineArgs, config and lazy architecture-registration ABI."""

import json


_registered_architectures = {}
_RECIPE_SELECTION = "_rtx_spark_prepared_recipe"


def integration_options(arguments):
    additional = getattr(arguments, "additional_config", None)
    if additional is None:
        additional = {}
    if not isinstance(additional, dict):
        raise TypeError("additional_config must be an object")
    options = additional.get("rtx_spark", {})
    if not isinstance(options, dict):
        raise TypeError("additional_config.rtx_spark must be an object")
    if "model" in options and not isinstance(options["model"], str):
        raise TypeError("rtx_spark.model must be a string")
    return options


def prepared_recipe(arguments):
    """Distinguish an unprepared object from a prepared ordinary-vLLM model."""
    return hasattr(arguments, _RECIPE_SELECTION), getattr(arguments, _RECIPE_SELECTION, None)


def remember_recipe(arguments, name):
    """Keep selection on EngineArgs; VllmConfig must contain only declared fields."""
    setattr(arguments, _RECIPE_SELECTION, name)


def bind_recipe(arguments, name):
    """Carry a selected recipe to workers without editing caller-owned maps."""
    if name is None:
        return
    additional = getattr(arguments, "additional_config", None) or {}
    options = integration_options(arguments)
    updated = dict(additional)
    updated["rtx_spark"] = dict(options, model=name)
    arguments.additional_config = updated


def model_name(arguments):
    config = getattr(arguments, "model_config", None)
    return str(getattr(arguments, "model", getattr(config, "model", "")))


def architecture_names(arguments):
    config = getattr(arguments, "model_config", arguments)
    hf = getattr(config, "hf_config", None)
    names = getattr(hf, "architectures", None)
    return tuple(names or ())


def speculative_tokens(arguments):
    speculative = getattr(arguments, "speculative_config", None)
    if isinstance(speculative, str):
        speculative = json.loads(speculative)
    if isinstance(speculative, dict):
        return speculative.get("num_speculative_tokens", speculative.get("num-speculative-tokens"))
    return getattr(speculative, "num_speculative_tokens", None)


def tensor_parallel_size(config):
    return config.parallel_config.tensor_parallel_size


def wrap_hf_overrides(arguments, transform):
    overrides = getattr(arguments, "hf_overrides", None)
    if isinstance(overrides, dict) and overrides.get("text_config"):
        arguments.hf_overrides = transform(overrides)


def register_architectures(models):
    """Keep architecture classes as lazy import strings in vLLM's registry."""
    pending = {name: implementation for name, implementation in models.items()
               if _registered_architectures.get(name) != implementation}
    if not pending:
        return
    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for name, implementation in pending.items():
        if name in supported:
            raise ValueError("vLLM already owns architecture: " + name)
        if not isinstance(implementation, str) or ":" not in implementation:
            raise TypeError("Architecture implementations must be lazy module:class strings")
    for name, implementation in pending.items():
        ModelRegistry.register_model(name, implementation)
        _registered_architectures[name] = implementation


def get_b12x_mxfp8_linear():
    from vllm.utils.b12x import get_b12x_mxfp8_linear as original

    return original()
