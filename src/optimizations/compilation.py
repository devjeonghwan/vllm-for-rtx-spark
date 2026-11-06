"""Recipe policy for the pinned vLLM compilation/graph boundary; CPU only."""

import json
import os

from . import Option


_OPTION = "breakable_cudagraph"
_ENVIRONMENT = "VLLM_USE_BREAKABLE_CUDAGRAPH"
# Keep a policy projection distinct from the caller's environment when another
# engine is prepared in this process. Workers use their serialized engine option.
_projection = None


def _refresh_environment_cache():
    import sys

    envs = sys.modules.get("vllm.envs")
    reader = getattr(envs, "__getattr__", None)
    if hasattr(reader, "cache_clear"):
        reader.cache_clear()


def restore_compilation_environment():
    """Remove only our previous projection when preparing an ordinary model."""
    global _projection
    if _projection is None:
        return
    original, projected = _projection
    _projection = None
    if os.environ.get(_ENVIRONMENT) != projected:
        return  # Preserve a new external caller setting.
    if original is None:
        os.environ.pop(_ENVIRONMENT, None)
    else:
        os.environ[_ENVIRONMENT] = original
    if original != projected:
        _refresh_environment_cache()


def _external_environment():
    current = os.environ.get(_ENVIRONMENT)
    if _projection is not None and current == _projection[1]:
        return _projection[0]
    return current


def resolve_compilation_options(options, declarations):
    """Explicit flat/grouped config > caller environment > recipe default.

    The ordinary resolver still owns duplicate, type, and unknown-key checks.
    This boundary only supplies an omitted declared compilation policy from the
    existing vLLM environment variable. It introduces no alternate launcher.
    """
    if not any(option.name == _OPTION for option in declarations):
        return options
    nested = options.get("optimizations", {})
    if _OPTION in options or (isinstance(nested, dict) and _OPTION in nested):
        return options
    value = _external_environment()
    if value is None:
        return options
    if value not in ("0", "1"):
        raise ValueError(_ENVIRONMENT + " must be 0 or 1")
    return dict(options, **{_OPTION: value == "1"})


def _compilation_field(arguments, field):
    config = getattr(arguments, "compilation_config", None)
    if isinstance(config, str):
        config = json.loads(config)
    return config.get(field) if isinstance(config, dict) else getattr(config, field, None)


def _mode_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        names = {"none": 0, "stock_torch_compile": 1, "dynamo_trace_once": 2, "vllm_compile": 3}
        return int(value) if value.isdigit() else names.get(value.lower(), value)
    return value


def validate_breakable_cudagraph(arguments, enabled):
    if arguments is None:
        return
    mode = _mode_number(_compilation_field(arguments, "mode"))
    model = getattr(arguments, "model_config", None)
    enforce_eager = getattr(arguments, "enforce_eager", False) or getattr(model, "enforce_eager", False)
    if mode not in (None, 0) and (enforce_eager
                                or os.environ.get("TORCH_COMPILE_DISABLE") == "1"):
        raise ValueError("compilation-config.mode requests Torch Compile but enforce_eager or "
                         "TORCH_COMPILE_DISABLE=1 disables it")
    if enabled and mode not in (None, 0):
        raise ValueError("rtx_spark.breakable_cudagraph=true conflicts with compilation-config.mode; "
                         "the pinned breakable graph path forces mode=NONE. Set breakable_cudagraph=false "
                         "to request Torch Compile")


def apply_breakable_cudagraph(arguments, enabled):
    """Project one validated engine policy to vLLM and serialize it for spawn."""
    global _projection
    validate_breakable_cudagraph(arguments, enabled)
    current = os.environ.get(_ENVIRONMENT)
    original = _external_environment()
    projected = "1" if enabled else "0"
    _projection = (original, projected)
    os.environ[_ENVIRONMENT] = projected
    # vLLM normally caches env reads after startup. Its existing cache API must
    # see a changed engine policy before model/graph decorators are imported.
    if current != projected:
        _refresh_environment_cache()
    additional = getattr(arguments, "additional_config", None) or {}
    options = additional.get("rtx_spark", {})
    arguments.additional_config = dict(additional, rtx_spark=dict(options, **{_OPTION: enabled}))


BREAKABLE_CUDAGRAPH = Option(_OPTION, bool, True,
                            validate=validate_breakable_cudagraph,
                            apply=apply_breakable_cudagraph)


def prepare_compilation_policy(arguments):
    """Replay the selected recipe policy before config decisions/model imports.

    Used by VllmConfig and worker startup. It applies only this declaration;
    Qwen preparation, providers, weights and other optimization hooks stay out
    of a common non-Qwen startup path.
    """
    from ..recipes import resolve_model
    from ..runtime.vllm import get_adapter
    from . import resolve_options

    options = get_adapter().integration_options(arguments)
    # EngineArgs serializes a selected name before VllmConfig construction.
    # Absence must keep an earlier no-match decision, even when HF architecture
    # metadata only becomes available now. Direct VllmConfig callers can select
    # a recipe explicitly; replay never performs a second automatic selection.
    selected = options.get("model")
    if selected is None:
        restore_compilation_environment()
        return
    integration = resolve_model(arguments, name=selected)
    if integration is None or not any(option.name == _OPTION for option in integration.options):
        restore_compilation_environment()
        return
    supplied = resolve_compilation_options(options, integration.options)
    resolved = resolve_options(supplied, integration.options)
    if _OPTION in resolved:
        apply_breakable_cudagraph(arguments, resolved[_OPTION])


def effective_compilation_policy(config):
    """Report final values without claiming compilation or Graph replay ran."""
    from ..runtime.vllm import get_adapter

    def name(value):
        return getattr(value, "name", value)

    compilation = getattr(config, "compilation_config", None)
    return {
        "breakable_cudagraph": get_adapter().integration_options(config).get(_OPTION),
        "breakable_environment": os.environ.get(_ENVIRONMENT),
        "mode": name(getattr(compilation, "mode", None)),
        "backend": getattr(compilation, "backend", None),
        "cudagraph_mode": name(getattr(compilation, "cudagraph_mode", None)),
        "custom_ops": list(getattr(compilation, "custom_ops", ())),
    }
