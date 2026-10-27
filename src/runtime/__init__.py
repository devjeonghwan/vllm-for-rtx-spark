"""Resource locations and model configuration; no GPU work at import time."""

from __future__ import annotations

import json
import os
from pathlib import Path


def source_root():
    """The private integration's Python source, in a checkout or vLLM wheel."""
    return Path(__file__).resolve().parents[1]


def _checkout_root():
    source = source_root()
    if source.name == "src" and (source.parent / "pyproject.toml").is_file():
        return source.parent
    return None


def resources_root():
    """Prebuilt libraries, kernel payloads and recipe data."""
    checkout = _checkout_root()
    return checkout / "assets" if checkout is not None else source_root() / "assets"


def vendor_root():
    """Private third-party Python dependencies, without importing them."""
    checkout = _checkout_root()
    return checkout / "third_party" / "python" if checkout is not None else source_root() / "vendor"


def recipe_helpers(recipe="qwen3_8"):
    return source_root() / "recipes" / recipe / "helpers"


def cache_root():
    if os.environ.get("VLLM_RTX_SPARK_CACHE"):
        return Path(os.environ["VLLM_RTX_SPARK_CACHE"]).expanduser().resolve()
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "vllm-rtx-spark"


def native_manifest():
    manifest = json.loads((resources_root() / "native" / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version"), manifest.get("host"), manifest.get("gpu")) != (1, "win_arm64", "sm_121a"):
        raise RuntimeError("Unsupported native resource schema or target")
    return manifest


def artifact(name):
    record = native_manifest()["files"][name]
    root = (resources_root() / "native").resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Native artifact escapes its package: " + name)
    if not path.is_file():
        raise FileNotFoundError("RTX Spark native artifact is missing: " + str(path))
    return path, record


def _rtx_spark_options(args):
    from .vllm import get_adapter

    return get_adapter().integration_options(args)


def _model_integration(arguments, options, *, prepared=None):
    from ..recipes import resolve_model, resolve_model_name

    if prepared is None:
        name = resolve_model_name(arguments, name=options.get("model"))
    else:
        name = prepared[1]
        requested = options.get("model")
        if requested is not None and requested != name:
            raise ValueError("RTX Spark model recipe changed after preparation")
    integration = resolve_model(arguments, name=name) if name is not None else None
    if integration is None and options:
        raise ValueError("No RTX Spark model integration matches the supplied settings")
    return name, integration


def prepare_engine_args(args):
    """Called before ModelConfig is created, while original hook envs are mutable."""
    from .vllm import get_adapter

    adapter = get_adapter()
    options = _rtx_spark_options(args)
    name, integration = _model_integration(args, options)
    if integration is not None:
        from ..optimizations import resolve_options, validate_options, apply_options

        resolved = resolve_options(dict(options, model=name), integration.options)
        validate_options(args, resolved, integration.options)
        from ..plugin import register

        adapter.bind_recipe(args, name)
        register()
        integration.prepare(args, resolved)
        apply_options(args, resolved, integration.options)
    adapter.remember_recipe(args, name)


def configure_engine(config, *, engine_args=None):
    """Preserve vLLM's config; validate the model integration, without allocation."""
    options = _rtx_spark_options(config)
    from .vllm import get_adapter

    adapter = get_adapter()
    owner = config if engine_args is None else engine_args
    is_prepared, name = adapter.prepared_recipe(owner)
    if engine_args is not None and not is_prepared:
        raise RuntimeError("RTX Spark engine arguments were not prepared")
    name, integration = _model_integration(config, options, prepared=(True, name) if is_prepared else None)
    if integration is not None:
        from ..optimizations import resolve_options, validate_options

        resolved = resolve_options(dict(options, model=name), integration.options)
        validate_options(config, resolved, integration.options)
        integration.validate(config, resolved)
        adapter.bind_recipe(config, name)
    return config
