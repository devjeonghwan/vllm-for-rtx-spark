"""Resource locations and model configuration; no GPU work at import time."""

from __future__ import annotations

import hashlib
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
    return json.loads((resources_root() / "native" / "manifest.json").read_text(encoding="utf-8"))


def artifact(name, *, verify=True):
    record = native_manifest()["files"][name]
    path = resources_root() / "native" / record["file"]
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError("RTX Spark native artifact changed: " + name)
    return path, record


def _spark_options(args):
    additional = getattr(args, "additional_config", None) or {}
    if not isinstance(additional, dict):
        raise TypeError("additional_config must be an object")
    options = additional.get("rtx_spark", {})
    if not isinstance(options, dict):
        raise TypeError("additional_config.rtx_spark must be an object")
    if "model" in options and not isinstance(options["model"], str):
        raise TypeError("rtx_spark.model must be a string")
    return options


def _model_integration(arguments, options):
    from ..recipes import resolve_model

    integration = resolve_model(arguments, name=options.get("model"))
    if integration is None and options:
        raise ValueError("No RTX Spark model integration matches the supplied settings")
    return integration


def prepare_engine_args(args):
    """Called before ModelConfig is created, while original hook envs are mutable."""
    options = _spark_options(args)
    integration = _model_integration(args, options)
    if integration is not None:
        integration.prepare(args, options)


def configure_engine(config):
    """Preserve vLLM's config; validate the model integration, without allocation."""
    options = _spark_options(config)
    integration = _model_integration(config, options)
    if integration is not None:
        integration.validate(config, options)
    return config
