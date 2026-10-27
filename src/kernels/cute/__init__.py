"""Install a kernel implementation without exposing its build mechanism."""
import importlib
import os
from pathlib import Path
import sys

from . import assets

_ALIASES = {
    "original_b12x_aot": "b12x_aot",
    "hc_aot": "hc_aot",
    "original_skinny_complete_aot": "skinny_aot",
    "original_gdn_aot": "gdn_aot",
    "original_cute_coff_runtime": "coff_runtime",
}
_installed_mode = None
_bindings_installed = False
_runtime_providers = {}
_preferred_runtime = "aot"


def register_runtime(name, configure_provider, *, preferred=False):
    """Register local metadata; use a provider module to replay it in workers."""
    global _preferred_runtime
    if name in {"auto", "aot"} or name in _runtime_providers or not callable(configure_provider):
        raise ValueError("Invalid or duplicate runtime provider: " + name)
    _runtime_providers[name] = configure_provider
    if preferred:
        _preferred_runtime = name


def install(mode=None):
    """Register portable host bindings before vLLM imports model modules.

    AOT is the tested default. A future native runtime is registered here;
    a Python package's presence alone is not considered support.
    """
    global _installed_mode, _bindings_installed
    from ..providers import initialize_providers

    initialize_providers()
    requested = mode or os.environ.get("VLLM_RTX_SPARK_CUTE_MODE")
    if requested == "auto":
        requested = _preferred_runtime
    if requested is not None and requested != "aot" and requested not in _runtime_providers:
        raise RuntimeError("Kernel runtime provider is not installed: " + requested)
    if _installed_mode is not None and requested is not None and _installed_mode != requested:
        raise RuntimeError("Change kernel mode before model loading, not during serving")
    root = Path(__file__).with_name("device")
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    if not _bindings_installed:
        for alias, module in _ALIASES.items():
            target = importlib.import_module("." + module, __name__)
            existing = sys.modules.get(alias)
            if existing is not None and existing is not target:
                raise RuntimeError("Another kernel package owns the host binding: " + alias)
            sys.modules[alias] = target
        _bindings_installed = True
    if requested in (None, "aot"):
        os.environ["MBX_B12X_AOT"] = "1"
        os.environ["MBX_ORIGINAL_FI_GDN_AOT"] = "1"
        os.environ["MBX_HC_AOT_MANIFEST"] = str(assets.resolve("hc.json"))
        os.environ["MBX_SKINNY_AOT_MANIFEST"] = str(assets.resolve("skinny.json"))
    else:
        _runtime_providers[requested]()
    if requested is not None:
        os.environ["VLLM_RTX_SPARK_CUTE_MODE"] = requested
        _installed_mode = requested


def configure(mode="auto"):
    """Select a provider before model loading; unknown providers fail explicitly."""
    install(mode)


def status():
    return {"mode": _installed_mode, "target": assets.metadata("manifest")["target"],
            "build_compiler_required_for_inference": False if _installed_mode == "aot" else None}
