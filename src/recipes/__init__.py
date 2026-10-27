"""Small, lazy registry of model preparation and validation callbacks."""

from dataclasses import dataclass
import importlib
from typing import Callable


@dataclass(frozen=True)
class ModelIntegration:
    prepare: Callable
    validate: Callable
    matches: Callable | None = None


_MODELS = {"qwen3.8-flash-next": __package__ + ".qwen3_8"}


def register_model(name, prepare, validate, *, matches=None):
    """Register another model without editing the platform or kernel runtime."""
    if not isinstance(name, str) or not name:
        raise ValueError("A model integration needs a name")
    if name in _MODELS:
        raise ValueError("Model integration is already registered: " + name)
    if not callable(prepare) or not callable(validate) or (matches is not None and not callable(matches)):
        raise TypeError("Model preparation, validation and optional matching must be callable")
    _MODELS[name] = ModelIntegration(prepare, validate, matches)


def _load(name):
    integration = _MODELS[name]
    if isinstance(integration, str):
        module = importlib.import_module(integration)
        integration = ModelIntegration(module.prepare, module.validate, module.matches)
        _MODELS[name] = integration
    return integration


def resolve_model(arguments, *, name=None):
    """Resolve an explicit recipe, or an optional match for ordinary vLLM args."""
    if name is not None:
        if name not in _MODELS:
            raise ValueError("Unsupported RTX Spark model recipe: " + name)
        return _load(name)
    for registered in _MODELS:
        integration = _load(registered)
        if integration.matches is not None and integration.matches(arguments):
            return integration
    return None
