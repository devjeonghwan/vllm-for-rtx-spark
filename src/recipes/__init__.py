"""Small, lazy registry of model preparation and validation callbacks."""

from dataclasses import dataclass, field
import importlib
from typing import Callable


@dataclass(frozen=True)
class ModelIntegration:
    prepare: Callable
    validate: Callable
    matches: Callable | None = None
    options: tuple = ()
    architectures: tuple[str, ...] = ()
    model_classes: dict[str, str] = field(default_factory=dict)


_MODELS = {}


def register_builtin_models():
    """Declare bundled recipes without importing their model implementations."""
    _MODELS.setdefault("qwen3.8-flash-next", __package__ + ".qwen3_8")


def register_model(name, prepare, validate, *, matches=None, options=(),
                   architectures=(), model_classes=None):
    """Register another model without editing the platform or kernel runtime."""
    if not isinstance(name, str) or not name:
        raise ValueError("A model integration needs a name")
    if name in _MODELS:
        raise ValueError("Model integration is already registered: " + name)
    if not callable(prepare) or not callable(validate) or (matches is not None and not callable(matches)):
        raise TypeError("Model preparation, validation and optional matching must be callable")
    from ..optimizations import resolve_options

    options = tuple(options)
    resolve_options({}, options)
    architectures = tuple(architectures)
    classes = dict(model_classes or {})
    if any(not isinstance(architecture, str) for architecture in (*architectures, *classes)):
        raise TypeError("Architecture names must be strings")
    if any(not isinstance(value, str) or ":" not in value for value in classes.values()):
        raise TypeError("Architecture implementations must be lazy module:class strings")
    _MODELS[name] = ModelIntegration(prepare, validate, matches, options, architectures, classes)


def _load(name):
    integration = _MODELS[name]
    if isinstance(integration, str):
        module = importlib.import_module(integration)
        integration = ModelIntegration(module.prepare, module.validate, getattr(module, "matches", None),
                                       tuple(getattr(module, "OPTIONS", ())),
                                       tuple(getattr(module, "ARCHITECTURES", ())),
                                       dict(getattr(module, "MODEL_CLASSES", {})))
        _MODELS[name] = integration
    return integration


def resolve_model_name(arguments, *, name=None):
    """Select one named recipe; automatic matches must be unambiguous."""
    register_builtin_models()
    from ..runtime.vllm import get_adapter

    adapter = get_adapter()
    if name is None:
        name = adapter.integration_options(arguments).get("model")
    if name is not None:
        if name not in _MODELS:
            raise ValueError("Unsupported RTX Spark model recipe: " + name)
        return name
    architectures = set(adapter.architecture_names(arguments))
    matches = []
    for registered in _MODELS:
        integration = _load(registered)
        if ((integration.matches is not None and integration.matches(arguments))
                or architectures.intersection(integration.architectures)):
            matches.append(registered)
    if len(matches) > 1:
        raise ValueError("Ambiguous RTX Spark model recipes: " + ", ".join(matches)
                         + "; set additional_config.rtx_spark.model explicitly")
    return matches[0] if matches else None


def resolve_model(arguments, *, name=None):
    """Resolve an explicit recipe, or an unambiguous automatic match."""
    selected = resolve_model_name(arguments, name=name)
    return _load(selected) if selected is not None else None


def register_architectures():
    """Use the official registry only for explicitly supplied new architectures."""
    from ..runtime.vllm import get_adapter

    models = {}
    for name in _MODELS:
        for architecture, implementation in _load(name).model_classes.items():
            if architecture in models and models[architecture] != implementation:
                raise ValueError("Conflicting recipe architecture: " + architecture)
            models[architecture] = implementation
    if models:
        get_adapter().register_architectures(models)
