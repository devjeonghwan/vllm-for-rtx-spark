"""Replay explicit provider registrations in the API process and spawned workers.

Registration modules expose ``register()`` and only register CPU metadata. They
must not load model weights, native libraries, or execute kernels at import time.
"""

import importlib
import json
import os
import threading


PROVIDER_MODULES = ()
_ENVIRONMENT = "VLLM_RTX_SPARK_PROVIDER_MODULES"
_initialized_modules = set()
_initializing = False
_lock = threading.RLock()


def _module_name(value):
    if not isinstance(value, str) or not value or not all(part.isidentifier() for part in value.split(".")):
        raise ValueError("A provider registration needs an importable module name")
    return value


def _declared_modules():
    modules = json.loads(os.environ.get(_ENVIRONMENT, "[]"))
    if not isinstance(modules, list):
        raise ValueError(_ENVIRONMENT + " must be a JSON list of module names")
    return tuple(dict.fromkeys(_module_name(name) for name in (*PROVIDER_MODULES, *modules)))


def _registry_snapshot():
    from . import csf, cute
    from .cute import builder

    return (cute, dict(cute._runtime_providers), cute._preferred_runtime,
            csf, dict(csf._CSF_PROVIDERS), builder, dict(builder._PROVIDERS))


def _restore_registries(snapshot):
    cute, runtime, preferred, csf, compressed, builder, compilers = snapshot
    for registry, saved in ((cute._runtime_providers, runtime), (csf._CSF_PROVIDERS, compressed),
                            (builder._PROVIDERS, compilers)):
        registry.clear()
        registry.update(saved)
    cute._preferred_runtime = preferred


def initialize_providers():
    """Load declared providers once per process before any provider is selected."""
    global _initializing
    with _lock:
        if _initializing:
            return
        _initializing = True
        try:
            for name in _declared_modules():
                if name in _initialized_modules:
                    continue
                snapshot = _registry_snapshot()
                try:
                    module = importlib.import_module(name)
                    register = getattr(module, "register", None)
                    if not callable(register):
                        raise TypeError("Provider module must expose register()")
                    register()
                except BaseException:
                    _restore_registries(snapshot)
                    raise
                _initialized_modules.add(name)
        finally:
            _initializing = False


def register_provider_module(name):
    """Register now and publish the importable declaration to future workers.

    Repeating the same declaration is harmless. Arbitrary parent-only callbacks
    cannot be replayed by Windows spawn; put them in this module's register().
    """
    name = _module_name(name)
    with _lock:
        previous = os.environ.get(_ENVIRONMENT)
        declarations = list(_declared_modules())
        if name not in declarations:
            declarations.append(name)
        os.environ[_ENVIRONMENT] = json.dumps(declarations, separators=(",", ":"))
        try:
            initialize_providers()
        except BaseException:
            if previous is None:
                os.environ.pop(_ENVIRONMENT, None)
            else:
                os.environ[_ENVIRONMENT] = previous
            raise
