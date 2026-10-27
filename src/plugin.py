"""Official vLLM general-plugin entry point for the bundled integration."""


def register():
    """Register CPU metadata in each process; repeated calls are harmless."""
    from .recipes import register_builtin_models, register_architectures
    from .runtime.bootstrap import initialize

    register_builtin_models()
    initialize()
    register_architectures()
