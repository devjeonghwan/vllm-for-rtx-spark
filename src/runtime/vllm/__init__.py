"""Access the vLLM compatibility adapter shipped with this release."""


def get_adapter():
    from . import adapter

    return adapter
