"""Kernel build entry point shared by tools and model integrations."""


def build_kernels(output, **options):
    from .cute.builder import build

    return build(output, **options)
