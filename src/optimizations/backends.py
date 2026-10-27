"""Reusable provider choices; recipes opt in to these declarations."""

import os

from . import Option


def _configure_cute(arguments, mode):
    from ..kernels.cute import configure

    configure(mode)


def _configure_csf(arguments, selected):
    from ..kernels.csf import resolve_csf_provider

    provider = resolve_csf_provider(selected)
    os.environ["MBX_MARLIN_CSF"] = "1"
    os.environ["MBX_MARLIN_CSF_PROVIDER"] = selected
    os.environ["MBX_MARLIN_CSF_NATIVE_LIB"] = str(provider["library"])
    if "cubin" in provider:
        os.environ["MBX_MARLIN_CSF_CUBIN"] = str(provider["cubin"])
    else:
        os.environ.pop("MBX_MARLIN_CSF_CUBIN", None)


CUTE_MODE = Option("cute_mode", str, "auto", apply=_configure_cute)
CSF_BACKEND = Option("csf_backend", str, "cubin", apply=_configure_csf)
