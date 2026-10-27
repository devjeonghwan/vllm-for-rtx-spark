"""Replaceable CSF providers behind the original Torch operator ABI."""

from dataclasses import dataclass

from ...runtime import artifact


@dataclass(frozen=True)
class CsfProvider:
    """A provider supplies the same Torch op and three-scale-plane ABI."""

    name: str
    library: str
    op_namespace: str = "mbx_marlin_csf_native"
    op_name: str = "moe_wna16_marlin_gemm_csf"
    cubin: str | None = None


_CSF_PROVIDERS = {
    "cubin": CsfProvider("cubin", "csf_host", cubin="csf_cubin"),
}


def register_csf_provider(provider):
    """Register local metadata; use a provider module to replay it in workers."""
    if not isinstance(provider, CsfProvider):
        raise TypeError("Expected CsfProvider")
    if provider.name in _CSF_PROVIDERS:
        raise ValueError("CSF provider is already registered: " + provider.name)
    _CSF_PROVIDERS[provider.name] = provider


def resolve_csf_provider(name="cubin"):
    from ..providers import initialize_providers

    initialize_providers()
    try:
        provider = _CSF_PROVIDERS[name]
    except KeyError:
        raise ValueError("CSF provider is not installed: " + name) from None
    library, _ = artifact(provider.library)
    result = dict(provider=provider, library=library)
    if provider.cubin:
        cubin, _ = artifact(provider.cubin)
        result.update(cubin=cubin)
    return result
