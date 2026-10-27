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
    """Register a source-built replacement before model weights are prepared."""
    if not isinstance(provider, CsfProvider):
        raise TypeError("Expected CsfProvider")
    if provider.name in _CSF_PROVIDERS:
        raise ValueError("CSF provider is already registered: " + provider.name)
    _CSF_PROVIDERS[provider.name] = provider


def resolve_csf_provider(name="cubin"):
    try:
        provider = _CSF_PROVIDERS[name]
    except KeyError:
        raise ValueError("CSF provider is not installed: " + name) from None
    library, record = artifact(provider.library)
    result = dict(provider=provider, library=library, library_sha256=record["sha256"])
    if provider.cubin:
        cubin, cubin_record = artifact(provider.cubin)
        result.update(cubin=cubin, cubin_sha256=cubin_record["sha256"])
    return result
