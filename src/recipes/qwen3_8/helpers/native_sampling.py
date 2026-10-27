"""Windows library boundary for the unchanged original FlashInfer sampler."""
import functools
from pathlib import Path

HERE = Path(__file__).resolve().parent
from ....runtime import resources_root
LIBRARY = resources_root() / "native/flashinfer_sampling.dll"
FUNCTIONS = (
    "softmax", "sampling_from_probs", "sampling_from_logits", "top_k_sampling_from_probs",
    "min_p_sampling_from_probs", "top_p_sampling_from_probs", "top_k_top_p_sampling_from_probs",
    "top_k_renorm_probs", "top_p_renorm_probs", "top_k_mask_logits", "chain_speculative_sampling",
)


@functools.cache
def load_sampling_module():
    """Cold load only. Original get_sampling_module keeps its normal cache."""
    import tvm_ffi
    raw = LIBRARY.read_bytes()
    pe = int.from_bytes(raw[0x3c:0x40], "little")
    if raw[:2] != b"MZ" or raw[pe:pe+4] != b"PE\0\0" or raw[pe+4:pe+6] != b"\x64\xaa":
        raise RuntimeError("Native sampling library must be ARM64 PE")
    module = tvm_ffi.load_module(str(LIBRARY))
    for name in FUNCTIONS:
        if not callable(getattr(module, name)):
            raise RuntimeError("Original sampling function unavailable: " + name)
    return module


class NativeSamplingSpec:
    name = "sampling"

    def build_and_load(self):
        return load_sampling_module()


def gen_native_sampling_spec():
    return NativeSamplingSpec()
