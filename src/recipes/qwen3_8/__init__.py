"""Qwen3.8 target/draft configuration compatibility."""

import copy
import os


def matches(arguments):
    model_config = getattr(arguments, "model_config", None)
    model = getattr(arguments, "model", getattr(model_config, "model", ""))
    return "qwen3.8-flash-next-fp4-local" in str(model).lower()


def _validate_options(options):
    allowed = {"model", "csf_backend", "draft_fp8", "draft_head2", "cute_mode"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError("Unknown RTX Spark setting: " + ", ".join(sorted(unknown)))
    for key in ("draft_fp8", "draft_head2"):
        if key in options and not isinstance(options[key], bool):
            raise TypeError("rtx_spark." + key + " must be a boolean")
    for key in ("model", "csf_backend", "cute_mode"):
        if key in options and not isinstance(options[key], str):
            raise TypeError("rtx_spark." + key + " must be a string")


def prepare(args, options):
    """Apply the verified Qwen recipe before its target/draft config is built."""
    _validate_options(options)
    from ...runtime.bootstrap import initialize
    from ...runtime import resources_root
    from ...kernels.csf import resolve_csf_provider

    initialize()
    os.environ.setdefault("VLLM_DISABLED_KERNELS", "FlashInferCutedslMxfp8LinearKernel,FlashInferTrtllmMxfp8LinearKernel,DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel")
    os.environ.setdefault("VLLM_MARLIN_USE_ATOMIC_ADD", "1")
    from ...kernels.cute import configure

    configure(options.get("cute_mode", "auto"))
    selected = str(options.get("csf_backend", "cubin"))
    provider = resolve_csf_provider(selected)
    os.environ["MBX_MARLIN_CSF"] = "1"
    os.environ["MBX_MARLIN_CSF_PROVIDER"] = selected
    os.environ["MBX_MARLIN_CSF_NATIVE_LIB"] = str(provider["library"])
    os.environ["MBX_MARLIN_CSF_NATIVE_SHA256"] = provider["library_sha256"]
    if "cubin" in provider:
        os.environ["MBX_MARLIN_CSF_CUBIN"] = str(provider["cubin"])
    else:
        os.environ.pop("MBX_MARLIN_CSF_CUBIN", None)
    os.environ["MBX_DRAFT_FP8"] = "1" if options.get("draft_fp8", False) else "0"
    if options.get("draft_head2", False):
        os.environ["MBX_DRAFT_HEAD2"] = str(resources_root() / "recipes" / "qwen3_8" / "qwen3.8-head2.basis.pt")
        os.environ["MBX_DRAFT_HEAD2_K"] = "2048"
        os.environ["MBX_DRAFT_HEAD2_SCORER"] = "fp8"
    else:
        os.environ.pop("MBX_DRAFT_HEAD2", None)
    overrides = getattr(args, "hf_overrides", None)
    if isinstance(overrides, dict) and overrides.get("text_config"):
        args.hf_overrides = TargetAndDraftOverrides(overrides)


def validate(config, options):
    _validate_options(options)
    if options.get("model") == "qwen3.8-flash-next":
        if config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("Qwen3.8 on RTX Spark is currently tested with TP1")


class TargetAndDraftOverrides:
    """Picklable equivalent of the recipe's target+draft YaRN transform."""

    def __init__(self, overrides):
        self.overrides = copy.deepcopy(overrides)

    def __call__(self, config):
        text = getattr(config, "text_config", None)
        if text is None:
            if not hasattr(config, "num_attention_heads"):
                return config
            text = config
        fields = copy.deepcopy(self.overrides.get("text_config", {}))
        text.update(fields)
        for name, value in self.overrides.items():
            if name != "text_config":
                setattr(config, name, copy.deepcopy(value))
        return config
