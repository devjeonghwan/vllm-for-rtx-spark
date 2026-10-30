"""Qwen3.8 configuration with the FP8 drafter and two-stage head by default."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ...optimizations import Option, resolve_options, validate_options
from ...optimizations.backends import CUTE_MODE, CSF_BACKEND

ARCHITECTURES = ("Qwen4ExpForConditionalGeneration",)
# vLLM already supplies this architecture; do not replace its model class.
MODEL_CLASSES = {}


def matches(arguments):
    from ...runtime.vllm import get_adapter

    return "qwen3.8-flash-next-fp4-local" in get_adapter().model_name(arguments).lower()


def _validate_options(options):
    resolved = resolve_options(options, OPTIONS)
    validate_options(None, resolved, OPTIONS)
    return resolved


def _validate_draft_min_depth(arguments, options):
    if "draft_min_depth" not in options:
        return
    from ...runtime.vllm import get_adapter

    maximum = get_adapter().speculative_tokens(arguments)
    if isinstance(maximum, int) and options["draft_min_depth"] > maximum:
        raise ValueError("rtx_spark.draft_min_depth exceeds speculative_config.num_speculative_tokens")


def _configure_draft_min_depth(options):
    if "draft_min_depth" not in options:
        return
    from ...runtime import cache_root, resources_root

    policy = json.loads((resources_root() / "recipes/qwen3_8/mtp-depth.json").read_text(encoding="utf-8"))
    policy["min"] = options["draft_min_depth"]
    payload = (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    directory = (cache_root() / "recipe-configs").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ("qwen3_8-mtp-" + digest + ".json")

    def verify():
        if destination.read_bytes() != payload:
            raise RuntimeError("Cached MTP policy has changed: " + str(destination))

    if destination.exists():
        verify()
    else:
        # Publish a complete file before making its path visible to workers.
        # Windows rename refuses an existing destination; racing identical
        # policies reuse that immutable file rather than editing it in place.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".mtp-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                temporary.rename(destination)
            except FileExistsError:
                verify()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    # Explicit YAML takes precedence; an omitted option never changes this env.
    os.environ["MBX_DEPTH_CFG"] = str(destination)


def _validate_minimum(arguments, minimum):
    if minimum < 1:
        raise ValueError("rtx_spark.draft_min_depth must be at least 1")
    _validate_draft_min_depth(arguments, {"draft_min_depth": minimum})


def _apply_minimum(arguments, minimum):
    _configure_draft_min_depth({"draft_min_depth": minimum})


def _apply_draft_fp8(arguments, enabled):
    os.environ["MBX_DRAFT_FP8"] = "1" if enabled else "0"


def _apply_draft_head2(arguments, enabled):
    from ...runtime import resources_root

    if enabled:
        os.environ["MBX_DRAFT_HEAD2"] = str(resources_root() / "recipes" / "qwen3_8" / "qwen3.8-head2.basis.pt")
        os.environ["MBX_DRAFT_HEAD2_K"] = "2048"
        os.environ["MBX_DRAFT_HEAD2_SCORER"] = "fp8"
    else:
        os.environ.pop("MBX_DRAFT_HEAD2", None)


OPTIONS = (
    # Omission preserves a caller's external policy; the bundled policy is min3.
    Option("draft_min_depth", int, apply=_apply_minimum, validate=_validate_minimum),
    CUTE_MODE,
    CSF_BACKEND,
    Option("draft_fp8", bool, True, apply=_apply_draft_fp8),
    Option("draft_head2", bool, True, apply=_apply_draft_head2),
)


def _prepare_environment():
    from ...runtime import cache_root, resources_root

    root = resources_root()
    defaults = {
        "MBX_LOADER_PREPARE": "0", "MBX_PLE_MMAP": "1", "MBX_PLE_MMAP_MODE": "nvme",
        "MBX_PLE_ASYNC": "1", "MBX_PLE_ASYNC_MAX": "32768", "MBX_PLE_NVME_THREADS": "4",
        "MBX_PLE_QD": "256", "MBX_PLE_ASYNC_PREFETCH": "1", "MBX_PLE_MMAP_PREWARM": "0",
        "MBX_LOADER_EMPTY_CACHE": "1", "MBX_PLE_PREFILL_CKPT": "1", "MBX_DEPTH_SYNCFREE": "1",
        "VLLM_GDN_DECODE_KERNEL": "cuda", "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD": "0",
        "MBX_CSF_VISUAL_BF16": "0", "MBX_OPT_QUANT_HC_DOWN": "0",
        "VLLM_DISABLED_KERNELS": "FlashInferCutedslMxfp8LinearKernel,FlashInferTrtllmMxfp8LinearKernel,DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel",
        "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
    }
    paths = {
        "MBX_PLE_LIB": root / "native/mbx_ple_nvme.dll",
        "MBX_PLE_ASYNC_LIB": root / "native/mbx_ple_async.dll",
        "MBX_DEPTH_LIB": root / "native/mbx_mtp.dll",
        "MBX_DEPTH_CFG": root / "recipes/qwen3_8/mtp-depth.json",
        "MBX_DEPTH_STATS": cache_root() / "depth-stats.json",
        "MBX_DEPTH_STEPTIME": cache_root() / "depth-steptime.json",
        "MBX_PLE_NVME_DIR": cache_root() / "ple-nvme",
    }
    for name, value in {**defaults, **paths}.items():
        os.environ.setdefault(name, str(value))


def prepare(args, options):
    """Prepare Qwen's environment before declared optimization callbacks run."""
    options = _validate_options(options)
    _validate_draft_min_depth(args, options)
    from ...runtime.bootstrap import initialize
    from ...runtime.vllm import get_adapter

    initialize()
    _prepare_environment()
    get_adapter().wrap_hf_overrides(args, TargetAndDraftOverrides)


def validate(config, options):
    _validate_options(options)
    _validate_draft_min_depth(config, options)
    if options.get("model") == "qwen3.8-flash-next":
        from ...runtime.vllm import get_adapter

        if get_adapter().tensor_parallel_size(config) != 1:
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
