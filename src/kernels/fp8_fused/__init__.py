"""Opt-in FP8 per-token/channel epilogue provider; no GPU imports at registration."""

UPSTREAM_REPOSITORY = "https://github.com/Monash-Neurology-AI-inference-lab/rtx-spark-fp8-kernel"
UPSTREAM_REVISION = "b2de7164a65d1a7cff4ae328236eaf13cc568c13"


def enabled(config):
    """Read the engine's validated option, without process-global selection state."""
    additional = getattr(config, "additional_config", None) or {}
    options = additional.get("rtx_spark", {})
    value = options.get("fused_fp8", options.get("optimizations", {}).get("fused_fp8", False))
    return value is True
