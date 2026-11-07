# Configuration

Bundled profiles are verified presets for the named checkpoints. Custom profiles
may select other weights, supported architectures, precision and resource policies;
their quality and performance are the user's responsibility. Configuration does
not add missing model implementations, loaders or native kernels. Loader/provider
format checks and ABI, alignment, state and stream-lifetime requirements still apply.
Checkpoint files are never edited by these overrides.

Use normal vLLM settings for model/revision, `hf-overrides` (including RoPE),
KV precision and excluded layers, context length, batch size, Graph/compilation,
sampling and backend selection. Target and external draft configuration are
separate: the Qwen recipe does not copy target RoPE overrides into DFlash2.

Project settings live under `additional-config.rtx_spark`. `model` selects a recipe;
omitting it enables unambiguous automatic selection. Set `model: "none"` to disable
recipe defaults and hooks while retaining native dependency/provider initialization.
Recipe-specific options cannot accompany `"none"`. Selection is preserved in workers.
Opt-out also removes Qwen's target-to-MTP override transform and resource defaults;
supply the required loader/resource settings and valid target/draft context limits
explicitly when constructing a custom profile.

## Qwen HEAD2

| Key | Default | Meaning |
|---|---|---|
| `draft_head2` | `true` | Enable the existing two-stage MTP draft head |
| `draft_head2_basis` | Bundled Qwen basis | Basis file path; hidden size and rank are checked when loaded |
| `draft_head2_k` | `2048` | Positive candidate count, capped at the actual vocabulary size |
| `draft_head2_scorer` | `fp8` | Existing `fp8` or `fp4` scorer implementation |

For the three detailed settings, explicit config takes precedence over
`MBX_DRAFT_HEAD2`, `MBX_DRAFT_HEAD2_K`, `MBX_DRAFT_HEAD2_SCORER`, then recipe defaults.
Relative basis paths use the process working directory and are resolved before
worker startup. Explicit detailed settings conflict with `draft_head2: false`.
Resolved settings are serialized for workers and logged with their sources.
Alternative bases/scorers are custom configurations; the default remains FP8.

## Resource and depth policy overrides

| Setting | Interface |
|---|---|
| Runtime cache root | `VLLM_RTX_SPARK_CACHE` |
| PLE NVMe cache directory | `MBX_PLE_NVME_DIR` |
| PLE I/O threads / queue depth | `MBX_PLE_NVME_THREADS` / `MBX_PLE_QD` |
| PLE prefetch / async capacity | `MBX_PLE_ASYNC_PREFETCH` / `MBX_PLE_ASYNC_MAX` |
| Dynamic MTP policy JSON | `MBX_DEPTH_CFG` |
| Minimum draft depth | `rtx_spark.draft_min_depth` |

An explicit minimum copies the selected policy and changes only `min`; the source
JSON remains unchanged. Without an external policy, it uses the bundled policy.
An unreadable or malformed selected policy is an error when applying a minimum.
Existing native policy restrictions (dynamic mode, TP1, syncfree) still apply.

The existing `draft_fp8`, `cute_mode`, `csf_backend`, `breakable_cudagraph`,
`fused_fp8` and DFlash options remain recipe-specific choices. Gemma's preset
does not whitelist checkpoint dimensions or a global quantization-group layout;
the actual upstream loader and selected kernels validate formats. Its execution
restrictions remain in place. FlashInfer's unused FLA metadata is handled
automatically by the resolved backend and has no user option.
