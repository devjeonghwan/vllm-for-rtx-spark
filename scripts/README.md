# Source preparation

| Code | Location |
|---|---|
| Maintained kernels and Windows bindings | `src/kernels/` |
| Model and recipe integration | `src/recipes/` |
| Changes to upstream vLLM | `patches/vllm/` |
| Changes to FlashInfer | `patches/flashinfer/` |
| External source revisions and archive checksums | [sources.lock.json](../sources.lock.json) |

The maintained implementation includes code adapted from the myllmbox recipe. Original copyright and license notices remain with the code.

`build_vllm.py prepare` checks out and patches vLLM. During `build`, vLLM's CMake files fetch their pinned dependencies, apply the required Windows patches, and install the upstream Python/header components. The Windows ARM64 build compiles only the two core extensions selected in the manifest; optional native kernel targets remain disabled.

The additional ARM64 Python bindings and FlashInfer inputs are prepared separately from vLLM's CMake dependencies:

```powershell
python scripts/prepare_dependencies.py runtime
python scripts/prepare_dependencies.py flashinfer-native
python scripts/prepare_dependencies.py oss-harmony
python scripts/prepare_dependencies.py tiktoken
python scripts/prepare_dependencies.py llguidance
```

These fixed archives are downloaded to `.build/downloads/`, with Python dependencies in `.build/vendor/` and native build sources in `.build/sources/`. FlashInfer receives the patches above. This preparation does not compile or install packages.

For `prepare_dependencies.py`, use `--offline` with a populated archive cache and `--output` for a separate preparation directory. Prepared outputs are immutable; preserve the old output and its adjacent JSON receipt before preparing changed inputs.

Validated ARM64 Python bindings remain in `assets/python/`. No rebuild is needed to use them. The final vLLM wheel includes the prepared Python dependencies under `vllm/_rtx_spark/vendor/` to preserve their native dependency graph, together with upstream licenses.
