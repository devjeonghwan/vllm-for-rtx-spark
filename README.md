# vLLM for RTX Spark

vLLM builds and model recipes for NVIDIA RTX Spark on Windows ARM64.

Currently supported: [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on RTX Spark N1X, with 96 GB GPU / 32 GB CPU memory and NVMe PLE. Based on vLLM 0.32.0 and [myllmbox's v5.3.1 recipe](https://github.com/myllmbox/qwen38-flash-next-recipe).

## Benchmarks

### 1. DGX Spark (Original MyLLMBox v5.3 Recipe) vs RTX Spark (This project)

| Workload | RTX Spark, 1 request | DGX Spark, 1 request | RTX Spark, 16 requests | DGX Spark, 16 requests |
|---|---:|---:|---:|---:|
| Mixed | 51.45 | 57.58 | 194.90 | 187.78 |
| Code | 50.35 | 57.23 | 249.65 | 268.15 |
| JSON schema | 41.63 | 44.90 | 215.83 | 222.28 |
| Prose | 38.15 | 44.53 | 190.28 | 212.20 |

### 2. RTX Spark: NVIDIA recipe (llama.cpp) vs This project

NVIDIA figures are from its [published Windows recipe](https://nvidia.github.io/local-ai-inference-recipes/hardware/rtx-spark/index.html?model=qwen%2Fqwen3.8-flash-next&os=windows&backend=llama-cpp&memory=78&precision=iq4-xs&variant=vision-mtp&memoryConfig=128). Our measurements use `qwen3.8-fast.yaml` on the same RTX Spark N1X as above.

| Benchmark | NVIDIA llama.cpp, IQ4_XS | This project, NVFP4 |
|---|---:|---:|
| AA-AgentPerf-local completion, full replay | 1,118.16 s | **847.11 s** |
| SPEED-Bench coding decode, all 80 conversations | 35.37 tok/s | **53.20 tok/s** |

## Install

Requirements: ARM64 Python 3.13, an NVIDIA driver compatible with CUDA 13.4, and the model checkpoint. Allow several GB for dependencies and at least 32 GB of free NVMe space for the PLE cache.

Download the Windows ARM64 ZIP from [Releases](https://github.com/devjeonghwan/vllm-for-rtx-spark/releases), extract it, and run:

```powershell
.\scripts\install.ps1 -Python python
```

This creates `.venv` and installs the bundled vLLM wheel and its dependencies. PyTorch and FlashAttention are downloaded from NVIDIA.

## Run

```powershell
.\.venv\Scripts\vllm.exe serve C:\models\Qwen3.8-Flash-Next-FP4-Local --config configs\qwen3.8.yaml
```

The API is available at `http://127.0.0.1:8000/v1`, with model name `Qwen/Qwen3.8-Flash-Next`. The first start prepares the PLE cache and can take several minutes.

- `configs/qwen3.8.yaml`: BF16 drafter and full draft vocabulary.
- `configs/qwen3.8-fast.yaml`: FP8 drafter and two-stage draft head.

Both files accept vLLM command-line overrides.

## Build

Use a Git checkout or the `source.zip` asset in a short directory path. Install Git, native ARM64 Python 3.13, Visual Studio ARM64 C++ tools, and CUDA Toolkit 13.4.

```powershell
python -m pip install --only-binary=:all: -r requirements\runtime.txt -r requirements\vendor.txt -r requirements\build.txt
.\scripts\build-vllm.ps1 -Python python
```

The build fetches the pinned upstream vLLM, applies the patches, and produces one wheel containing the runtime and kernel assets.

Prebuilt native assets are included. To rebuild them, use `scripts/build-native.ps1 -Install` and `scripts/build_harmony.py` before building the vLLM wheel. The Rust bindings require the `aarch64-pc-windows-msvc` toolchain.

For a CuTe kernel rebuild, install `src/kernels/cute/compiler/requirements.txt` in the compiler's Python environment. Copy `src/kernels/cute/compiler/settings.example.json` to `local-build.json`, set the tool paths, and run:

```powershell
python scripts\build_kernels.py --provider source --settings local-build.json --output .build\kernels-new --install
```

The current provider uses an x64 CuTe compiler during builds and ARM64 libraries during inference. CuTe and CSF providers can be replaced independently. CSF currently includes the recipe's original cubin; its device source is unavailable.

## Layout

- `src/runtime/`: Windows integration and native host code.
- `src/kernels/`: kernel providers, sources, and compiler support.
- `src/recipes/`: model-specific integration.
- `assets/`: prebuilt kernels, libraries, and model support data.
- `third_party/`: vendored dependencies.
- `patches/vllm/`: versioned upstream patches.
- `configs/`: model launch settings.

CPU tests: `python -m unittest discover -s tests -p '*_test.py'`. Release build and GPU validation details are in `build-info.json`. Model weights are obtained separately. This is a preview release.

See `sources.lock.json` and `THIRD_PARTY_NOTICES.md` for source revisions and credits.
