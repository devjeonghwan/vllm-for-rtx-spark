# vLLM for RTX Spark

vLLM builds and model recipes for NVIDIA RTX Spark on Windows ARM64.

Currently supported: [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on RTX Spark N1X, with 96 GB GPU / 32 GB CPU memory and NVMe PLE. Based on vLLM 0.32.0 and [myllmbox's v5.3.1 recipe](https://github.com/myllmbox/qwen38-flash-next-recipe).

## Benchmarks

Measured with the earlier 27 GB KV configuration, FP8 drafter and HEAD2, before min3 became the default.

### 1. DGX Spark (Original MyLLMBox v5.3 Recipe) vs RTX Spark (This project)

| Workload | RTX Spark, 1 request | DGX Spark, 1 request | RTX Spark, 16 requests | DGX Spark, 16 requests |
|---|---:|---:|---:|---:|
| Mixed | 51.45 | 57.58 | 194.90 | 187.78 |
| Code | 50.35 | 57.23 | 249.65 | 268.15 |
| JSON schema | 41.63 | 44.90 | 215.83 | 222.28 |
| Prose | 38.15 | 44.53 | 190.28 | 212.20 |

### 2. RTX Spark: NVIDIA recipe (llama.cpp) vs This project

NVIDIA figures are from its [published Windows recipe](https://nvidia.github.io/local-ai-inference-recipes/hardware/rtx-spark/index.html?model=qwen%2Fqwen3.8-flash-next&os=windows&backend=llama-cpp&memory=78&precision=iq4-xs&variant=vision-mtp&memoryConfig=128). Our measurements use the RTX Spark N1X described above.

| Benchmark | NVIDIA llama.cpp, IQ4_XS | This project, NVFP4 |
|---|---:|---:|
| AA-AgentPerf-local completion, full replay | 1,118.16 s | **847.11 s** |
| SPEED-Bench coding decode, all 80 conversations | 35.37 tok/s | **53.20 tok/s** |

## Install

Requirements: ARM64 Python 3.13, an NVIDIA driver compatible with CUDA 13.4, and the model checkpoint. Allow several GB for dependencies and at least 32 GB of free NVMe space for the PLE cache.

On the **RTX Spark N1X 128 GB**, set dedicated GPU memory to **96 GB**. This leaves about **32 GB for Windows**.

Set a custom Windows paging file:

- **Initial size (MB): 98304**
- **Maximum size (MB): 131072**

Open **Advanced system settings → Performance → Settings → Advanced → Virtual memory → Change**. Turn off **Automatically manage paging file size for all drives**. Select your NVMe drive and choose **Custom size**. Enter the values above, click **Set**, then **OK**, and restart Windows.

CUDA allocations also use Windows system commit. A small paging file can cause OOM errors even when GPU memory is still free. See the [memory issue report](https://forums.developer.nvidia.com/t/rtx-spark-on-windows-cuda-allocations-consume-system-commit-even-with-96-gib-dedicated-gpu-memory/386453).

Download the Windows ARM64 ZIP from [Releases](https://github.com/devjeonghwan/vllm-for-rtx-spark/releases), extract it, and run:

```powershell
.\scripts\install.ps1 -Python python
```

This creates `.venv` and installs the bundled vLLM wheel and its dependencies. PyTorch and FlashAttention are downloaded from NVIDIA.

## Run

```powershell
.\.venv\Scripts\vllm.exe serve C:\models\Qwen3.8-Flash-Next-FP4-Local --config configs\qwen3.8-flash-next\gpu-94gb.yaml
```

The API is available at `http://127.0.0.1:8000/v1`, with model name `Qwen/Qwen3.8-Flash-Next`. The first start prepares the PLE cache and can take several minutes.

All three profiles target the same GPU 96 GB / CPU 32 GB allocation. They use FP8 drafter projections, HEAD2, dynamic draft depth 3–7, FP8 KV, NVMe PLE, decode graphs, a 409,600-token context and up to 16 requests. Only the KV cache budget changes.

Config names indicate approximate total GPU memory use, rounded up for headroom.

| Config | KV budget | Use |
|---|---:|---|
| [gpu-90gb.yaml](configs/qwen3.8-flash-next/gpu-90gb.yaml) | 12 GB | More room for Windows and other applications |
| [gpu-94gb.yaml](configs/qwen3.8-flash-next/gpu-94gb.yaml) | 16 GB | Recommended starting point |
| [gpu-96gb.yaml](configs/qwen3.8-flash-next/gpu-96gb.yaml) | 18 GB | More room for concurrent long contexts; close to the 96 GB allocation |

Model allocations, activations and Windows add to these budgets; no profile guarantees that WDDM shared memory will stay unused. The profiles accept vLLM command-line overrides. `draft_min_depth` under `additional-config.rtx_spark` overrides the default minimum of 3.

## Build

Use a Git checkout or the `source.zip` asset in a short directory path. Install Git, native ARM64 Python 3.13, Visual Studio ARM64 C++ tools, and CUDA Toolkit 13.4.

```powershell
python -m pip install --only-binary=:all: -r requirements\runtime.txt -r requirements\vendor.txt -r requirements\build.txt
.\scripts\build-vllm.ps1 -Python python
```

Currently, project is pinned to vLLM 0.32.0 at commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. The build applies the patches and copies the sources listed in `patches/vllm/manifest.json`, then produces one wheel containing the runtime and kernel assets.

Prebuilt native assets are included. To rebuild them, use `scripts/build-native-assets.ps1 -Install` and `scripts/build_harmony.py` before building the vLLM wheel. The Rust bindings require the `aarch64-pc-windows-msvc` toolchain.

For a CuTe kernel rebuild, install `src/kernels/cute/compiler/requirements.txt` in the compiler's Python environment. Copy `src/kernels/cute/compiler/settings.example.json` to `local-build.json`, set the tool paths, and run:

```powershell
python scripts\build_kernels.py --provider source --settings local-build.json --output .build\kernels-new --install
```

The current provider uses an x64 CuTe compiler during builds and ARM64 libraries during inference. CuTe and CSF providers can be replaced independently. CSF currently includes the recipe's original cubin; its device source is unavailable.

## Layout

- `src/runtime/`: Windows integration and native host code.
- `src/runtime/vllm/`: this release's vLLM adapter.
- `src/kernels/`: kernel providers, sources, and compiler support.
- `src/recipes/`: model-specific integration and helpers.
- `src/optimizations/`: shared option declarations and application callbacks.
- `assets/`: prebuilt kernels, libraries, and model support data.
- `third_party/`: external source dependencies, including the added vLLM kernel sources.
- `patches/vllm/`: vLLM patches and their manifest.
- `configs/`: model launch settings.

The wheel registers `rtx_spark` through `vllm.general_plugins`. Windows startup and recipe configuration also use explicit vLLM hooks. See [development notes](docs/development.md) for updating vLLM and adding models or options.

CPU tests: `python -m unittest discover -s tests -p '*_test.py'`. Release build and GPU validation details are in `build-info.json`. Model weights are obtained separately. This is a preview release.

See `sources.lock.json` and `THIRD_PARTY_NOTICES.md` for source revisions and credits.
